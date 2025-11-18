#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate images using trained DDPM model
Usage:
    python generate.py --checkpoint runs/ddpm_mnist/weights/best.pt --num_images 10000 --save_process
"""
import os
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from hooks import build_model
from tools.io import load_cfg


def denormalize(x):
    """Denormalize from [-1, 1] to [0, 255]"""
    x = (x + 1) / 2  # [-1, 1] -> [0, 1]
    x = (x * 255).clamp(0, 255)
    return x.to(torch.uint8)


@torch.no_grad()
def generate_images(model, num_images, output_dir, batch_size=100, device='cuda', 
                   save_process=False, num_process_samples=8, num_process_steps=8):
    """
    Generate images using trained DDPM model
    
    Args:
        model: Trained DDPM model
        num_images: Total number of images to generate
        output_dir: Directory to save generated images
        batch_size: Batch size for generation
        device: Device to use
        save_process: Whether to save diffusion process visualization
        num_process_samples: Number of samples to show in process (default: 8)
        num_process_steps: Number of timesteps to visualize (default: 8, including final)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    model = model.to(device)
    
    # Move diffusion schedule tensors to device
    if hasattr(model, 'betas'):
        model.betas = model.betas.to(device)
        model.alphas = model.alphas.to(device)
        model.alphas_cumprod = model.alphas_cumprod.to(device)
        model.alphas_cumprod_prev = model.alphas_cumprod_prev.to(device)
        model.sqrt_alphas_cumprod = model.sqrt_alphas_cumprod.to(device)
        model.sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod.to(device)
        model.posterior_variance = model.posterior_variance.to(device)
    
    num_batches = (num_images + batch_size - 1) // batch_size
    img_idx = 1
    
    # For saving diffusion process
    process_images = []
    if save_process:
        # Calculate which timesteps to save (evenly spaced)
        # We want num_process_steps images total (including final result)
        step_interval = model.timesteps // (num_process_steps - 1)
        save_timesteps = set([model.timesteps - 1 - i * step_interval for i in range(num_process_steps - 1)] + [0])
        print(f"Will save diffusion process at timesteps: {sorted(save_timesteps, reverse=True)}")
    
    print(f"Generating {num_images} images in batches of {batch_size}...")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
            
            # Start from pure noise
            x = torch.randn(current_batch_size, 3, 28, 28, device=device)
            
            # For first batch, optionally save diffusion process
            if save_process and batch_idx == 0:
                process_batch = []
                current_samples = min(num_process_samples, current_batch_size)
            
            # Reverse diffusion process
            for t in reversed(range(model.timesteps)):
                t_batch = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
                
                # Denoise one step
                betas_t = model.betas[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alphas_cumprod_t = model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_recip_alphas_t = torch.sqrt(1.0 / model.alphas[t]).view(-1, 1, 1, 1)
                
                model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * model.model(x, t_batch) / sqrt_one_minus_alphas_cumprod_t
                )
                
                if t == 0:
                    x = model_mean
                else:
                    posterior_variance_t = model.posterior_variance[t].view(-1, 1, 1, 1)
                    noise = torch.randn_like(x)
                    x = model_mean + torch.sqrt(posterior_variance_t) * noise
                
                # Save intermediate steps for first batch
                if save_process and batch_idx == 0 and t in save_timesteps:
                    process_batch.append(x[:current_samples].cpu().clone())
            
            # Store process images for visualization
            if save_process and batch_idx == 0:
                process_images = process_batch
            
            # Denormalize and save individual images
            x = denormalize(x)
            x = x.cpu().numpy().transpose(0, 2, 3, 1)  # BCHW -> BHWC
            
            for i in range(current_batch_size):
                img = Image.fromarray(x[i])
                img.save(output_dir / f"{img_idx:05d}.png")
                img_idx += 1
    
    print(f"✓ Generated {num_images} images in {output_dir}")
    
    # Save diffusion process visualization
    if save_process and process_images:
        print(f"Creating diffusion process visualization...")
        save_diffusion_process(process_images, output_dir / "diffusion_process.png")
        print(f"✓ Saved diffusion process visualization")


def save_diffusion_process(process_images, save_path):
    """
    Create a grid visualization of the diffusion process
    
    Args:
        process_images: List of tensors, each of shape [N, 3, 28, 28]
        save_path: Path to save the visualization
    
    Creates an 8x8 grid where:
    - Each row shows the same sample evolving through timesteps
    - Each column shows a specific timestep for all samples
    """
    num_steps = len(process_images)
    num_samples = process_images[0].shape[0]
    
    print(f"Creating grid: {num_samples} samples × {num_steps} timesteps")
    
    # Create a grid: num_samples rows × num_steps columns
    grid_h = num_samples * 28
    grid_w = num_steps * 28
    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for step_idx, step_imgs in enumerate(process_images):
        # Denormalize images
        step_imgs = denormalize(step_imgs).numpy().transpose(0, 2, 3, 1)  # [N, 28, 28, 3]
        
        for sample_idx in range(num_samples):
            row_start = sample_idx * 28
            col_start = step_idx * 28
            grid_img[row_start:row_start+28, col_start:col_start+28] = step_imgs[sample_idx]
    
    # Save the grid
    Image.fromarray(grid_img).save(save_path)
    print(f"Grid size: {grid_h}×{grid_w} pixels")


def main():
    parser = argparse.ArgumentParser(description='Generate images using trained DDPM')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (e.g., runs/ddpm_mnist/weights/best.pt)')
    parser.add_argument('--config', type=str, default='configs/exp.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='generated_images',
                       help='Output directory for generated images')
    parser.add_argument('--num_images', type=int, default=10000,
                       help='Number of images to generate (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for generation (default: 100)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda or cpu (default: cuda)')
    parser.add_argument('--save_process', action='store_true',
                       help='Save diffusion process visualization (8×8 grid for report)')
    parser.add_argument('--num_process_samples', type=int, default=8,
                       help='Number of samples to show in process visualization (default: 8)')
    parser.add_argument('--num_process_steps', type=int, default=8,
                       help='Number of timesteps to show in visualization (default: 8)')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first using: python train.py")
        return
    
    # Load config
    print(f"Loading config from {args.config}")
    cfg = load_cfg(args.config)
    
    # Build model
    print("Building DDPM model...")
    model = build_model(cfg)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Generate images
    print(f"\nGenerating {args.num_images} images...")
    if args.save_process:
        print(f"Will also create diffusion process visualization ({args.num_process_samples}×{args.num_process_steps} grid)")
    
    generate_images(
        model=model,
        num_images=args.num_images,
        output_dir=args.output,
        batch_size=args.batch_size,
        device=args.device,
        save_process=args.save_process,
        num_process_samples=args.num_process_samples,
        num_process_steps=args.num_process_steps
    )
    
    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)
    print(f"Generated images saved to: {args.output}/")
    print(f"Total images: {args.num_images}")
    if args.save_process:
        print(f"Diffusion process visualization: {args.output}/diffusion_process.png")
    print("\nNext steps:")
    print("1. Check the generated images visually")
    print("2. Calculate FID score:")
    print(f"   python -m pytorch_fid {args.output} data/MNIST/raw")
    print("="*60)


if __name__ == '__main__':
    main()