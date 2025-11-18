#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot training curves from results.csv for report
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_training_curves(csv_path, output_dir='report_figures'):
    """
    Plot training and validation loss curves
    
    Args:
        csv_path: Path to results.csv
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} epochs of training data")
    print(f"\nFinal metrics:")
    print(f"  Train Loss: {df['train_loss'].iloc[-1]:.6f}")
    print(f"  Val Loss: {df['val_loss'].iloc[-1]:.6f}")
    print(f"  Final LR: {df['lr'].iloc[-1]:.6f}")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot 1: Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'loss_curves.png'}")
    plt.close()
    
    # Plot 2: Learning rate schedule
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['lr'], linewidth=2, color='red', marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule (Cosine Annealing)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'lr_schedule.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'lr_schedule.png'}")
    plt.close()
    
    # Plot 3: Combined view
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Loss on top
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # LR on bottom
    ax2.plot(df['epoch'], df['lr'], linewidth=2, color='red')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'training_summary.png'}")
    plt.close()
    
    # Summary statistics
    print(f"\n{'='*50}")
    print("Training Statistics Summary")
    print(f"{'='*50}")
    print(f"Total Epochs: {len(df)}")
    print(f"Best Train Loss: {df['train_loss'].min():.6f} (Epoch {df['train_loss'].idxmin() + 1})")
    print(f"Best Val Loss: {df['val_loss'].min():.6f} (Epoch {df['val_loss'].idxmin() + 1})")
    print(f"Final Train Loss: {df['train_loss'].iloc[-1]:.6f}")
    print(f"Final Val Loss: {df['val_loss'].iloc[-1]:.6f}")
    print(f"{'='*50}")


def plot_sample_images(image_dir='generated_images', output_path='report_figures/sample_generations.png', num_samples=10):
    """
    Create a grid of sample generated images for report
    
    Args:
        image_dir: Directory containing generated images
        output_path: Path to save the sample grid
        num_samples: Number of samples to show
    """
    from PIL import Image
    import numpy as np
    
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select evenly spaced images
    image_files = sorted(list(image_dir.glob('[0-9][0-9][0-9][0-9][0-9].png')))
    if len(image_files) == 0:
        print("Error: No generated images found!")
        return
    
    # Select samples
    indices = np.linspace(0, len(image_files)-1, num_samples, dtype=int)
    selected_files = [image_files[i] for i in indices]
    
    # Create grid
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i, (ax, img_path) in enumerate(zip(axes, selected_files)):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f'Sample {indices[i]+1}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Generated MNIST Samples', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved sample images: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training curves for report')
    parser.add_argument('--results', type=str, default='runs/ddpm_mnist/results.csv',
                       help='Path to results.csv')
    parser.add_argument('--output', type=str, default='report_figures',
                       help='Output directory for plots')
    parser.add_argument('--images', type=str, default='generated_images',
                       help='Directory containing generated images')
    parser.add_argument('--samples', action='store_true',
                       help='Also create sample images grid')
    
    args = parser.parse_args()
    
    # Check if results exist
    if not Path(args.results).exists():
        print(f"Error: Results file not found at {args.results}")
        print("Please train the model first using: python train.py")
        return
    
    # Plot training curves
    print("Plotting training curves...")
    plot_training_curves(args.results, args.output)
    
    # Plot sample images
    if args.samples:
        print("\nCreating sample images grid...")
        plot_sample_images(args.images, Path(args.output) / 'sample_generations.png')
    
    print("\n" + "="*50)
    print("All plots generated successfully!")
    print("="*50)
    print(f"\nGenerated files in {args.output}/:")
    print("  - loss_curves.png")
    print("  - lr_schedule.png")
    print("  - training_summary.png")
    if args.samples:
        print("  - sample_generations.png")
    print("\nYou can now insert these figures into your report!")


if __name__ == '__main__':
    main()