# hooks.py
# DDPM implementation for MNIST image generation
from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


# ===================== Diffusion Components =====================
class SinusoidalPositionEmbeddings(nn.Module):
    """Time step encoding using sinusoidal embeddings"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.GroupNorm(8, out_channels)
        
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        
        # Add time embedding
        t_emb = self.act(self.time_mlp(t_emb))
        h = h + t_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        
        # Compute attention
        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(C), dim=-1)
        h = torch.bmm(attn, v)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        
        return x + h


class UNet(nn.Module):
    """U-Net architecture for denoising"""
    def __init__(self, in_channels=3, model_channels=64, out_channels=3, 
                 num_res_blocks=2, attention_resolutions=(1,)):
        super().__init__()
        
        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Encoder
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        self.down1 = nn.ModuleList([
            ResidualBlock(model_channels, model_channels, time_emb_dim)
            for _ in range(num_res_blocks)
        ])
        self.down1_pool = nn.Conv2d(model_channels, model_channels * 2, 3, stride=2, padding=1)
        
        self.down2 = nn.ModuleList([
            ResidualBlock(model_channels * 2, model_channels * 2, time_emb_dim)
            for _ in range(num_res_blocks)
        ])
        self.down2_pool = nn.Conv2d(model_channels * 2, model_channels * 4, 3, stride=2, padding=1)
        
        # Bottleneck with attention
        self.bottleneck = nn.ModuleList([
            ResidualBlock(model_channels * 4, model_channels * 4, time_emb_dim),
            AttentionBlock(model_channels * 4),
            ResidualBlock(model_channels * 4, model_channels * 4, time_emb_dim),
        ])
        
        # Decoder
        self.up2_upsample = nn.ConvTranspose2d(model_channels * 4, model_channels * 2, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([
            ResidualBlock(model_channels * 4 if i == 0 else model_channels * 2, model_channels * 2, time_emb_dim)
            for i in range(num_res_blocks)
        ])
        
        self.up1_upsample = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResidualBlock(model_channels * 2 if i == 0 else model_channels, model_channels, time_emb_dim)
            for i in range(num_res_blocks)
        ])
        
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Initial projection
        h = self.init_conv(x)
        
        # Encoder
        h1 = h
        for block in self.down1:
            h1 = block(h1, t_emb)
        
        h2 = self.down1_pool(h1)
        for block in self.down2:
            h2 = block(h2, t_emb)
        
        # Bottleneck
        h3 = self.down2_pool(h2)
        for block in self.bottleneck:
            if isinstance(block, AttentionBlock):
                h3 = block(h3)
            else:
                h3 = block(h3, t_emb)
        
        # Decoder with skip connections
        h = self.up2_upsample(h3)
        h = torch.cat([h, h2], dim=1)  # Concat: (B, 128, 14, 14) + (B, 128, 14, 14) = (B, 256, 14, 14)
        for i, block in enumerate(self.up2):
            h = block(h, t_emb)
        
        h = self.up1_upsample(h)
        h = torch.cat([h, h1], dim=1)  # Concat: (B, 64, 28, 28) + (B, 64, 28, 28) = (B, 128, 28, 28)
        for i, block in enumerate(self.up1):
            h = block(h, t_emb)
        
        return self.final_conv(h)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model"""
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # Define beta schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        """Calculate loss for training"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss

    def forward(self, x, t=None):
        """Training forward pass"""
        if t is None:
            t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, t)

    @torch.no_grad()
    def p_sample(self, x, t):
        """Single reverse diffusion step"""
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1, 1)
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, batch_size, img_size=(28, 28), device='cuda'):
        """Generate samples from noise"""
        self.eval()
        # Start from pure noise
        x = torch.randn(batch_size, 3, img_size[0], img_size[1], device=device)
        
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
        
        return x


# ===================== 1) Build Model =====================
def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Build DDPM model for image generation"""
    task = cfg.get("task", "generation")
    
    if task == "generation":
        model_channels = int(cfg.get("model_channels", 64))
        num_res_blocks = int(cfg.get("num_res_blocks", 2))
        timesteps = int(cfg.get("timesteps", 1000))
        
        unet = UNet(
            in_channels=3,
            model_channels=model_channels,
            out_channels=3,
            num_res_blocks=num_res_blocks
        )
        
        ddpm = DDPM(
            model=unet,
            timesteps=timesteps,
            beta_start=float(cfg.get("beta_start", 0.0001)),
            beta_end=float(cfg.get("beta_end", 0.02))
        )
        
        # Move schedule tensors to the same device as model
        return ddpm
    else:
        raise ValueError(f"Unknown task: {task}")


# ===================== 2) DataLoaders =====================
def build_dataloaders(cfg: Dict[str, Any],
                      fold_split: Optional[Dict[str, List[str]]] = None,
                      dist: bool = False) -> Dict[str, Any]:
    """Build MNIST dataloaders"""
    B = int(cfg.get("batch", 128))
    num_workers = int(cfg.get("workers", 4))
    data_path = cfg.get("data_path", "./data")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to RGB
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Load MNIST dataset
    train_dataset = MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )
    
    # Distributed samplers
    sampler_train = DistributedSampler(train_dataset, shuffle=True) if dist else None
    sampler_val = DistributedSampler(val_dataset, shuffle=False) if dist else None
    
    dl_train = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    dl_val = DataLoader(
        val_dataset,
        batch_size=B,
        shuffle=False,
        sampler=sampler_val,
        num_workers=num_workers,
        pin_memory=True
    )
    
    meta = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset)
    }
    
    return {"train": dl_train, "val": dl_val, "test": None, "meta": meta}


# ===================== 3) Evaluation =====================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Evaluate the diffusion model
    Returns validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, _ = batch
        else:
            x = batch
        
        x = x.to(device, non_blocking=True)
        
        # Random timesteps for evaluation
        t = torch.randint(0, model.timesteps, (x.shape[0],), device=device).long()
        
        # Calculate loss
        loss = model(x, t)
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / max(1, n_batches)
    
    metrics = {
        "val_loss": avg_loss,
        "map50": None,  # Not applicable for generation
        "map5095": None
    }
    
    return metrics, [], []