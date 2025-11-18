# 114-1_TAICA_cv_hw3
MNIST
.
├── train.py                 # Training entry point
├── hooks.py                 # Model, dataloader, and evaluation implementation
├── generate.py              # Image generation script
├── plot_training_curves.py  # Script to plot training curves
|
├── configs/
│   └── exp.yaml             # Configuration file
├── data/                    # MNIST dataset (auto-downloaded)
├── generated_images/        # Generated images will be saved here
├── plots/                   # Training curves will be saved here
├── runs/ddpm_mnist/         # Training outputs (checkpoints, logs, etc.)
|
└── requirements.txt         # Python dependencies

## Environment Setup
```bash
# Install dependencies
    pip install -r requirements.txt
```
## run code
```bash
# Install dependencies
    pip install -r requirements.txt
# Train the model
    python train.py
# Generate images with diffusion process visualization
    python generate.py \
        --checkpoint runs/ddpm_mnist/weights/best.pt \
        --config configs/exp.yaml \
        --output generated_images \
        --num_images 10000 \
        --batch_size 100 \
        --save_process
# Compute FID
    python -m pytorch_fid generated_images data

# plot loss curve
    python plot_training_curves.py --samples
```

## Dataset
The MNIST dataset will be automatically downloaded to ./data on first run.

- Training set: 60,000 images
- Test set: 10,000 images
- Image size: 28×28 (converted to RGB)

## Training Output
Results will be saved in runs/ddpm_mnist/:
- weights/ - Model checkpoints (epoch_*.pt, best.pt)
- results.csv - Training metrics per epoch
- hparams.json - Hyperparameters
- env.txt - Environment info
