# 114-1_TAICA_cv_hw3
MNIST
.
├── train.py              # Training entry point
├── generate.py           # Image generation script
├── hooks.py              # Model, dataloader, and evaluation implementation
├── configs/
│   └── exp.yaml         # Configuration file
├── tools/
│   ├── utils.py         # Utility functions
│   ├── io.py            # I/O operations
│   └── kfold.py         # K-fold utilities
└── data/                # MNIST dataset (auto-downloaded)

## Environment Setup
### Install pyenv
curl -fsSL https://pyenv.run | bash
export PATH="$HOME/.pyenv/bin:$PATH"

### Install Python 3.12.7
pyenv install 3.12.7
pyenv local 3.12.7

### Create virtual environment
python -m venv .venv
source .venv/bin/activate

### Install dependencies
pip install -r requirements.txt


## Data
Dataset
The MNIST dataset will be automatically downloaded to ./data on first run.

- Training set: 60,000 images
- Test set: 10,000 images
- Image size: 28×28 (converted to RGB)

## Training Output
Results will be saved in runs/exp1/:

- weights/ - Model checkpoints (epoch_*.pt, best.pt)
- results.csv - Training metrics per epoch
- hparams.json - Hyperparameters
- env.txt - Environment info

## Image Generation
### Generate 10,000 images for submission
``` bash
python generate.py \
    --checkpoint runs/exp1/weights/best.pt \
    --config configs/exp.yaml \
    --output generated_images \
    --num_images 10000 \
    --batch_size 100
```
### Generate with diffusion process visualization
``` bash
python generate.py \
    --checkpoint runs/exp1/weights/best.pt \
    --config configs/exp.yaml \
    --output generated_images \
    --num_images 10000 \
    --batch_size 100 \
    --save_process
```

## run code
`python train.py`
`python generate.py --checkpoint runs/ddpm_mnist/weights/best.pt--config configs/exp.yaml --output generated_images --num_images 10000 --batch_size 100`
`python generate.py --checkpoint runs/ddpm_mnist/weights/best.pt --config configs/exp.yaml --output generated_images --num_images 10000 --batch_size 100 --save_process`