# project-kirby
Poyo!

# Installation
## Environment setup with `venv`
Clone the project, enter the project's root directory, and then run the following:
```bash
python3.9 -m venv venv           # create an empty virtual environment
source venv/bin/activate         # activate it
pip install -r requirements.txt  # install required packages
pip install -r requirements.txt  # yes, run it again
pip install -e .                 # install project-kirby into your path
```

Currently this project requires the following:
- Python 3.9 (also requires python3.9-dev)
- PyTorch 2.0.0
- CUDA 11.3 - 11.7 
- xformers is optional, but recommended for training with memory efficient attention


## Downloading and preparing the data
Run the following to download and prepare the data:
```bash
snakemake --cores 8 odoherty_sabes
```

To prepare all of the datasets from the NeurIPS paper:
```bash
snakemake --cores 8 poyo_neurips
```

# Training
To train POYO you can run:
```bash
python train.py --config-name train.yaml
```
Everything is logged to wandb.

# Finetuning
## Unit-Identification
```bash
python python train.py --config-name unit_identification.yaml
```

## Full finetuning
```bash
python python train.py --config-name finetune.yaml
```
