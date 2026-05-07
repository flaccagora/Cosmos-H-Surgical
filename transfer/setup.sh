
#!/bin/bash

USERNAME=${USER}
WORKDIR="/leonardo_work/IscrC_FLAC_0"
UV_CACHE_DIR="/leonardo_work/IscrC_FLAC_0/.uv"
HF_HOME="/leonardo_work/IscrC_FLAC_0/.HF_DOWNLOADS"

echo "Username: $USERNAME"
echo "Working directory: $WORKDIR"
echo "Hugging Face cache directory: $HF_HOME"

echo "Setting up environment for user: $USERNAME"

echo "Loading necessary modules"
ml purge
ml load git-lfs/ 
ml load cuda/12.2

echo "Cloning the repository in $WORKDIR"
cd $WORKDIR
git clone https://github.com/flaccagora/Cosmos-H-Surgical.git 
cd Cosmos-H-Surgical
git lfs pull

echo "Copying sif container"
cd transfer
cp /leonardo/pub/userexternal/mnunzian/cosmos-h.sif .

echo "Setting up python environment"
uv sync --extra=cu128

echo "Removing all previous Hugging Face downloads to ensure a clean setup"
rm -rf $HF_HOME/*

echo "Downloading the model weights using Hugging Face"
source .venv/bin/activate
python3 hf_download.py --groups cosmos_h_surgical_transfer

echo "Setup complete. You can now run the Inference script or run inference interactively"