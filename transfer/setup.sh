#!/bin/bash

USERNAME=${USER}
WORKDIR="/leonardo_work/IscrC_FLAC_0"
export UV_CACHE_DIR="/leonardo_work/IscrC_FLAC_0/.uv"
export HF_HOME="/leonardo_work/IscrC_FLAC_0/.HF_DOWNLOADS"


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
source .venv/bin/activate

echo "Removing all previous Hugging Face downloads to ensure a clean setup"
rm -rf $HF_HOME/*

# check hf auth if needed
if command -v hf >/dev/null 2>&1; then
    echo "Checking Hugging Face authentication..."
    if hf auth whoami >/dev/null 2>&1; then
        echo "Hugging Face already authenticated: $(hf auth whoami)"
    else
        if [ -n "$HF_TOKEN" ]; then
            echo "HF_TOKEN found in environment — logging in using token..."
            hf auth login --token "$HF_TOKEN"
        else
            echo "Hugging Face not authenticated. You will be prompted to log in now."
            echo "If you prefer a non-interactive login, set the HF_TOKEN environment variable and re-run this script."
            hf auth login
        fi

        # verify login succeeded
        if hf auth whoami >/dev/null 2>&1; then
            echo "Hugging Face authentication successful: $(hf auth whoami)"
        else
            echo "Hugging Face authentication failed. Exiting."
            exit 1
        fi
    fi
else
    echo "'hf' CLI not found. Install the Hugging Face CLI (e.g. 'pip install huggingface_hub') or ensure 'hf' is on PATH."
    exit 1
fi

echo "Downloading the model weights using Hugging Face"
python3 hf_download.py --groups cosmos_h_surgical_transfer

echo "Setup complete. You can now run the Inference script (Cosmos-H-Surgical/transfer/inference.sh) or run inference interactively"