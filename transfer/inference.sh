#!/bin/bash
#SBATCH -A IscrC_FLAC
#SBATCH -p boost_usr_prod
#SBATCH --time 00:30:00      # format: HH:MM:SS
#SBATCH -N 1                 # 1 node
#SBATCH --ntasks-per-node=8  # 4 tasks out of 32
#SBATCH --mem=64000          # memory per node out of 494000MB (481GB)
#SBATCH --gres=gpu:1
#SBATCH --job-name=cosmos-h-inference
#SBATCH --output=inference.out
#SBATCH --error=inference.err

date
WORKDIR="/leonardo_work/IscrC_FLAC_0/Cosmos-H-Surgical/transfer"
USERNAME=${USER}

echo "Loading necessary modules"
ml load git-lfs/
ml load cuda/12.6

echo "Working directory: $WORKDIR"
cd $WORKDIR

echo "Setting up Container Inference Command"
CONTAINER_CMD="source .venv/bin/activate && python examples/inference.py -i assets/coagulation_example/depth/coagulation_depth_spec.json -o outputs/depth"
echo "Container command: $CONTAINER_CMD"

echo "Ensuring offline access to Hugging Face models"
export HF_HUB_OFFLINE=1
export COSMOS_HF_NO_DOWNLOAD=1

echo "Running inference inside the container"
singularity exec --nv --bind .:/workspace \
    --bind $HOME/.cache:/root/.cache \
    --bind /leonardo_scratch/large/userexternal/${USERNAME}/HF_DOWNLOADS:/leonardo_scratch/large/userexternal/${USERNAME}/HF_DOWNLOADS \
    cosmos-h.sif /bin/bash -c "$CONTAINER_CMD"


date
echo "Inference complete. Check outputs/depth for results."

