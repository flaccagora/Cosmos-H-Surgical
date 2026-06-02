#!/bin/bash
#SBATCH -A IscrC_FLAC
#SBATCH -p boost_usr_prod
#SBATCH --time 00:30:00      # format: HH:MM:SS
#SBATCH -N 1                 # 1 node
#SBATCH --ntasks-per-node=8  # 4 tasks out of 32
#SBATCH --mem=64000          # memory per node out of 494000MB (481GB)
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=cosmos-h-inference
#SBATCH --output=inference_output/inference_%j.out
#SBATCH --error=inference_output/inference_%j.err

date
WORKDIR="/leonardo_work/IscrC_FLAC/Cosmos-H-Surgical/transfer"
USERNAME=${USER}
export HF_HOME="/leonardo_scratch/large/userexternal/mnunzian/HF_DOWNLOADS"
export HF_HUB_OFFLINE=1
export COSMOS_HF_NO_DOWNLOAD=1

echo "Loading necessary modules"
ml purge
ml load cuda/12.2

echo "Working directory: $WORKDIR"
cd $WORKDIR

ENV_COMMAND="source .venv/bin/activate"
PYTHON_CMD="python examples/inference.py -i assets/my_example/edge/myexample_edge_spec.json -o outputs/myexample/edge"
CONTAINER_CMD="$ENV_COMMAND && $PYTHON_CMD"
echo "Container command: $CONTAINER_CMD"

echo "Running inference inside the container"
singularity exec --nv --bind .:/workspace \
    --bind $HOME/.cache:/root/.cache \
    --bind $HF_HOME:$HF_HOME \
    cosmos-h.sif /bin/bash -c "$CONTAINER_CMD"



date
echo "Inference complete. Check outputs/depth for results."

