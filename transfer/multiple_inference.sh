#!/bin/bash
#SBATCH -A IscrC_FLAC
#SBATCH -p boost_usr_prod
#SBATCH --exclusive
#SBATCH --time 00:30:00      # format: HH:MM:SS
#SBATCH -N 1                 # 1 node
#SBATCH --ntasks-per-node=4  # 4 tasks 
#SBATCH --cpus-per-task=8    # 8 CPUs per task -> total 32 CPUs
#SBATCH --mem=480000         # memory per node out of 494000MB (512GB)
#SBATCH --gres=gpu:4
#SBATCH --job-name=cosmos-h-inference
#SBATCH --output=inference_output/inference_%j.out
#SBATCH --error=inference_output/inference_%j.err

date
WORKDIR="/leonardo_work/IscrC_FLAC_0/Cosmos-H-Surgical/transfer"
USERNAME=${USER}
export HF_HUB_OFFLINE=1
export COSMOS_HF_NO_DOWNLOAD=1
export HF_HOME="/leonardo_work/IscrC_FLAC_0/.HF_DOWNLOADS"
export OMP_NUM_THREADS=8

echo "Loading necessary modules"
ml purge
ml load cuda/12.2

echo "Working directory: $WORKDIR"
cd $WORKDIR

ENV_COMMAND="source .venv/bin/activate"
OUTPUT_DIR="outputs/batch"
INPUT_1="assets/coagulation_example/depth/coagulation_depth_spec.json"
INPUT_2="assets/needleGrasping_example/depth/needleGrasping_depth_spec.json"
INPUT_3="assets/needlePuncture_guided_example/depth/needlePuncture_depth_spec.json"

CMD_BATCH="torchrun --nproc_per_node=4 examples/inference.py \
    -i $INPUT_1 \
    -i $INPUT_2 \
    -i $INPUT_3 \
    -o $OUTPUT_DIR"

CONTAINER_CMD="$ENV_COMMAND && $CMD_BATCH"
echo "Container command: $CONTAINER_CMD"

echo "Running inference inside the container"
singularity exec --nv --bind .:/workspace \
    --bind $HOME/.cache:/root/.cache \
    --bind $HF_HOME:$HF_HOME \
    cosmos-h.sif /bin/bash -c "$CONTAINER_CMD"



date
echo "Inference complete. Check outputs for results."

