# How to login

`step ssh login matteo.nunziante@gmail.com --provisioner cineca-hpc`


`ssh leonardo` or `ssh cvaga000@login.leonardo.cineca.it`

se errore:

`ssh-keygen -f ~/.ssh/known_hosts -R login.leonardo.cineca.it`




### Had to modify

- [checkpoint_db.py](../transfer/cosmos_transfer2/_src/imaginaire/utils/checkpoint_db.py) COSMOS_HF_NO_DOWNLOAD=1 see [troubleshooting.md](./docs/troubleshooting.md)
- [siglip2.py](../transfer/cosmos_transfer2/_src/transfer2/networks/siglip2.py) 
- [hf_download.py](../transfer/hf_download.py) to manually download model checkpoints on login node

## Environment setup
Install dependencies on login node.

`uv sync` on login node 

Install models and dependencies on login node:

```shell
export HF_HOME=/leonardo_scratch/large/userexternal/mnunzian/HF_DOWNLOADS
python hf_download.py --groups cosmos_h_surgical_transfer
```

## Slurm Resource Request
`ml load cuda/12.3`

Resource request (minimum required for inference, may be adjusted based on model and inference pipeline):

`salloc -N1 -p boost_usr_prod -A IscrC_FLAC  --qos=boost_qos_dbg --gpus 1 --mem=100G`

`--qos=boost_qos_dbg` is required to access the **debug** partition, which allows for limited interactive sessions which are not counted against the user's allocation.

Move to compute node:

`srun --pty bash`

## Container deployment

make sure you have `cosmos-h.sif` in the right place 

Run Singularity image:
```shell
singularity exec --nv --bind .:/workspace --bind $HOME/.cache:/root/.cache --bind /leonardo_scratch/large/userexternal/mnunzian/HF_DOWNLOADS:/leonardo_scratch/large/userexternal/mnunzian/HF_DOWNLOADS cosmos-h.sif /bin/bash
```

env setup:
`source .venv/bin/activate`



## Run inference

```shell
export HF_HOME=/leonardo_scratch/large/userexternal/mnunzian/HF_DOWNLOADS
export HF_HUB_CACHE=\$HF_HOME/hub
export TRANSFORMERS_CACHE=\$HF_HOME/hub
export COSMOS_CACHE_DIR=\$HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COSMOS_HF_NO_DOWNLOAD=1
```

```shell
python examples/inference.py -i assets/coagulation_example/depth/coagulation_depth_spec.json -o outputs/depth
```

```shell
python examples/inference.py -i assets/coagulation_example/edge/coagulation_edge_spec.json -o outputs/edge

```