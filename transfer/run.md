# Login

Authenticate with Step:

```shell
step ssh login email@example.com --provisioner cineca-hpc
```

Connect to Leonardo:

```shell
ssh leonardo
```

If you see a host key error, remove the stale entry and retry:

```shell
ssh-keygen -f ~/.ssh/known_hosts -R login.leonardo.cineca.it
```

Windows example:

```shell
ssh-keygen -f C:\Users\chiar\.ssh\known_hosts -R login.leonardo.cineca.it
```

## Code adjustments

Local modifications required for offline and controlled downloads:

- [checkpoint_db.py](../transfer/cosmos_transfer2/_src/imaginaire/utils/checkpoint_db.py): support `COSMOS_HF_NO_DOWNLOAD=1` and a pinned tokenizer revision (see [troubleshooting.md](./docs/troubleshooting.md)).
- [siglip2.py](../transfer/cosmos_transfer2/_src/transfer2/networks/siglip2.py).
- [hf_download.py](../transfer/hf_download.py): manual checkpoint download on the login node.

## Environment setup (login node)

Install dependencies:

```shell
ml purge
ml load cuda/12.2
uv sync --extra=cu128
```
First time users:

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission
2. Install [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli): `uv tool install -U "huggingface_hub[cli]"`
3. Login: `hf auth login`
4. Accept the model license agreements on Hugging Face:
   - [Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) — base model used by Cosmos-H-Surgical-Predict
   - [Cosmos-Transfer2.5-2B](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B) — base model used by Cosmos-H-Surgical-Transfer
   - [Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1) — guardrail model




Download model checkpoints on the login node:


```shell
export HF_HOME="/leonardo_work/IscrC_FLAC_0/.HF_DOWNLOADS"
python hf_download.py --groups cosmos_h_surgical_transfer
```

## Slurm resource request

Minimum resources for inference (adjust as needed):

```shell
salloc -N1 -p boost_usr_prod -A IscrC_FLAC --qos=boost_qos_dbg --gpus 1 --mem=100G
```

`--qos=boost_qos_dbg` enables the debug partition for limited interactive sessions that do not count against allocation.

Start an interactive shell on the compute node:

```shell
srun --pty bash
```

## Container deployment

Ensure `cosmos-h.sif` is present in the current directory.

Load CUDA on the compute node:

```shell
ml purge
ml load cuda/12.2
```

Run the Singularity container:

```shell
singularity exec --nv \
	--bind .:/workspace \
	--bind $HOME/.cache:/root/.cache \
	--bind /leonardo_scratch/large/userexternal/mnunzian/HF_DOWNLOADS:/leonardo_scratch/large/userexternal/mnunzian/HF_DOWNLOADS \
	cosmos-h.sif /bin/bash
```

Activate the environment inside the container:

```shell
source .venv/bin/activate
```

## Run inference

Set runtime environment variables:

```shell
export HF_HOME=/leonardo_scratch/large/userexternal/mnunzian/HF_DOWNLOADS
export HF_HUB_OFFLINE=1
export COSMOS_HF_NO_DOWNLOAD=1
```

Depth example:

```shell
python examples/inference.py -i assets/coagulation_example/depth/coagulation_depth_spec.json -o outputs/depth
```

Edge example:

```shell
python examples/inference.py -i assets/coagulation_example/edge/coagulation_edge_spec.json -o outputs/edge
```

## Setup Script
use the setup script `/leonardo_work/IscrC_FLAC_0/setup.sh` to initialize project:
`bash setup.sh`

## Inference script
To submit inference job (non interactive) use `Cosmos-H-Surgical/Cosmos-H-Surgical/transfer/inference.sh`:
`sbatch inference.sh`

the output and error files will be `inference.out` and `inference.err`