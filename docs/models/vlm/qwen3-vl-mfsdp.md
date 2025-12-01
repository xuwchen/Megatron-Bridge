# Qwen3-VL pretraining with Megatron-FSDP

## Step 1: Download Source Code

```bash
# Download Megatron-Bridge
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge/
git checkout -b qwen3_vl origin/main

# Download Megatron-LM
cd 3rdparty/
git clone https://github.com/NVIDIA/Megatron-LM
cd Megatron-LM/
git checkout -b qwen3_vl_dev origin/dev
cd ../../
```

## Step 2: Update the Config File

Update the model config YAML file for your use case and add the Megatron-FSDP required arguments. An example configuration is provided for [Qwen3-VL-30B-A3B-Instruct](../../../examples/recipes/qwen_vl/conf/qwen3_vl_30b_a3b_pretrain_mfsdp_override_example.yaml).

## Step 3: Run Training

### Interactive Node Training

To train on an interactive node, run the following commands:

```bash
# Set docker image name
export DOCKER_IMAGE=/lustre/fsw/portfolios/coreai/users/shifangx/docker/mbridge-251110.sqsh
# Start docker container
srun -A coreai_dlalgo_mcore -p interactive --time=04:00:00 --gpus-per-node=8 --container-image=${DOCKER_IMAGE} --container-mounts=/lustre:/lustre --container-workdir="$PWD" -J coreai_dlalgo_mcore-test:test --pty bash

# Set env
export MEGATRON_BRIDGE_PATH=/path/to/Megatron-Bridge
export MEGATRON_LM_PATH=${MEGATRON_BRIDGE_PATH}/3rdparty/Megatron-LM
export PYTHONPATH=${MEGATRON_BRIDGE_PATH}:${MEGATRON_LM_PATH}:${PYTHONPATH}
export UV_CACHE_DIR=${${MEGATRON_BRIDGE_PATH}}/../uv_cache
export HF_CACHE_DIR=${${MEGATRON_BRIDGE_PATH}}/../hf_cache

# Run pretraining
export PYTHONPATH=${MEGATRON_BRIDGE_PATH}:${MEGATRON_LM_PATH}:${PYTHONPATH}
cd ${MEGATRON_BRIDGE_PATH}/examples/recipes/qwen_vl

uv run python -m torch.distributed.run --nproc_per_node=8 \
    finetune_qwen_vl.py \
    --recipe qwen3_vl_3b_active_30b_moe_finetune_config \
    --config-file conf/qwen3_vl_30b_a3b_pretrain_mfsdp_override_example.yaml
```

**Note:** If you want to track experiments using Wandb, use the command below to ensure detailed model configuration recorded in your experiment overview. Otherwise, Wandb will only display the config file path (`--config-file conf/qwen3_vl_30b_a3b_pretrain_mfsdp_override_example.yaml`) instead of the actual configuration values.

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    finetune_qwen_vl.py \
    --recipe qwen3_vl_3b_active_30b_moe_finetune_config \
    model.tensor_model_parallel_size=1 \
    model.expert_model_parallel_size=8 \
    model.freeze_language_model=false \
    model.freeze_vision_model=false \
    model.freeze_vision_projection=false \
    model.init_model_with_meta_device=true \
    model.seq_length=4096 \
    model.gradient_accumulation_fusion=false \
    train.train_iters=20 \
    train.global_batch_size=32 \
    train.micro_batch_size=1 \
    train.eval_iters=5 \
    optimizer.lr=2e-5 \
    optimizer.min_lr=2e-6 \
    optimizer.use_distributed_optimizer=true \
    scheduler.lr_warmup_iters=10 \
    checkpoint.save=your_own_path_to_checkpoints \
    checkpoint.ckpt_format=fsdp_dtensor \
    dist.use_megatron_fsdp=true \
    dist.use_torch_fsdp2=false \
    logger.log_interval=1 \
    logger.wandb_project=your_own_wandb_project \
    logger.wandb_exp_name=your_own_wandb_experiment_name \
    ddp.grad_reduce_in_fp32=false \
    ddp.use_megatron_fsdp=true \
    ddp.use_distributed_optimizer=true \
    ddp.data_parallel_sharding_strategy=optim_grads_params
```

### Multi-Node Training

For multi-node training, use this example script:

```bash
export MEGATRON_BRIDGE_PATH=/path/to/Megatron-Bridge
export MEGATRON_LM_PATH=${MEGATRON_BRIDGE_PATH}/3rdparty/Megatron-LM
export PYTHONPATH=${MEGATRON_BRIDGE_PATH}:${MEGATRON_LM_PATH}:${PYTHONPATH}
export UV_CACHE_DIR=${${MEGATRON_BRIDGE_PATH}}/../uv_cache
export HF_CACHE_DIR=${${MEGATRON_BRIDGE_PATH}}/../hf_cache
export CONTAINER_IMAGE=/lustre/fsw/portfolios/coreai/users/shifangx/docker/mbridge-251110.sqsh
export OUTPUT_PATH=${MEGATRON_BRIDGE_PATH}/../MCore/Qwen/Qwen3-VL-30B-A3B-Instruct/output

RUN_CMD="
export PYTHONPATH=${MEGATRON_BRIDGE_PATH}:${MEGATRON_LM_PATH}:${PYTHONPATH}
cd ${MEGATRON_BRIDGE_PATH}/examples/recipes/qwen_vl/;
uv run python \
    finetune_qwen_vl.py \
    --recipe qwen3_vl_3b_active_30b_moe_finetune_config \
    --config-file conf/qwen3_vl_30b_a3b_pretrain_mfsdp_override_example.yaml"

# SLURM settings
SLURM_LOGS="${OUTPUT_PATH}/slurm_logs"
mkdir -p ${SLURM_LOGS} || {
    echo "Error: Failed to create SLURM logs directory ${SLURM_LOGS}"
    exit 1
}

# Submit SLURM job
# Note: Update SBATCH parameters below according to your cluster configuration
set +e
sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=qwen3-vl-30b-a3b-pretrain-mfsdp
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:10:00
#SBATCH --account=coreai_devtech_all
#SBATCH --exclusive
#SBATCH --dependency=singleton

srun --mpi=pmix -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts="/lustre:/lustre" \
    --container-workdir=${MEGATRON_BRIDGE_PATH} \
    bash -x -c "${RUN_CMD}" 2>&1 | tee ${SLURM_LOGS}/\${SLURM_JOB_ID}.log

EOF
set -e

```