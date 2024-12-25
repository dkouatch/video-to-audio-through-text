#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=vt2a_mlm_stage_2
#SBATCH -o vt2a_mlm_stage_2%j.out
#SBATCH -e vt2a_mlm_stage_2%j.err
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90
# # SBATCH --requeue

#SBATCH -A atlas_g
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=none
export SLURM_CPU_BIND="cores"


# activate conda env
# source /gscratch/stf/suk4/miniconda3/etc/profile.d/conda.sh
module load python
# CONDA_ENV=/gscratch/stf/suk4/mm_diff
CONDA_ENV=venv_ltu_as
conda activate $CONDA_ENV
export PYTHONPATH=/pscratch/sd/x/xiuliu/ltu/src/
# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
# srun python3 /global/homes/x/xiuliu/VAB/vab_clip_mlm_train.py -b /global/homes/x/xiuliu/VAB/configs/vab_clip_mlm_ff_alibi_mix_large.yaml -l /pscratch/sd/x/xiuliu/vab_clip_mlm_ff_alibi_mix_large_logs

# srun python3 /global/homes/x/xiuliu/VAB/vab_clip_mlm_train.py -b /global/homes/x/xiuliu/VAB/configs/vab_mae_mlm_ff_alibi_mix_base.yaml -l /pscratch/sd/x/xiuliu/vab_mae_mlm_ff_alibi_mix_base_logs
srun python3 /pscratch/sd/x/xiuliu/ltu/src/vt2a/vt2a_mlm_train.py -b /pscratch/sd/x/xiuliu/ltu/src/vt2a/configs/vt2a_mlm_alibi_mix_large_unicodec_vgg_stage_2.yaml -l /pscratch/sd/x/xiuliu/vt2a_mlm_alibi_mix_large_unicodec_vgg_stage_2_ft_node_finetune_logs