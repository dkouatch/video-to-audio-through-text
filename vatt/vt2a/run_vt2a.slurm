#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=vt2a_mlm_stage_2
#SBATCH -o vt2a_mlm_stage_2%j.out
#SBATCH -e vt2a_mlm_stage_2%j.err
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90
# # SBATCH --requeue

#SBATCH -A xxx
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=none
export SLURM_CPU_BIND="cores"


module load python
CONDA_ENV=venv_ltu_as
conda activate $CONDA_ENV
export PYTHONPATH=/path/to/vatt

# stage 2 training of audio generation from video + text
srun python3 /path/to/vatt/vt2a_mlm_train.py -b /path/to/vt2a/configs/vt2a_mlm_alibi_mix_large_unicodec_vgg_stage_2.yaml -l /path/to/vt2a_mlm_alibi_mix_large_unicodec_vgg_stage_2_ft_node_finetune_logs