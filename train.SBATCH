#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=0:10:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=MOCO_200K_blur
#SBATCH --output=/scratch/mr6744/pytorch/outputs_slurm/%j.out

export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

module purge

srun \
	singularity exec --nv \
	--overlay /scratch/mr6744/pytorch/overlay-25GB-500K.ext3:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c "source /ext3/env.sh; python -u main_moco.py \
												-a resnet50 \
												--lr 0.03 \
												--batch-size 256 \
												--epochs 200 \
												--dist-url "env://" \
												--mlp --moco-t 0.2 --aug-plus --cos \
												--ckpt_path /vast/mr6744/pretrained_models_moco/200K_blur \
												--multiprocessing-distributed \
												/vast/mr6744/SAYCAM_blur/ "