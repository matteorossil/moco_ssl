#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=M_200K_full_blur
#SBATCH --output=/scratch/mr6744/pytorch/outputs_slurm/%j.out

module purge

singularity exec --nv \
	--overlay /scratch/mr6744/pytorch/overlay-25GB-500K.ext3:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c "source /ext3/env.sh; python -u main_moco.py \
												-a resnet50 \
												--lr 0.015 \
												--batch-size 128 \
												--epochs 1000 \
												--mlp --moco-t 0.2 --aug-plus --cos \
												--dist-url 'tcp://localhost:10009' --multiprocessing-distributed --world-size 1 --rank 0 \
												--ckpt_path /vast/mr6744/pretrained_models_moco/200K_blur_full \
												/vast/mr6744/SAYCAM_blur_full/ "