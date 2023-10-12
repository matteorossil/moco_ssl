#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=8:00:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=ML_200K_deblur
#SBATCH --output=/scratch/mr6744/pytorch/outputs_slurm/%j.out

module purge

singularity exec --nv \
	--overlay /scratch/mr6744/pytorch/overlay-25GB-500K.ext3:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c "source /ext3/env.sh; python -u main_lincls.py \
												-a resnet50 \
												--lr 30.0 \
												--batch-size 256 \
												--pretrained /vast/mr6744/pretrained_models_moco/200K_deblur_new/checkpoint_0250.pth.tar \
												--ckpt_path /vast/mr6744/linear_evaluation_moco/200K_deblur_new/ \ 
												/vast/mr6744/MiniImagenet "