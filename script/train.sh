#!/bin/bash

#SBATCH --job-name=LTT_NC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info
MIXUP=$1
ALPHA=$2
RESAMPLE=$3
IMBAMANCE=$4


# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
conda activate glmc
python main_wb.py --dataset cifar10 -a resnet32 --imbanlance_rate ${IMBAMANCE} --beta 0.5 --lr 0.01 \
 --epochs 200 --loss ce --resample_weighting ${RESAMPLE} --mixup ${MIXUP} --mixup_alpha ${ALPHA} --store_name ce_mx${MIXUP}_a${ALPHA}_resample${RESAMPLE}_imb${IMBAMANCE}
"
