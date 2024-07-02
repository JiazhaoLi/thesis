#!/bin/bash
#SBATCH --job-name=DDD
#SBATCH --account=drjieliu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00


# export CUDA_HOME=/home/jiazhaol/anaconda3/envs/dpo/

# output_dir=/nfs/turbo/umms-drjieliu/usr/zzh/tmp/hh_sft_debug/ \
module load gcc/10.3.0
module load cuda/12.1.1

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train.py \
    model=llama7b \
    model.block_name=LlamaDecoderLayer \
    output_dir=/scratch/vgvinodv_root/vgvinodv1/jiazhaol/dpo/sft_debug/ \
    datasets=[hh] \
    loss=sft \
    exp_name=anthropic_dpo_llama7b \
    gradient_accumulation_steps=2 \
    batch_size=64 \
    eval_batch_size=64 \
    trainer=FSDPTrainer \
    sample_during_eval=false