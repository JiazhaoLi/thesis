#!/bin/bash
#SBATCH --job-name=DDD
#SBATCH --account=vgvinodv1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32gb
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00


export CUDA_HOME=/home/jiazhaol/anaconda3/envs/dpo/


# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u train.py \
#     model=llama7b \
#     datasets=[hh] \
#     loss=dpo \
#     loss.beta=0.1 \
#     model.archive=/scratch/vgvinodv_root/vgvinodv1/jiazhaol/dpo/jiazhaol/anthropic_dpo_llama7b_2024-06-26_22-25-20_950661/LATEST/policy.pt \
#     output_dir=/scratch/vgvinodv_root/vgvinodv1/jiazhaol/dpo_clean_July1_0.5/ \
#     exp_name=anthropic_dpo_llama7b \
#     gradient_accumulation_steps=8 \
#     batch_size=64 \
#     eval_batch_size=64 \
#     trainer=FSDPTrainer \
#     sample_during_eval=false



HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -u train.py \
    model=llama7b \
    datasets=[hh] \
    loss=dpo \
    loss.beta=0.1 \
    model.archive=/scratch/vgvinodv_root/vgvinodv1/jiazhaol/dpo_clean_July1_0.5/policy.pt \
    output_dir=/scratch/vgvinodv_root/vgvinodv1/jiazhaol/dpo_clean_July1_0.5_debug/ \
    exp_name=anthropic_dpo_llama7b \
    gradient_accumulation_steps=8 \
    batch_size=64 \
    eval_batch_size=64 \
    trainer=FSDPTrainer \
    sample_during_eval=false