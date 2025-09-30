#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename

#SBATCH --time=4:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 8 processor core(s) per node
#SBATCH --mem=64G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="data_gen"
#SBATCH --mail-user=hungphd@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="hung-out.txt" # job standard output file (%j replaced by job id)
#SBATCH --error="hung-err.txt" # job standard error file (%j replaced by job id)

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
export CUDA_VISIBLE_DEVICES=0,1
export TRITON_CACHE_DIR='/work/LAS/jannesar-lab/hungphd/git/.triton/'
cd /work/LAS/jannesar-lab/hungphd/git/
module load python
source hungpy39/bin/activate
cd /work/LAS/jannesar-lab/hungphd/git/PowerOutageTracker/src/context/
python context_sample.py \
  --model_id /work/LAS/jannesar-lab/hungphd/git/pretrained_open_llms/Mistral-7B-v0.3 \
  --input_train ../../preprocess-dataset/input_train.json \
  --input_test ../../preprocess-dataset/input_test.json \
  --context_path ../../pair-pair-dataset/ODIN.json \
  --output ../../results_action/context_sample_Mistral-7B-v0.3.json \
  --max_attempts 3 \
  --max_input_tokens 2048