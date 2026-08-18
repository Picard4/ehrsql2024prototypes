#!/bin/bash
#SBATCH --job-name=Qwen_Training
#SBATCH --gres=gpu:nvidia_a100_2g.20gb:1
#SBATCH --output=logs/model_training.log
#SBATCH --error=logs/model_training.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=55:00:00

# 1. Navigate to your project directory (use your actual scratch path)
cd /nfs/speed-scratch/p_dimar/Thesis/ehrsql-custom-model

# 2. Activate your virtual environment (force bash compatibility if needed)
source .venv/bin/activate

# 3. Install dependencies
pip install --no-cache-dir --upgrade "torch<2.5.0" "transformers>=4.42.0" huggingface_hub tiktoken einops transformers_stream_generator datasets accelerate peft ipywidgets trl

# 4. Run your Python script
python qwen_lora_train.py

# 5. Deactivate the environment when done
deactivate
