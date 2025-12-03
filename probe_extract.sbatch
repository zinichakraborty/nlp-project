#!/bin/bash
#SBATCH --job-name=probe_qwen3
#SBATCH --output=logs/probe_qwen3_%j.out
#SBATCH --error=logs/probe_qwen3_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_6000:1

set -euo pipefail
mkdir -p logs

cd /storage/ice1/6/6/cchakraborty3/nlp-project

echo "Activating project venv: storyorder"
source storyorder/bin/activate

export PYTHONNOUSERSITE=1

echo "Starting probe embedding extraction..."
python probe_scripts/extract.py \
  --output_path probe_embeds/embeddings_qwen3_32b_gold_medium.npz

echo "Job finished successfully."
