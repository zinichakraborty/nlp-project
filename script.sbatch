#!/bin/bash
#
#SBATCH --job-name=reorder-stories
#SBATCH --output=logs/reorder_%j.out
#SBATCH --error=logs/reorder_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=20G

# OPTIONAL: load conda if you use conda; comment out if using venv only
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate nlp-env

# If using your venv:
source /storage/ice1/6/6/cchakraborty3/nlp-project/nlp-env/bin/activate

echo "Job started on $(hostname) at $(date)"

# Run script with unbuffered output
python -u /storage/ice1/6/6/cchakraborty3/nlp-project/script.py

echo "Job finished at $(date)"
