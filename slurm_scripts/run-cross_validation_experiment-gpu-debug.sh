#!/usr/bin/env bash
#SBATCH --job-name=cross_validation_experiment
####Change account below
#SBATCH --account=cso100
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=%x.o%j.%N

declare -xr SINGULARITY_MODULE='singularitypro/3.11'

module purge
module load "${SINGULARITY_MODULE}"
module list

# Check if model type argument is passed
if [ -z "$1" ]; then
    echo "Error: No model type specified. Usage: sbatch $0 <cnn|rnn>"
    exit 1
fi

MODEL_TYPE=$1  # Capture model type argument

export NVIDIA_DISABLE_REQUIRE=true

time -p singularity exec --bind /expanse,/scratch --nv ./train-container-sandbox python -u ./cross_validation_experiment.py --model "$MODEL_TYPE"
