#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:32gb:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --exclude=cn-c037
#SBATCH --requeue

# Change for re-runs
export SEED=2

export FILEK=${TASK}/${REP}/${TASK}_${LANGUAGE}_${PROP}_${REP}.kron.json
export FILED=${TASK}/${REP}/${TASK}_${LANGUAGE}_${PROP}_${REP}.diag.json

echo "Task: ${TASK}"
echo "Lang: ${LANGUAGE}"
echo "Prop: ${PROP}"
echo "Rep: ${REP}"
echo "Seed: ${SEED}"
echo "File (kron): ${FILEK}"
echo "File (diag): ${FILED}"

conda activate bayesian-probing

if [ ! -f "results/$FILEK" ]; then
    echo "Generating results/$FILEK"
    python -u run.py --output-file "${FILEK}" --task-type ${TASK} --language ${LANGUAGE} --attribute ${PROP} --contextualizer ${REP} --seed ${SEED} --gpu \
        --trainer-num-epochs 500 --step_size 1e-2 --quiet --trainer-batch-size 512 marglik --depths 0 1 2 --widths 100 --posterior-structure kron
else
    echo "Skipping since results/$FILEK already exists."
fi

if [ ! -f "results/$FILED" ]; then
    echo "Generating results/$FILED"
    python -u run.py --output-file "${FILED}" --task-type ${TASK} --language ${LANGUAGE} --attribute ${PROP} --contextualizer ${REP} --seed ${SEED} --gpu \
        --trainer-num-epochs 500 --step_size 1e-2 --quiet --trainer-batch-size 512 marglik --depths 0 1 2 --widths 100 --posterior-structure diag
else
    echo "Skipping since results/$FILED already exists."
fi
