#!/bin/bash

# Arc tasks
for L in eng tur ara mar zho deu
do
    for R in random_context random_word fasttext bert
    do
        sbatch --comment="Arc [$L, $R]" --export=TASK=arc,LANGUAGE=$L,PROP=dep,REP=$R run_job.sh
    done
done
