#!/bin/bash

# Sentence tasks
for P in Case Number Tense POS
do
    for L in eng tur ara mar zho deu
    do
        for R in random_context random_word fasttext bert
        do
            sbatch --comment="Token.$P [$L, $R]" --export=TASK=token,LANGUAGE=$L,PROP=$P,REP=$R run_job.sh
        done
    done
done
