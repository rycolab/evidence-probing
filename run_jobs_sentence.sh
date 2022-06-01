#!/bin/bash

# Sentence tasks
for R in random_context fasttext bert roberta xlnet albert t5
do
    sbatch --comment="Sentence [NLI, $R]" --export=TASK=sentence,LANGUAGE=eng,PROP=nli,REP=$R run_job.sh
    sbatch --comment="Sentence [BoolQ, $R]" --export=TASK=boolq,LANGUAGE=eng,PROP=boolq,REP=$R run_job.sh
    sbatch --comment="Sentence [CB, $R]" --export=TASK=cb,LANGUAGE=eng,PROP=cb,REP=$R run_job.sh
    sbatch --comment="Sentence [COPA, $R]" --export=TASK=copa,LANGUAGE=eng,PROP=copa,REP=$R run_job.sh
    sbatch --comment="Sentence [RTE, $R]" --export=TASK=rte,LANGUAGE=eng,PROP=rte,REP=$R run_job.sh
done
