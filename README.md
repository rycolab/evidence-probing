# Bayesian Probing

This is the repository for Bayesian probing.

## Setup

The following instructions are enough to setup the code and download and process all the data.

1. (Optional) Create environment `conda env create -f environment.yml` and activate it with `conda activate bayesian-probing`.
2. Run `pip install requirements.txt`
3. Run `make install`
4. Install the appropriate [pytorch\_scatter](https://github.com/rusty1s/pytorch_scatter) for your CUDA version.
5. Run `python -m spacy download en_core_web_sm` to setup Spacy
6. Download and process all the required data with `make data`. If you are running OS X, you might be asked to install some additional tools.

If you encounter any problems in final step, verify that you activated the environment and installed all dependencies.

### Generating only part of the data
The data for this paper comes from a few different sources, but the Make should pull everything together for you if you setup everything correctly.
If not, open an issue.

Below are some details on how the data for each sub-experiment is generated.

#### Token-level data
For token-level tasks, we use the procedure from Intrinsic Probing, except that we run it on the UD 2.5 treebanks (like in Pareto Probing).
You can obtain it with `make data_token`.

#### Arc-level data
For the arc tasks, we use the Pareto probing data as-is. You can download everything and process it with `make data_arc`.

#### Sentence-level data
For the sentence-level tasks, we use the [MultiNLI dataset](https://github.com/nyu-mll/multiNLI) and some [SuperGLUE tasks](https://super.gluebenchmark.com/).
In general, we obtain representations for every token and then average them to obtain a sentence-level representation.
That said, there are some task-specific variations.
You can download and prepare this data with `make data_sentence`.

## Usage

You should be able to replicate our experiments with the following command:
```
python -u run.py --task-type ${TASK} --language ${LANGUAGE} --attribute ${PROP} --contextualizer ${REP} --seed 2 --gpu \
    --trainer-num-epochs 500 --step_size 1e-2 --trainer-batch-size 512 marglik --depths 0 1 2 --widths 100 --posterior-structure ${STRUCT}
```
where:
- `TASK`: This is one of `token` (for all the morphosyntactic token-level tasks), `arc` (for the arc-level dependency task), `sentence` (this is the NLI task), `boolq`, `cb`, `copa`, `rte`
- `PROP`: This is dependent on the setting of `TASK`. In short:
    - If `TASK` is `token`, then this can be either `Case`, `Number`, `Tense` or `POS`
    - If `TASK` is `arc`, then this must be set to `dep`
    - If `TASK` is `sentence`, this this must be set to `nli`
    - If `TASK` is anything else, then this must be set to the same value as `TASK`
- `LANGUAGE`: This is the language to probe. Again, this is task-specific, since not all tasks are available for all languages:
    - If `TASK` is set to `token` or `arc`, then this can be set to either: `eng`, `tur`, `ara`, `mar`, `zho`, `deu`
    - If `TASK` is set to anything else, then this must be set to `eng`
- `REP`: This is the representation whose inductive bias is being measured. This is task-specific.
    - If `TASK` is set to `token` or `arc`, then this can be set to either: `random_context` (fully random), `random_word` (per-word random), `fasttext` (language-specific fastText), `bert` (multilingual BERT)
    - If `TASK` is set to anything else, then this must be set to `random_context` (fully random), `fasttext`, `bert` (English BERT), `roberta`, `xlnet`, `albert`, `t5`
- `STRUCT` controls how the Laplace approximation is built, and must be set to either `kron` or `diag`.

Some other handy flags are `--quiet` (to suppress most logging), and `--output-file` (to specify where experiment data should be saved).
