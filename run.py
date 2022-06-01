from argparse import ArgumentParser

import commands


parser = ArgumentParser()
subparsers = parser.add_subparsers(description="General script to run experiments. Different modes are \
                                   available depending on what type of experiment you want to run. The \
                                   options are split into two groups: general options (shown below), and \
                                   mode-specific options.")

parser.add_argument("--task-type", choices=["token", "arc", "sentence", "boolq", "cb", "copa", "rte"],
                    help="If using token tasks, \
                    embeddings correspond to contextualized tokens. If using arc tasks, embeddings \
                    correspond to the concatenation of the child and head embeddings. For sentence-tasks, this \
                    is task-sepcific.")
parser.add_argument("--language", type=str, required=True, help="The three-letter code for the language \
                    you want to probe (e.g., eng). Should be the same as used by Unimorph.")
parser.add_argument("--attribute", type=str, required=True, help="The attribute (aka. Unimorph dimension) \
                    to be probed (e.g., \"Number\", \"Gender and Noun Class\"). For arc/sentence tasks this might \
                    be the metadata attribute containing the label, e.g. nli, dep, boolq, etc.")
parser.add_argument("--contextualizer", type=str, required=True, help="The contextualizer used. The contextualizers \
                    available depend on the task. For token- and arc- tasks, we support 'bert', 'fasttext', \
                    'random_word', and 'random_context', where the first two correspond to the multilingual variants \
                    of BERT and fastText, respectively. For sentence-level tasks, we support 'bert', 'albert', \
                    'roberta', 'xlnet', 't5', 'fasttext', using English-specific model variants.", default='bert')
parser.add_argument("--output-file", type=str, help="If provided, results of the experiment will be written \
                    to this file in JSON format.")
parser.add_argument("--gpu", default=False, action="store_true", help="Pass this flag if you want to use a \
                    GPU to speed up the experiments. Multiple GPUs are not supported.")
parser.add_argument("--step_size", type=float, default=1e-2)
parser.add_argument("--trainer-num-epochs", type=int, default=500, help="The maximum number of epochs that \
                    probes should be trained for (default: 10000).")
parser.add_argument("--trainer-batch-size", type=int, default=500, help="Batch size of trainer.")
parser.add_argument("--reduce_dimensions", action="store_true", help="Run only on pre-selected dimensions.")
parser.add_argument("--seed", type=int, default=711)
parser.add_argument("--word-split", action="store_true", help="Split train test dev by word not context.")
parser.add_argument("--quiet", action="store_true", default=False, help="Reduce logging output.")

# ::::: MANUAL MODE :::::
# You can manually specify the dimensions you want to check--mostly for development and debugging purposes.
parser_manual = subparsers.add_parser("manual", description="Train a probe and compute its associated \
                                      metrics")
parser_manual.add_argument("--probe-num-layers", type=int, default=0, help="The number of layers the probe \
                           should have. If set to 1, corresponds to logistic regression (default: 1).")
parser_manual.add_argument("--probe-num-hidden-units", type=int, default=50, help="The number of hidden units in \
                           each (hidden) layer of the probe (default: 50).")
parser_manual.add_argument("--regularization", type=float, default=1e-3, help="Hyperparameter controlling the weight \
                           of L2 regularization in the loss function (default: 0.001).")
parser_manual.set_defaults(func=commands.manual)

# ::::: SWEEP MODE :::::
# Re-train probe and compute metrics for a series of values of the number hidden units.
parser_sweep = subparsers.add_parser("sweep", description="Train and compute metrics for a list of values \
                                     for some hyperparameter.")
# 0 is logistic regression (no hidden layers)
parser_sweep.add_argument("--depths", type=int, nargs="+", help="List of layers", default=[0])
parser_sweep.add_argument("--widths", type=int, nargs="+", help="List of hidden units.", default=[50])
parser_sweep.add_argument('--n_deltas', help='number of deltas to try', default=13, type=int)
parser_sweep.add_argument('--logd_min', help='min log delta', default=-3.0, type=float)
parser_sweep.add_argument('--logd_max', help='max log delta', default=3.0, type=float)
parser_sweep.set_defaults(func=commands.sweep)

# ::::: MARGLIK MODE :::::
# Re-train probe with marglik objective and compute simple metrics
parser_marglik = subparsers.add_parser("marglik", description="Marglik based training of various architectures.")
# 0 is logistic regression (no hidden layers)
parser_marglik.add_argument("--trainer-lr-hyp", type=float, default=1e-2, help="Batch size of trainer.")
parser_marglik.add_argument("--depths", type=int, nargs="+", help="List of layers", default=[0])
parser_marglik.add_argument("--widths", type=int, nargs="+", help="List of hidden units.", default=[100])
parser_marglik.add_argument("--posterior-structure", choices=['diag', 'kron'], default='diag')
parser_marglik.add_argument("--activation", choices=['relu', 'tanh'], default='tanh')
parser_marglik.add_argument("--nostop", action='store_true')
parser_marglik.set_defaults(func=commands.marglik_sweep)


args = parser.parse_args()
args.func(args)
