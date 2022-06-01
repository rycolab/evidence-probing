from typing import Type, Any, Dict
from itertools import product
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from probekit.models.probe import Probe
from probekit.models.discriminative.neural_probe import NeuralProbe

from trainers.convergence_trainer import ConvergenceTrainer
from trainers.marglik_trainer import MargLikTrainer
from models.mlp import MLPS

from utils import load_word_lists_for_language, get_classification_datasets, setup_probe, get_metrics, \
    setup_metrics, create_word_item_map, make_word_list, create_word_item_map_arcs, setup_simple_metrics
import pprint
import json
from pathlib import Path


def manual(args):
    raise Warning('Manual unmaintained')
    # ::::: SETUP :::::
    print(f"Setting up datasets ({args.language}, {args.attribute}) and probes...")
    word_lists = load_word_lists_for_language(args.language)

    # FIXME: can be done better once we have other contextualizers within `load_word_lists`
    if args.contextualizer == 'random':
        def randomize_embeddings(word_list):
            for w in word_list:
                w._embedding = torch.randn_like(w._embedding)
        randomize_embeddings(word_lists['train'])
        randomize_embeddings(word_lists['dev'])
        randomize_embeddings(word_lists['test'])

    # TODO: add argument and option to split by word instead of by context.
    attribute = 'Part of Speech' if args.attribute == 'POS' else args.attribute
    attribute = 'Gender and Noun Class' if args.attribute == 'Gender' else attribute
    datasets = get_classification_datasets(args, word_lists, attribute, pca=args.pca)
    dataset_train, dataset_dev, dataset_test = datasets["train"], datasets["dev"], datasets["test"]  # noqa
    embedding_size = dataset_train.get_dimensionality()

    setup = setup_probe(args, dataset_train, dims=embedding_size)
    trainer, probe_family = setup["trainer"], setup["probe_family"]

    # NOTE: Comment out/delete the appropriate lines depending on whether the metrics should be computed
    #       w.r.t. the train/dev/test sets.
    metrics = setup_metrics(dataset_train, dataset_test, args.regularization)

    # ::::: RUN PROBE & COMPUTE METRICS :::::
    print("Probing...")
    metrics = get_metrics(trainer=trainer, probe_family=probe_family, metrics=metrics,
                          dims=None)

    # Log to console
    pp = pprint.PrettyPrinter(indent=4)
    print("Results:")
    pp.pprint(metrics)

    # Save to file
    if args.output_file:
        with open(args.output_file, "w") as h:
            json.dump(metrics, h)

        print(f"Saved output to '{args.output_file}'.")


def sweep(args):
    # ::::: SETUP :::::
    print(f"Setting up datasets ({args.language}, {args.attribute}) and probes...")
    deltas = np.logspace(args.logd_min, args.logd_max, num=args.n_deltas)

    word_lists = load_word_lists_for_language(args)

    if args.word_split:
        if args.task_type == 'arc':
            raise ValueError('For arc labeling, splitting by words is not available.')
        # TODO: on clean up, add to load_word_lists?
        # update word_lists with new split
        word_list = word_lists['train'] + word_lists['test'] + word_lists['dev']
        word_item_map, words = create_word_item_map(word_list)
        train_words, val_words = train_test_split(words, test_size=0.35, shuffle=True, random_state=args.seed)
        test_words, dev_words = train_test_split(val_words, test_size=0.5, shuffle=True, random_state=args.seed)
        word_lists = dict(train=make_word_list(word_list, word_item_map, train_words),
                          test=make_word_list(word_list, word_item_map, test_words),
                          dev=make_word_list(word_list, word_item_map, dev_words))

    # FIXME: can be done better once we have other contextualizers within `load_word_lists`
    if args.contextualizer == 'random_context':
        # random representation per context
        def randomize_embeddings(word_list):
            for w in word_list:
                w.randomize()
        randomize_embeddings(word_lists['train'])
        randomize_embeddings(word_lists['dev'])
        randomize_embeddings(word_lists['test'])
    elif args.contextualizer == 'random_word':
        # random representation per word
        word_list = word_lists['train'] + word_lists['dev'] + word_lists['test']
        if args.task_type == 'token':
            word_item_map, _ = create_word_item_map(word_list)
            for w in word_item_map:
                # randomize first embedding
                ixs = word_item_map[w]
                first_word = word_list[ixs[0]]
                first_word.randomize()
                embedding = first_word.get_embedding()
                for ix in ixs[1:]:
                    # set for all other embeddings of the same word
                    word_list[ix].set_embedding(embedding.clone())
        elif args.task_type == 'arc':
            word_item_map, _ = create_word_item_map_arcs(word_list)
            rep_size = int(len(word_list[0].get_embedding()) / 2)
            example = word_list[0].get_embedding()[:rep_size]
            for w in word_item_map:
                embedding = torch.randn_like(example)
                ixs = word_item_map[w]
                for ix in ixs:
                    i, node = ix
                    if node == 0:  # set head
                        word_list[i].set_head_embedding(embedding)
                    elif node == 1:  # set tail
                        word_list[i].set_tail_embedding(embedding)
                    else:
                        raise ValueError('Invalid index')

    attribute = 'Part of Speech' if args.attribute == 'POS' else args.attribute
    attribute = 'Gender and Noun Class' if args.attribute == 'Gender' else attribute
    datasets = get_classification_datasets(args, word_lists, attribute)
    dataset_train, dataset_dev, dataset_test = datasets["train"], datasets["dev"], datasets["test"]  # noqa
    embedding_size = dataset_train.get_dimensionality()

    device = "cuda:0" if args.gpu else "cpu"
    N = dataset_train._embeddings_tensor_concat.size(0)

    all_metrics: Dict[int, Any] = {'N': N}
    all_params = list(product(args.depths, args.widths, deltas))
    for idx, params in enumerate(all_params):
        depth, width, delta = params
        hidden_sizes = [width] * depth
        neural_probe_model = MLPS(input_size=embedding_size, output_size=len(dataset_train.keys()),
                                  hidden_sizes=hidden_sizes, activation='tanh').to(device)

        trainer = ConvergenceTrainer(
            model=neural_probe_model, dataset=dataset_train, device=device, lr=args.step_size,
            num_epochs=args.trainer_num_epochs, report_progress=not args.quiet, l2_regularization=delta)

        probe_family: Type[Probe] = NeuralProbe

        # ::::: EVALUATING THE PROBE :::::
        # We figure out what the parameters (the "specification") of the probe should be
        # print("Probing...")

        eval_metrics = setup_metrics(dataset_train, dataset_test, delta, device="cpu")
        metrics = get_metrics(trainer=trainer, probe_family=probe_family, metrics=eval_metrics,
                              dims=None)
        all_metrics[params] = metrics
        print(f"Done ({idx + 1}/{len(all_params)}):\t", params)
        print({k: v for k, v in metrics.items() if k != "losses"})
        print()

    # Save to file
    if args.output_file:
        outfile = Path("results") / Path(args.output_file)
        outfile.parent.mkdir(parents=True, exist_ok=True)

        # unfortunately json is not so flexible, convert to string (biject)
        all_metrics = {"-".join(str(val) for val in k): v for k, v in all_metrics.items()}
        with open(outfile, "w") as h:
            json.dump(all_metrics, h)

        print(f"Saved output to '{args.output_file}'.")


def marglik_sweep(args):
    # ::::: SETUP :::::
    print(f"Setting up datasets ({args.language}, {args.attribute}) and probes...")
    word_lists = load_word_lists_for_language(args)

    if args.word_split:
        if args.task_type == 'arc':
            raise ValueError('For arc labeling, splitting by words is not available.')
        # TODO: on clean up, add to load_word_lists?
        # update word_lists with new split
        word_list = word_lists['train'] + word_lists['test'] + word_lists['dev']
        word_item_map, words = create_word_item_map(word_list)
        train_words, val_words = train_test_split(words, test_size=0.35, shuffle=True, random_state=args.seed)
        test_words, dev_words = train_test_split(val_words, test_size=0.5, shuffle=True, random_state=args.seed)
        word_lists = dict(train=make_word_list(word_list, word_item_map, train_words),
                          test=make_word_list(word_list, word_item_map, test_words),
                          dev=make_word_list(word_list, word_item_map, dev_words))

    # FIXME: can be done better once we have other contextualizers within `load_word_lists`
    if args.contextualizer == 'random_context':
        # random representation per context
        def randomize_embeddings(word_list):
            for w in word_list:
                w.randomize()
        randomize_embeddings(word_lists['train'])
        randomize_embeddings(word_lists['dev'])
        randomize_embeddings(word_lists['test'])
    elif args.contextualizer == 'random_word':
        # random representation per word
        word_list = word_lists['train'] + word_lists['dev'] + word_lists['test']
        if args.task_type == 'token':
            word_item_map, _ = create_word_item_map(word_list)
            for w in word_item_map:
                # randomize first embedding
                ixs = word_item_map[w]
                first_word = word_list[ixs[0]]
                first_word.randomize()
                embedding = first_word.get_embedding()
                for ix in ixs[1:]:
                    # set for all other embeddings of the same word
                    word_list[ix].set_embedding(embedding.clone())
        elif args.task_type == 'arc':
            word_item_map, _ = create_word_item_map_arcs(word_list)
            rep_size = int(len(word_list[0].get_embedding()) / 2)
            example = word_list[0].get_embedding()[:rep_size]
            for w in word_item_map:
                embedding = torch.randn_like(example)
                ixs = word_item_map[w]
                for ix in ixs:
                    i, node = ix
                    if node == 0:  # set head
                        word_list[i].set_head_embedding(embedding)
                    elif node == 1:  # set tail
                        word_list[i].set_tail_embedding(embedding)
                    else:
                        raise ValueError('Invalid index')

    attribute = 'Part of Speech' if args.attribute == 'POS' else args.attribute
    attribute = 'Gender and Noun Class' if args.attribute == 'Gender' else attribute
    datasets = get_classification_datasets(args, word_lists, attribute)
    dataset_train, dataset_dev, dataset_test = datasets["train"], datasets["dev"], datasets["test"]  # noqa
    embedding_size = dataset_train.get_dimensionality()

    device = "cuda:0" if args.gpu else "cpu"
    N = dataset_train._embeddings_tensor_concat.size(0)

    all_metrics: Dict[int, Any] = {'N': N}
    all_params = list(product(args.depths, args.widths))
    for idx, params in enumerate(all_params):
        depth, width = params
        hidden_sizes = [width] * depth
        neural_probe_model = MLPS(input_size=embedding_size, output_size=len(dataset_train.keys()),
                                  hidden_sizes=hidden_sizes, activation=args.activation).to(device)

        trainer = MargLikTrainer(
            model=neural_probe_model, dataset=dataset_train, device=device, lr=args.step_size,
            num_epochs=args.trainer_num_epochs, report_progress=not args.quiet, 
            batch_size=args.trainer_batch_size, lr_hyp=args.trainer_lr_hyp, 
            posterior_structure=args.posterior_structure, early_stopping=not args.nostop)

        probe_family: Type[Probe] = NeuralProbe

        # ::::: EVALUATING THE PROBE :::::
        eval_metrics = setup_simple_metrics(dataset_train, dataset_test, device="cpu")
        metrics = get_metrics(trainer=trainer, probe_family=probe_family, metrics=eval_metrics, dims=None)
        metrics['marglik'] = trainer._marglik
        metrics['effective_dimensions'] = trainer._effective_dimensions
        metrics['prior_precision'] = trainer._prior_precision
        metrics['effective_parameters'] = trainer._effective_parameters
        metrics['network_parameters'] = trainer._network_parameters
        metrics['losses'], metrics['margliks'] = metrics['losses']
        all_metrics[params] = metrics
        print(f"Done ({idx + 1}/{len(all_params)}):\t", params)
        print({k: v for k, v in metrics.items() 
               if k not in ['losses', 'margliks', 'prior_precision', 'effective_parameters', 'network_parameters']})
        print()

    # Save to file
    if args.output_file:
        outfile = Path("results") / Path(args.output_file)
        outfile.parent.mkdir(parents=True, exist_ok=True)

        # unfortunately json is not so flexible, convert to string (biject)
        all_metrics = {"-".join(str(val) for val in k): v for k, v in all_metrics.items()}
        with open(outfile, "w") as h:
            json.dump(all_metrics, h)

        print(f"Saved output to '{args.output_file}'.")

