from typing import List, Union, Dict, Type, Any, Optional, Sequence, Mapping
import pickle
from tqdm import tqdm
import torch
from pathlib import Path
import yaml
import pycountry

from probekit.utils.types import ProbingUnit, Word, Arc, Inference, BooleanQuestion, COPAQuestion, PyTorchDevice
from probekit.utils.dataset import ClassificationDataset
from probekit.utils.transforms import PCATransform
from probekit.models.probe import Probe
from probekit.trainers.trainer import Trainer
from probekit.metrics.metric import Metric
from probekit.metrics.mutual_information import MutualInformation
from probekit.metrics.accuracy import Accuracy
from probekit.models.discriminative.neural_probe import NeuralProbe

from trainers.convergence_trainer import ConvergenceTrainer
from metrics.bic import BIC
from metrics.laplace_ggn import LaplaceGGN
from metrics.logpredlik import LogPredictiveLikelihood
from models.mlp import MLPS


def get_sentence_model_mapping():
    return {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",   # Same as in Pareto probing
        "albert": "albert-base-v2",  # Same as in Pareto probing
        "t5": "t5-base",
        "xlnet": "xlnet-base-cased",
        "fasttext": "cc.en.300.bin"
    }


def get_config():
    with open("config.yml", "r") as h:
        config = yaml.load(h, Loader=yaml.FullLoader)

    return config


def convert_pickle_to_word_list(path: Union[Path, str]) -> Sequence[Word]:
    # Load pickle file
    with open(path, "rb") as h:
        data = pickle.load(h)

    # Convert data to example collection
    word_list: List[Word] = []
    for w in tqdm(data):
        word_list.append(
            Word(w["word"], torch.tensor(w["embedding"]).squeeze(0), w["attributes"]))

    return word_list


def convert_pickle_to_arc_list(path: Union[Path, str]) -> Sequence[Arc]:
    # Load pickle file
    with open(path, "rb") as h:
        data = pickle.load(h)

    # Convert data to example collection
    word_list: List[Arc] = []
    for w in tqdm(data):
        word_list.append(
            Arc(w["head"], w["tail"], torch.tensor(w["embedding"]).squeeze(0), w["attributes"]))

    return word_list


def convert_pickle_to_inference_list(path: Union[Path, str]) -> Sequence[Arc]:
    # Load pickle file
    with open(path, "rb") as h:
        data = pickle.load(h)

    # Convert data to example collection
    word_list: List[Inference] = []
    for w in tqdm(data):
        word_list.append(
            Inference(w["premise"], w["hypothesis"], torch.tensor(w["embedding"]).squeeze(0), w["attributes"]))

    return word_list


def convert_pickle_to_boolq_list(path: Union[Path, str]) -> Sequence[BooleanQuestion]:
    # Load pickle file
    with open(path, "rb") as h:
        data = pickle.load(h)

    # Convert data to example collection
    word_list: List[BooleanQuestion] = []
    for w in tqdm(data):
        word_list.append(
            BooleanQuestion(w["question"], w["passage"], torch.tensor(w["embedding"]).squeeze(0), w["attributes"]))

    return word_list


def convert_pickle_to_copa_list(path: Union[Path, str]) -> Sequence[COPAQuestion]:
    # Load pickle file
    with open(path, "rb") as h:
        data = pickle.load(h)

    # Convert data to example collection
    word_list: List[COPAQuestion] = []
    for w in tqdm(data):
        word_list.append(
            COPAQuestion(w["premise"], w["choice1"], w["choice2"], torch.tensor(w["embedding"]).squeeze(0),
                         w["attributes"]))

    return word_list


def load_word_lists_for_language(args) -> Mapping[str, Sequence[ProbingUnit]]:
    if args.task_type == "arc" and args.attribute != "dep":
        raise ValueError("Only the 'dep' arc task is supported.")

    if args.task_type == "token":
        return load_intrinsic_word_lists_for_language(args)
    elif args.task_type == "arc":
        return load_pareto_arc_word_lists_for_language(args)
    elif args.task_type == "sentence":
        return load_sentence_word_lists_for_language(args)
    elif args.task_type in ["boolq", "cb", "copa", "rte"]:
        return load_superglue_word_lists_for_language(args)

    raise Exception("Invalid task type")


def load_intrinsic_word_lists_for_language(args) -> Mapping[str, Sequence[Word]]:
    lang_code = args.language

    # Read files and obtain word list
    config = get_config()
    data_root = Path(config["data"]["ud_treebanks_root"])

    lang_code_mapping = {
        "eng": ("English-EWT", "en_ewt", "en"),
        "tur": ("Turkish-IMST", "tr_imst", "tr"),
        "mar": ("Marathi-UFAL", "mr_ufal", "mr"),
        "ara": ("Arabic-PADT", "ar_padt", "ar"),
        "deu": ("German-GSD", "de_gsd", "de"),
        "zho": ("Chinese-GSDSimp", "zh_gsdsimp", "zh"),
    }
    if lang_code not in lang_code_mapping:
        raise ValueError(f"Unsupported language. Please choose one from the following list: {lang_code_mapping.keys()}")

    treebank, lang_code, shortcode = lang_code_mapping[lang_code]

    # TODO: add other word representations here (also random?)
    if args.contextualizer == "bert":
        embedding = "bert-base-multilingual-cased"
    elif args.contextualizer == "fasttext":
        embedding = f"cc.{shortcode}.300.bin"
    elif args.contextualizer in ["random_word", "random_context"]:
        print(f"Embedding was '{args.contextualizer}'. Using preprocessed word lists for BERT as a starting point,"
              "overriding them with random variants.")
        embedding = "bert-base-multilingual-cased"
    else:
        raise NotImplementedError(f"Embedding '{args.contextualizer}' is unsupported in this task.")

    file_path_train = data_root / f"UD_{treebank}/{lang_code}-um-train-{embedding}.pkl"
    file_path_dev = data_root / f"UD_{treebank}/{lang_code}-um-dev-{embedding}.pkl"
    file_path_test = data_root / f"UD_{treebank}/{lang_code}-um-test-{embedding}.pkl"

    print("Loading files:")
    print(f"\tTrain: {file_path_train}")
    print(f"\tDev: {file_path_dev}")
    print(f"\tTest: {file_path_test}")

    return {
        "train": convert_pickle_to_word_list(file_path_train),
        "dev": convert_pickle_to_word_list(file_path_dev),
        "test": convert_pickle_to_word_list(file_path_test)
    }


def load_pareto_arc_word_lists_for_language(args) -> Mapping[str, Sequence[Arc]]:
    lang_code = args.language

    # Read files and obtain word list
    config = get_config()
    data_root = Path(config["data"]["pareto_datasets_root"])

    language = pycountry.languages.get(alpha_3=lang_code).name.lower()

    if args.contextualizer == "bert":
        representation = "bert"
    elif args.contextualizer == "fasttext":
        representation = "fast"
    elif args.contextualizer in ["random_word", "random_context"]:
        print(f"Embedding was '{args.contextualizer}'. Using preprocessed word lists for BERT as a starting point,"
              "overriding them with random variants.")
        representation = "bert"
    else:
        raise NotImplementedError(f"Embedding '{args.contextualizer}' is unsupported in this task.")

    file_path_train = data_root / f"dep_label-{language}-{representation}-train.pkl"
    file_path_dev = data_root / f"dep_label-{language}-{representation}-valid.pkl"
    file_path_test = data_root / f"dep_label-{language}-{representation}-test.pkl"

    print("Loading files:")
    print(f"\tTrain: {file_path_train}")
    print(f"\tDev: {file_path_dev}")
    print(f"\tTest: {file_path_test}")

    return {
        "train": convert_pickle_to_arc_list(file_path_train),
        "dev": convert_pickle_to_arc_list(file_path_dev),
        "test": convert_pickle_to_arc_list(file_path_test)
    }


def load_sentence_word_lists_for_language(args) -> Mapping[str, Sequence[Word]]:
    lang_code = args.language

    if lang_code != "eng":
        raise ValueError("Sentence-level tasks only available for English.")

    # Read files and obtain word list
    config = get_config()
    data_root = Path(config["data"]["multinli_dataset_root"])

    # language = pycountry.languages.get(alpha_3=lang_code).name
    if args.contextualizer in ["bert", "fasttext", "t5", "xlnet", "roberta", "albert"]:
        representation = args.contextualizer
    elif args.contextualizer in ["random_context"]:
        print(f"Embedding was '{args.contextualizer}'. Using preprocessed word lists for BERT as a starting point,"
              "overriding them with random variants.")
        representation = "bert"
    else:
        raise NotImplementedError(f"Embedding '{args.contextualizer}' is unsupported in this task.")

    file_path_train = data_root / f"multinli_1.0_train_{representation}.pkl"
    file_path_dev = data_root / f"multinli_1.0_dev_{representation}.pkl"
    file_path_test = data_root / f"multinli_1.0_test_{representation}.pkl"

    print("Loading files:")
    print(f"\tTrain: {file_path_train}")
    print(f"\tDev: {file_path_dev}")
    print(f"\tTest: {file_path_test}")

    # Using the whole training set causes GPU problems right now due to limited memory.
    return {
        "train": convert_pickle_to_inference_list(file_path_train),
        "dev": convert_pickle_to_inference_list(file_path_dev),
        "test": convert_pickle_to_inference_list(file_path_test)
    }


def load_superglue_word_lists_for_language(args) -> Mapping[str, Sequence[Word]]:
    lang_code = args.language

    if lang_code != "eng":
        raise ValueError("Sentence-level tasks only available for English.")

    # Read files and obtain word list
    config = get_config()
    data_root = Path(config["data"]["superglue_datasets_root"])

    if args.contextualizer in ["bert", "fasttext", "t5", "xlnet", "roberta", "albert"]:
        representation = args.contextualizer
    elif args.contextualizer in ["random_context"]:
        print(f"Embedding was '{args.contextualizer}'. Using preprocessed word lists for BERT as a starting point,"
              "overriding them with random variants.")
        representation = "bert"
    else:
        raise NotImplementedError(f"Embedding '{args.contextualizer}' is unsupported in this task.")

    if args.task_type == "boolq":
        task_folder = data_root / "BoolQ"
        data_parser_fn = convert_pickle_to_boolq_list
    elif args.task_type == "cb":
        task_folder = data_root / "CB"
        data_parser_fn = convert_pickle_to_inference_list
    elif args.task_type == "copa":
        task_folder = data_root / "COPA"
        data_parser_fn = convert_pickle_to_copa_list
    elif args.task_type == "rte":
        task_folder = data_root / "RTE"
        data_parser_fn = convert_pickle_to_inference_list

    file_path_train = task_folder / f"train_{representation}.pkl"
    file_path_dev = task_folder / f"val_{representation}.pkl"

    print("Loading files:")
    print(f"\tTrain: {file_path_train}")
    print(f"\tDev (used ALSO as test): {file_path_dev}")
    print("\t\tNB: SuperGLUE datasets do not have a test set; hence we only use their dev set")
    # print(f"\tTest: {file_path_test}")  # SuperGLUE datasets do not have labelled test sets

    # Using the whole training set causes GPU problems right now due to limited memory.
    return {
        "train": data_parser_fn(file_path_train),
        "dev": data_parser_fn(file_path_dev),
        "test": data_parser_fn(file_path_dev)
    }


def get_classification_datasets(
        args, unit_lists: Mapping[str, Sequence[ProbingUnit]], attribute: str,
        pca: Optional[int] = None) -> Dict[str, ClassificationDataset]:
    device = "cuda:0" if args.gpu else "cpu"
    units_train, units_dev, units_test = unit_lists["train"], unit_lists["dev"], unit_lists["test"]

    # Figure out list of all values, for this property, that meet the minimum threshold of 20 instances
    min_count = 20
    property_value_list = ClassificationDataset.get_property_value_list(
        attribute, units_train, units_dev, units_test, min_count=min_count)
    print("Possible classes:", property_value_list)

    if len(property_value_list) <= 1:
        raise Exception(f"Not enough classes to run an experiment. This attribute has {len(property_value_list)} "
                        f"classes with at least {min_count} examples across all splits (need at least 2).")

    transform = None
    if pca is not None:
        pca_dataset = ClassificationDataset.from_unit_list(
            units_train, attribute=attribute, device=device, property_value_list=property_value_list)
        transform = PCATransform.from_dataset(pca_dataset, num_components=pca)

    # Convert unit list into ClassificationDataset. This is the type of the datasets expected by the tool
    # (it's just a dictionary, where the keys are the values of the property, e.g., "Singular", "Plural",
    # and the values are a list of training examples)
    dataset_train = ClassificationDataset.from_unit_list(
        units_train, attribute=attribute, device=device, transform=transform,
        property_value_list=property_value_list)
    dataset_dev = ClassificationDataset.from_unit_list(
        units_dev, attribute=attribute, device=device, transform=transform,
        property_value_list=property_value_list)
    dataset_test = ClassificationDataset.from_unit_list(
        units_test, attribute=attribute, device=device, transform=transform,
        property_value_list=property_value_list)

    return {
        "train": dataset_train,
        "dev": dataset_dev,
        "test": dataset_test
    }


def get_metrics(trainer: Trainer, probe_family: Type[Probe], metrics: Dict[str, Metric],
                dims) -> Dict[str, float]:
    # NOTE: Depending on the trainer used, training may actually happen when get_specification() is called.
    #       This should not matter for us in this case as we are computing metrics after training.
    losses = trainer._train_for_dimensions(dims)

    specification = trainer.get_specification()

    # The approximation requires a lot of memory in some cases, so we can't do it on the GPU
    # device = trainer.get_device()
    device = "cpu"
    probe = probe_family.from_specification(specification, device=device)

    # Compute & report metrics
    metrics = {metric_name: metric.compute(probe, dims) for metric_name, metric in metrics.items()}
    return {**metrics, **dict(losses=losses)}


def setup_simple_metrics(train_dataset: ClassificationDataset, test_dataset: ClassificationDataset,
                         device: Optional[PyTorchDevice] = None) -> Dict[str, Metric]:
    if device is not None:
        train_dataset = ClassificationDataset.from_dataset(train_dataset, device=device)
        test_dataset = ClassificationDataset.from_dataset(test_dataset, device=device)

    return {
        "mi_train": MutualInformation(dataset=train_dataset, bits=True, normalize=True),
        "mi_test": MutualInformation(dataset=test_dataset, bits=True, normalize=True),
        "loglik_train": LogPredictiveLikelihood(dataset=train_dataset),
        "loglik_test": LogPredictiveLikelihood(dataset=test_dataset),
        "accuracy_train": Accuracy(dataset=train_dataset),
        "accuracy_test": Accuracy(dataset=test_dataset),
        # complexity metrics/marginal likelihoods
        "bic": BIC(dataset=train_dataset),
    }


def setup_metrics(train_dataset: ClassificationDataset, test_dataset: ClassificationDataset,
                  regularization: float, device: Optional[PyTorchDevice] = None) -> Dict[str, Metric]:
    if device is not None:
        train_dataset = ClassificationDataset.from_dataset(train_dataset, device=device)
        test_dataset = ClassificationDataset.from_dataset(test_dataset, device=device)

    return {
        "mi_train": MutualInformation(dataset=train_dataset, bits=True, normalize=True),
        "mi_test": MutualInformation(dataset=test_dataset, bits=True, normalize=True),
        "loglik_train": LogPredictiveLikelihood(dataset=train_dataset),
        "loglik_test": LogPredictiveLikelihood(dataset=test_dataset),
        "accuracy_train": Accuracy(dataset=train_dataset),
        "accuracy_test": Accuracy(dataset=test_dataset),
        # complexity metrics/marginal likelihoods
        "bic": BIC(dataset=train_dataset),
        "laplace_ggn_diag": LaplaceGGN(dataset=train_dataset, prior_precision=regularization,
                                       cov_type='diag'),
        "effective_params": LaplaceGGN(dataset=train_dataset, prior_precision=regularization,
                                       cov_type='diag', compute_effective_parameters=True),
        # both below are too slow for a grid search, the diagonal often ends up nan anyway.
        # "laplace_ggn": LaplaceGGN(dataset=train_dataset, prior_precision=regularization,
        #                           cov_type='full'),
        # "laplace_diag": Laplace(dataset=train_dataset, prior_precision=regularization,
        #                         cov_type='diag')
    }


def setup_probe(args, dataset_train: ClassificationDataset, report_prog=True, dims=None) -> Dict[str, Any]:
    # We use a general neural probe
    device = "cuda:0" if args.gpu else "cpu"
    e = dataset_train.get_dimensionality() if dims is None else len(dims)
    torch.manual_seed(args.seed)

    hidden_sizes = [args.probe_num_hidden_units] * args.probe_num_layers
    neural_probe_model = MLPS(input_size=e, output_size=len(dataset_train.keys()),
                              hidden_sizes=hidden_sizes, activation='tanh').to(device)

    # neural_probe_model = MLPProbeModel(
    #     embedding_size=e, num_classes=len(dataset_train.keys()),
    #     hidden_size=args.probe_num_hidden_units, num_layers=args.probe_num_layers).to(device)

    trainer = ConvergenceTrainer(
        model=neural_probe_model, dataset=dataset_train, device=device, num_epochs=args.trainer_num_epochs,
        report_progress=report_prog, l2_regularization=args.regularization)
    probe_family: Type[Probe] = NeuralProbe

    return {
        "trainer": trainer,
        "probe_family": probe_family,
    }


def create_word_item_map(word_list):
    words = sorted(list({e.get_word() for e in word_list}))
    word_item_map = {w: list() for w in words}
    for i, w in enumerate(word_list):
        word = w.get_word()
        word_item_map[word].append(i)
    return word_item_map, words


def create_word_item_map_arcs(word_list):
    words = sorted(list({w for dep in word_list for w in [dep.get_head(), dep.get_tail()]}))
    word_item_map = {w: list() for w in words}
    for i, w in enumerate(word_list):
        w1 = w.get_head()
        w2 = w.get_tail()
        word_item_map[w1].append((i, 0))
        word_item_map[w2].append((i, 1))
    return word_item_map, words


def make_word_list(word_list, word_item_map, words):
    return [word_list[ix] for w in words for ix in word_item_map[w]]
