import pandas as pd
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import numpy as np
import torch
from argparse import ArgumentParser
from pathlib import Path
import pickle
from torch_scatter import scatter_mean
import fasttext
from utils import get_config, get_sentence_model_mapping
from nltk import Tree


model_mapping = get_sentence_model_mapping()


def batch(iterable, batch_size):
    """
    Given an iterable, yields it in batches of `batch_size`.

    Source: https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    """
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx:min(ndx + batch_size, length)]


def embed_sentences_transformer(sentences, tokenizer, model, device):
    sentences = list(sentences)

    # Obtain representations
    tokenized_sentences = tokenizer(sentences, truncation=True, padding=True, return_tensors="pt",
                                    return_special_tokens_mask=True).to(device)
    special_tokens_mask = tokenized_sentences.pop("special_tokens_mask")
    outputs = model(**tokenized_sentences, return_dict=True)

    # Apply pooling
    if pooling_strategy == "pool":
        pooled = outputs["pooler_output"]
    elif pooling_strategy == "mean":
        pooled = scatter_mean(outputs["last_hidden_state"], special_tokens_mask, dim=1, dim_size=2)[:, 0]
    elif pooling_strategy == "cls":
        pooled = outputs["last_hidden_state"][:, 0, :]
    else:
        raise ValueError("Invalid pooling strategy.")

    assert pooled.shape == (len(sentences), outputs["last_hidden_state"].shape[2])

    # Split and numpy
    return [x.squeeze(0).numpy() for x in pooled.cpu().split(dim=0, split_size=1)]


def embed_sentences_fasttext(sentences, model):
    # Tokenize using pre-parsed inputs
    tokenized_sentences = [Tree.fromstring(s).leaves() for s in sentences]

    # Apply pooling
    if pooling_strategy == "pool":
        raise ValueError(f"Only mean pooling is supported for '{args.model}'.")
    elif pooling_strategy == "mean":
        # TODO: Implement mean pooling
        pooled_array = [torch.tensor([model[w] for w in s]).mean(dim=0).unsqueeze(0) for s in tokenized_sentences]
        pooled = torch.cat(pooled_array, dim=0)
    elif pooling_strategy == "cls":
        raise ValueError(f"Only mean pooling is supported for '{args.model}'.")
    else:
        raise ValueError("Invalid pooling strategy.")

    assert pooled.shape == (len(sentences), 300)

    # Split and numpy
    return [x.squeeze(0).numpy() for x in pooled.cpu().split(dim=0, split_size=1)]


parser = ArgumentParser(description="Preprocesses and embeds MultiNLI data. This is done by converting \
                        the BERT representations into a sentence-level representation.")
parser.add_argument("split", choices=["train", "dev", "test"],
                    help="Which split to preprocess.")
parser.add_argument("output", type=str, help="File to store preprocessed outputs to.")
parser.add_argument("--model", default="bert", choices=list(model_mapping.keys()),
                    help="The name of the pretrained model to use.")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size to use when computing \
                    representations.")
parser.add_argument("--pooling-strategy", default="mean", choices=["pool", "mean", "cls"],
                    help="Which pooling strategy to use to obtain sentence-level representations out of the \
                    subtoken level representations. `mean` computes the mean over all subtoken \
                    representations. `pool` uses the pretrained representations pooler_output (e.g., for \
                    BERT this is a linear layer + tanh activation, trained on the next sentence prediction \
                    task). `cls` takes the activation of the [CLS] token.")
args = parser.parse_args()

split = args.split
batch_size = args.batch_size
pooling_strategy = args.pooling_strategy
device = 0 if torch.cuda.is_available() else "cpu"

# Load data
data = pd.read_json(f"data/multinli_1.0/multinli_1.0_{split}.jsonl", lines=True)

input_type = ""
if args.model in ["bert", "roberta", "albert", "xlnet"]:
    # Load pretrained model
    model_name = model_mapping[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    embed_sentences = lambda sents: embed_sentences_transformer(sents, tokenizer, model, device)  # noqa
elif args.model == "t5":
    # Load pretrained T5 model specifically for encoding (the standard model is encoder-decoder)
    model_name = model_mapping[args.model]
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()
    embed_sentences = lambda sents: embed_sentences_transformer(sents, tokenizer, model, device)  # noqa
elif args.model == "fasttext":
    model_file = Path(get_config()["data"]["fasttext_root"]) / "cc.en.300.bin"
    model = fasttext.load_model(str(model_file))
    embed_sentences = lambda sents: embed_sentences_fasttext(sents, model)  # noqa
    input_type = "_parse"

outputs = []
for zipped in tqdm(
        list(batch(list(zip(data[f"sentence1{input_type}"].tolist(), data[f"sentence2{input_type}"].tolist(),
                        data["gold_label"].tolist())), batch_size))):

    sentences1, sentences2, gold_labels = zip(*zipped)

    # Get representations for batch
    with torch.no_grad():
        repr_sentences1 = embed_sentences(sentences1)
        repr_sentences2 = embed_sentences(sentences2)

    assert len(sentences1) == len(repr_sentences1)
    assert len(sentences2) == len(repr_sentences2)

    # Store results
    for sent1, repr1, sent2, repr2, nli_tag in zip(
            sentences1, repr_sentences1, sentences2, repr_sentences2, gold_labels):
        concat_repr = np.concatenate([repr1, repr2], axis=0)
        outputs += [{
            "premise": f"{sent1}",
            "hypothesis": f"{sent2}",
            "embedding": concat_repr,
            "attributes": {
                "nli": nli_tag,
            }
        }]

# Save results
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
with open(args.output, "wb") as h:
    pickle.dump(outputs, h)
