import os
import json
import yaml
import torch
import spacy
import numpy as np
from itertools import chain
from collections import Counter
from tqdm.auto import tqdm

def build_nltk_tokenizer(vocab):
    raise NotImplementedError()

def build_spacy_tokenizer(vocab):
    # get german tokenizer
    from spacy.lang.de import German
    # build tokenizer parameters
    prefixes = German.Defaults.prefixes
    suffixes = German.Defaults.suffixes
    infixes = German.Defaults.infixes
    prefix_search = spacy.util.compile_prefix_regex(prefixes).search if prefixes else None
    suffix_search = spacy.util.compile_suffix_regex(suffixes).search if suffixes else None
    infix_finditer = spacy.util.compile_infix_regex(infixes).finditer if infixes else None
    # add tokenizer exception for special tokens
    exc = German.Defaults.tokenizer_exceptions
    exc = spacy.util.update_exc(exc, {
        '[SEP]': [{spacy.symbols.ORTH: "[SEP]"}]
    })
    # create tokenizer
    return spacy.tokenizer.Tokenizer(
        vocab=spacy.vocab.Vocab(strings=vocab.keys()),
        rules=exc,
        prefix_search=prefix_search,
        suffix_search=suffix_search,
        infix_finditer=infix_finditer,
        token_match=German.Defaults.token_match,
        url_match=German.Defaults.url_match
    )


def tokenize(tokenizer, texts):
    """ tokenize all given texts """ 
    return [
        tuple(map(lambda t: str(t).lower(), tokenizer(text)))
        for text in tqdm(texts, "Tokenizing")
    ]

def truncate_pad(tokenized_texts, max_length=256, padding_token="[PAD]".lower()):
    # truncate and pad all tokenized texts to match the `max_legth`
    return [
        tokens[:max_length] + (padding_token,) * max(0, max_length - len(tokens))
        for tokens in tokenized_texts
    ]

def filter_vocab(vocab, embed, tokenized_texts, min_freq=1, max_size=200_000):
    # count token occurances and ignore tokens
    # that are not in the vocabulary
    counter = Counter(chain(*tokenized_texts))
    # create filtered vocabulary containing the most frequent words
    filtered_vocab = [
        word
        for word, freq in counter.most_common()
        if (freq > min_freq) and (word in vocab)
    ]
    filtered_vocab = filtered_vocab[:max_size]
    # filtered_vocab = counter.most_common(max_size)
    # filtered_vocab = [w for w, f in filtered_vocab if f >= min_freq]
    # add special tokens
    if "[SEP]".lower() in filtered_vocab:
        filtered_vocab.remove("[SEP]".lower())
    if "[UNK]".lower() in filtered_vocab:
        filtered_vocab.remove("[UNK]".lower())
    if "[PAD]".lower() in filtered_vocab:
        filtered_vocab.remove("[PAD]".lower())
    filtered_vocab.insert(0, "[SEP]".lower())
    filtered_vocab.insert(0, "[UNK]".lower())
    filtered_vocab.insert(0, "[PAD]".lower())
    # build embedding matrix for filtered vocab
    filtered_embed = [
        embed[vocab[token]] if token in vocab else np.random.uniform(-1, 1, size=(embed.shape[1],))
        for token in filtered_vocab
    ]
    filtered_embed = np.stack(filtered_embed, axis=0)
    # create mapping for filtered vocab
    filtered_vocab = {token: i for i, token in enumerate(filtered_vocab)}
    assert len(filtered_vocab) == filtered_embed.shape[0]
    # return
    return filtered_vocab, filtered_embed

def convert_tokens_to_ids(vocab, tokenized_texts):
    unk_token_id = vocab["[unk]"]
    return [
        [vocab.get(t.lower(), unk_token_id) for t in tokens]
        for tokens in tokenized_texts
    ]


if __name__ == '__main__':

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Preprocess the texts and build the vocabulary.")
    parser.add_argument("--train-texts", type=str, help="Path to the prepared train texts file.")
    parser.add_argument("--train-labels", type=str, help="Path to the training labels file.")
    parser.add_argument("--val-texts", type=str, help="Path to the prepared validation texts file.")
    parser.add_argument("--val-labels", type=str, help="Path to the validation labels file.")
    parser.add_argument("--test-texts", type=str, help="Path to the prepared test texts file.")
    parser.add_argument("--test-labels", type=str, help="Path to the test labels file.")
    parser.add_argument("--tokenizer", type=str, choices=["Spacy", "NLTK"], help="Specify the tokenizer to use.")
    parser.add_argument("--max-length", type=int, help="The maximum allowed number of tokens per input text.")
    parser.add_argument("--pretrained-vocab", type=str, help="Path to the pretrained vocab.")
    parser.add_argument("--pretrained-embed", type=str, help="Path to the pretrained embedding vectors.")
    parser.add_argument("--output-dir", type=str, help="Output directory.")
    # parser arguments
    args = parser.parse_args()

    # load pretrained embeddings
    vocab = np.load(args.pretrained_vocab)
    embed = np.load(args.pretrained_embed)
    # change special tokens
    vocab[vocab == "<SEP>"] = "[SEP]"
    vocab[vocab == "<PAD>"] = "[PAD]"
    vocab[vocab == "<UNK>"] = "[UNK]"
    # convert vocab to list
    vocab = {token.lower(): i for i, token in enumerate(vocab.tolist())}

    # get the tokenizer
    tokenizer = {
        "Spacy": build_spacy_tokenizer,
        "NLTK": build_nltk_tokenizer
    }[args.tokenizer](vocab)

    # load train texts
    with open(args.train_texts, "r") as f:
        train_texts = f.readlines()
    # tokenizer train texts
    train_tokenized = tokenize(tokenizer, train_texts)
    train_tokenized = truncate_pad(train_tokenized, max_length=args.max_length)
    # filter vocabulary to keep only the tokens that actually occur in the training set
    filtered_vocab, filtered_embed = filter_vocab(vocab, embed, train_tokenized)
    pad_token_id = filtered_vocab["[PAD]".lower()]
    # build train input features
    train_input_ids = torch.LongTensor(convert_tokens_to_ids(filtered_vocab, train_tokenized))
    
    # load val texts
    with open(args.val_texts, "r") as f:
        val_texts = f.readlines()
    # build test input features
    val_tokenized = tokenize(tokenizer, val_texts)
    val_tokenized = truncate_pad(val_tokenized, max_length=args.max_length)
    val_input_ids = torch.LongTensor(convert_tokens_to_ids(filtered_vocab, val_tokenized))
   
    # load test texts
    with open(args.test_texts, "r") as f:
        test_texts = f.readlines()
    # build test input features
    test_tokenized = tokenize(tokenizer, test_texts)
    test_tokenized = truncate_pad(test_tokenized, max_length=args.max_length)
    test_input_ids = torch.LongTensor(convert_tokens_to_ids(filtered_vocab, test_tokenized))

    # compute ratio of unkown tokens in texts
    unk_token_id = filtered_vocab["[UNK]".lower()]
    n_train_unk = (train_input_ids == unk_token_id).sum()
    n_val_unk = (val_input_ids == unk_token_id).sum()
    n_test_unk = (test_input_ids == unk_token_id).sum()
    # build metrics dict
    metrics = {
        "vocab_size": len(filtered_vocab),
        "train":      {"unk_tokens_ratio": n_train_unk.item() / train_input_ids.numel()},
        "validation": {"unk_tokens_ratio": n_val_unk.item() / val_input_ids.numel()},
        "test":       {"unk_tokens_ratio": n_test_unk.item() / test_input_ids.numel()}
    }

    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # save metrics
    with open(os.path.join(args.output_dir, "metrics.json"), "w+") as f:
        f.write(json.dumps(metrics))
    # save vocab and embeddings
    with open(os.path.join(args.output_dir, "vocab.json"), "w+") as f:
        f.write(json.dumps(filtered_vocab))
    np.save(os.path.join(args.output_dir, "vectors.npy"), filtered_embed)

    # save train and test input ids to disk
    with open(args.train_labels, "r") as f:
        torch.save({
                'input-ids': train_input_ids,
                'labels': [labels.strip().split() for labels in f.readlines()] 
            }, os.path.join(args.output_dir, "train_data.pkl")
        )
    with open(args.val_labels, "r") as f:
        torch.save({
                'input-ids': val_input_ids,
                'labels': [labels.strip().split() for labels in f.readlines()] 
            }, os.path.join(args.output_dir, "val_data.pkl")
        )
    with open(args.test_labels, "r") as f:
        torch.save({
                'input-ids': test_input_ids,
                'labels': [labels.strip().split() for labels in f.readlines()] 
            }, os.path.join(args.output_dir, "test_data.pkl")
        )

