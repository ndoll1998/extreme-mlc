import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

def get_tokenizer(tokenizer_type, tokenizer_name):
    # get the sentence transformer tokenizer    
    if tokenizer_type == 'sentence-transformers':
        return AutoTokenizer.from_pretrained("sentence-transformers/" + tokenizer_name)   

    raise NotImplementedError()

def preprocess(dpath, tokenizer, max_seq_length, max_inst_count):
    
    # read the data
    df = pd.read_csv(dpath, index_col=0)
    # tokenize all instances
    instance_ids = []
    instance_labels = []
    # get all text columns
    text_columns = [name for name in df.columns if name != 'labels']

    for _, row in tqdm(df.iterrows(), desc=dpath, total=len(df.index)):
        # get the texts from the current row
        texts = [row[name] for name in text_columns]
        texts = [text if isinstance(text, str) else "" for text in texts]
        # encode texts
        enc = tokenizer.batch_encode_plus(
            texts[:max_inst_count],
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # add to lists
        instance_ids.append(enc.input_ids)
        instance_labels.append(row['labels'].strip().split())

    # stack tensors
    instance_ids = torch.stack(instance_ids, dim=0)
    # return preprocessed data
    return {
        'input-ids': instance_ids,
        'labels': instance_labels,
    }
 
if __name__ == '__main__':

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Prepare data for multi-instance learning scenario.")
    parser.add_argument("--train-csv", type=str, help="Path to the csv file containing the train data.")
    parser.add_argument("--val-csv", type=str, help="Path to the csv file containing the validation data.")
    parser.add_argument("--test-csv", type=str, help="Path to the csv file containing the test data.")
    parser.add_argument("--tokenizer-type", type=str, help="Which tokenizer type to use.")
    parser.add_argument("--tokenizer-name", type=str, help="The name/directory of the pretrained tokenizer.")
    parser.add_argument("--max-seq-length", type=int, help="Maximum size of an input sequence.")
    parser.add_argument("--max-inst-count", type=int, help="Maximum number of instances.")
    parser.add_argument("--output-dir", type=str, help="Output Directory")
    # parse arguments
    args = parser.parse_args()
    
    # get the tokenizer
    tokenizer = get_tokenizer(args.tokenizer_type, args.tokenizer_name)
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # preprocess the datasets
    train = preprocess(dpath=args.train_csv, tokenizer=tokenizer, max_seq_length=args.max_seq_length, max_inst_count=args.max_inst_count)
    torch.save(train, os.path.join(args.output_dir, "train_data.pkl"))
    val = preprocess(dpath=args.val_csv, tokenizer=tokenizer, max_seq_length=args.max_seq_length, max_inst_count=args.max_inst_count)
    torch.save(val, os.path.join(args.output_dir, "val_data.pkl"))
    test = preprocess(dpath=args.test_csv, tokenizer=tokenizer, max_seq_length=args.max_seq_length, max_inst_count=args.max_inst_count)
    torch.save(test, os.path.join(args.output_dir, "test_data.pkl"))
