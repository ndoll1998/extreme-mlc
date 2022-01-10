import os
import numpy as np
import pandas as pd

def csv_split_train_val(
    source_file:str,
    validation_size:int,
    output_dir:str
) -> None:
   
    # read input csv file
    df = pd.read_csv(source_file, index_col=0)

    validation_size = min(validation_size, len(df.index) // 10)
    # randomly select samples
    val_df = df.sample(n=validation_size)
    train_df = df.drop(val_df.index)

    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    # save splits in output directory
    train_df.to_csv(os.path.join(output_dir, "train-split.csv"))
    val_df.to_csv(os.path.join(output_dir, "val-split.csv"))

def txt_split_train_val(
    texts_file:str,
    labels_file:str,
    validation_size:int,
    output_dir:str
) -> None:

    # read texts and labels
    with open(texts_file, "r") as f:
        texts = f.readlines()
    with open(labels_file, "r") as f:
        labels = f.readlines()
    assert len(texts) == len(labels)

    validation_size = min(validation_size, len(texts) // 10)
    # split into training and validation sets
    idx = np.random.choice(len(texts), size=validation_size)
    idx = idx.tolist()
    
    # build validation split
    val_texts = [texts[i].strip() for i in idx]
    val_labels = [labels[i].strip() for i in idx]
    # build train split
    idx = set(idx)
    train_texts = [texts[i].strip() for i in range(len(texts)) if i not in idx]
    train_labels = [labels[i].strip() for i in range(len(labels)) if i not in idx]

    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    # save both splits to disk
    with open(os.path.join(output_dir, "train_texts.txt"), "w+") as f:
        f.write('\n'.join(train_texts))
    with open(os.path.join(output_dir, "train_labels.txt"), "w+") as f:
        f.write('\n'.join(train_labels))
    with open(os.path.join(output_dir, "val_texts.txt"), "w+") as f:
        f.write('\n'.join(val_texts))
    with open(os.path.join(output_dir, "val_labels.txt"), "w+") as f:
        f.write('\n'.join(val_labels))

if __name__ == '__main__':

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Split dataset into training and validation")
    parser.add_argument("--source-file", type=str, default=None, help="The path to the source .csv data file.")
    parser.add_argument("--texts-file", type=str, default=None, help="The path to the texts .txt file")
    parser.add_argument("--labels-file", type=str, default=None, help="The path to the labels .txt file")
    parser.add_argument("--validation-size", type=int, help="Number of validation samples.")
    parser.add_argument("--output-dir", type=str, help="The output directory.")
    # parser arguments
    args = parser.parse_args()

    if args.source_file is not None:
        assert args.texts_file is None
        assert args.labels_file is None

        args = vars(args)
        args.pop("texts_file")
        args.pop("labels_file")
        csv_split_train_val(**args)

    elif (args.texts_file is not None) and (args.labels_file is not None):
        assert args.source_file is None

        args = vars(args)
        args.pop("source_file")
        txt_split_train_val(**args)
