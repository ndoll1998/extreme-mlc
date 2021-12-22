import os
import numpy as np
import pandas as pd

def split_train_val(
    source_file:str,
    validation_size:int,
    output_dir:str
) -> None:
   
    # read input csv file
    df = pd.read_csv(source_file, index_col=0)
    # randomly select samples
    val_df = df.sample(n=validation_size)
    train_df = df.drop(val_df.index)

    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    # save splits in output directory
    train_df.to_csv(os.path.join(output_dir, "train-split.csv"))
    val_df.to_csv(os.path.join(output_dir, "val-split.csv"))

if __name__ == '__main__':

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Split dataset into training and validation")
    parser.add_argument("--source-file", type=str, help="The path to the source .csv data file.")
    parser.add_argument("--validation-size", type=int, help="Number of validation samples.")
    parser.add_argument("--output-dir", type=str, help="The output directory.")
    # parser arguments
    args = parser.parse_args()
    args = vars(args)
    # split dataframe
    split_train_val(**args)
