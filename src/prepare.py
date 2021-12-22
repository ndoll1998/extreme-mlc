import os
import json
import pandas as pd
from itertools import chain
from collections import Counter
from tqdm.auto import tqdm

def tqdm_read_csv_to_pandas(csv_fpath, **kwargs):
    # open file and wrap it with a progress bar for reading
    with open(csv_fpath, "r") as f:
        with tqdm.wrapattr(f, "read", total=os.path.getsize(csv_fpath), desc="Read CSV") as f_tqdm:
            # load content of the file into a pandas dataframe
            return pd.read_csv(f_tqdm, **kwargs)

def filter_labels(
    all_labels:list, 
    min_freq:int, 
    max_num:int,
    filtered_labels:set =None,
    force:bool =False
) -> tuple:
    # create flat list of all labels and count their occurances
    all_labels = [labels.split() for labels in all_labels]
    
    flat_labels = tuple(chain(*all_labels))
    if filtered_labels is not None:
        # make sure we only consider labels that are present in the data
        unique_labels = set(flat_labels)
        filtered_labels = set(filtered_labels).intersection(unique_labels)
        # filter out all labels that are not in the filtered labels
        flat_labels = filter(lambda x: x in filtered_labels, flat_labels)
    flat_labels = list(flat_labels)

    counter = Counter(tqdm(
        flat_labels,            # flat label list
        total=len(flat_labels), # total number of labels
        desc="Count Labels"
    ))
    
    if (filtered_labels is None) or force:
        # get the set of selected labels
        filtered_labels = counter.most_common(max_num) if max_num > 0 else counter.items()
        filtered_labels = set([l for l, c in filtered_labels if c > min_freq])

    # filter labels
    all_labels = [[l for l in labels if l in filtered_labels] for labels in tqdm(all_labels, desc="Filter Labels")]
    
    # some basic statistics
    counts = [counter[l] for l in filtered_labels]
    metrics = {
        'num_unique_labels': len(filtered_labels),
        'avg_label_occurance': sum(counts) / len(counts),
        'avg_labels_per_example': sum(map(len, all_labels)) / len(all_labels)
    }
    
    # concat labels into a single string per example and return all
    all_labels = [' '.join(labels) for labels in all_labels]
    return all_labels, filtered_labels, metrics

def filter_hospitals(
    df:pd.DataFrame,
    valid_hospitals:list =None
) -> pd.DataFrame:
    # no filter arguments given
    if valid_hospitals is None:
        return df
    # select data entries from hospitals
    mask = ~df['hospital_name'].isin(valid_hospitals)
    df.drop(df[mask].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def convert_csv_to_format(
    # input
    csv_fpath: str,
    code_type: str,
    # output
    out_texts_fpath: str,
    out_labels_fpath: str,
    out_info_fpath: str,
    # filter hospitals
    valid_hospitals:list,
    # filter labels
    min_label_freq: int,
    max_num_labels: int,
    # list of labels to use
    filtered_labels: set =None,
    force_filtering: bool =False,
) -> set:

    # load pandas dataframe
    data = tqdm_read_csv_to_pandas(csv_fpath)
    # filter hospitals
    data = filter_hospitals(data, valid_hospitals)
    # build info dataframe and save to disk
    info = data[['hospital_id', 'case_id']]
    info.to_csv(out_info_fpath) 
    # get all documents and corresponding labels
    all_texts = data['documents_merged'].tolist()
    all_labels = data[code_type].tolist()
    # cleanup texts and filter labels
    all_texts = [t.replace('\n', ' ') for t in all_texts]  
    all_labels, selected_labels, metrics = filter_labels(
        all_labels, min_label_freq, max_num_labels, 
        filtered_labels=filtered_labels, force=force_filtering
    ) 
    # remove all entries that have no more labels after filtering 
    mask = [len(labels) > 0 for labels in all_labels]
    all_labels = [l for l, v in zip(all_labels, mask) if v]
    all_texts = [t for t, v in zip(all_texts, mask) if v]
    # print total number of examples
    metrics["num_examples"] = len(all_texts)

    # write to files
    with open(out_texts_fpath, "w+") as f:
        f.write('\n'.join(all_texts))
    with open(out_labels_fpath, "w+") as f:
        f.write('\n'.join(all_labels))

    return selected_labels, metrics

def create_data(

    train_source:str,
    val_source:str,
    test_source:str,
    code_type:str,    

    output_dir:str,
    
    hospitals:list =None,
    min_label_freq:int =-1,
    max_num_labels:int =-1,
    preselected_labels_file:str =None
) -> None:

    # create output directory if necessary
    os.makedirs(output_dir, exist_ok=True)

    # read allowed labels if provided
    labels = None
    if preselected_labels_file is not None:
        with open(preselected_labels_file, "r") as f:
            labels = set(f.read().strip().splitlines())
        print("Using %i preselected labels from %s" % (len(labels), preselected_labels_file))

    print("Preparing Train Data...")
    # train data
    labels, train_metrics = convert_csv_to_format(
        csv_fpath=train_source,
        code_type=code_type,
        out_texts_fpath=os.path.join(output_dir, "train_texts.txt"),
        out_labels_fpath=os.path.join(output_dir, "train_labels.txt"),
        out_info_fpath=os.path.join(output_dir, "train_info.csv"),
        min_label_freq=min_label_freq,
        max_num_labels=max_num_labels,
        valid_hospitals=hospitals,
        filtered_labels=labels,
        force_filtering=True
    )
    print("Done")

    # test data
    print("\nPreparing Validation Data...")
    _, val_metrics = convert_csv_to_format(
        csv_fpath=val_source,
        code_type=code_type,
        out_texts_fpath=os.path.join(output_dir, "val_texts.txt"),
        out_labels_fpath=os.path.join(output_dir, "val_labels.txt"),
        out_info_fpath=os.path.join(output_dir, "val_info.csv"),
        valid_hospitals=hospitals,
        # use the same labels as for the training data
        min_label_freq=-1,
        max_num_labels=-1,
        filtered_labels=labels
    )
    print("Done")

    # test data
    print("\nPreparing Test Data...")
    _, test_metrics = convert_csv_to_format(
        csv_fpath=test_source,
        code_type=code_type,
        out_texts_fpath=os.path.join(output_dir, "test_texts.txt"),
        out_labels_fpath=os.path.join(output_dir, "test_labels.txt"),
        out_info_fpath=os.path.join(output_dir, "test_info.csv"),
        valid_hospitals=hospitals,
        # use the same labels as for the training data
        min_label_freq=-1,
        max_num_labels=-1,
        filtered_labels=labels
    )
    print("Done")

    # save labels to directory
    with open(os.path.join(output_dir, "labels.txt"), "w+") as f:
        f.write('\n'.join(labels))
    # save list of used hospitals to directory
    with open(os.path.join(output_dir, "hospitals.txt"), "w+") as f:
        f.write('\n'.join(hospitals))

    # write train and test metrics
    with open(os.path.join(output_dir, "metrics.json"), "w+") as f:
        f.write(json.dumps({'train': train_metrics, 'validation': val_metrics, 'test': test_metrics}))

if __name__ == '__main__':

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Prepare the raw train and test data.")
    parser.add_argument('--code-type', type=str, choices=["ops_codes", "icd_codes"])
    parser.add_argument('--train-source', type=str, help="The path to the .csv file containing the raw training data.")
    parser.add_argument('--val-source', type=str, help="The path to the .csv file containing the raw validation data.")
    parser.add_argument('--test-source', type=str, help="The path to the .csv file containing the raw testing data.")
    parser.add_argument('--min-label-freq', type=int, default=-1, help="All labels with less occurances will be deleted.")
    parser.add_argument('--max-num-labels', type=int, default=-1, help="Restrict the set of labels to at most n.")
    parser.add_argument('--output-dir', type=str, help="The directory where the prepared data and metrics will be stored.")
    # parse arguments
    args = parser.parse_args()
    args = vars(args)

    create_data(
        **args,
        hospitals = [
            "AlexianerLudgerus",
            "Bochum",
            "Brilon",
            "Bielefeld",
            "Castrop",
            "Hamm",
            "Herne",
            "Lippstadt",
            "OttoFricke",
            "Ruedesheim",
            "Wiesbaden",
            "Witten"
        ]
    )
