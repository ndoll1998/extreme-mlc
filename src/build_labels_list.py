import os

if __name__ == '__main__':

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Split dataset into training and validation")
    parser.add_argument("--train-labels", type=str, help="The path to the train labels .txt file")
    parser.add_argument("--val-labels", type=str, help="The path to the validation labels .txt file")
    parser.add_argument("--test-labels", type=str, help="The path to the test labels .txt file")
    parser.add_argument("--output-dir", type=str, help="The output directory.")
    # parser arguments
    args = parser.parse_args()

    # read all labels
    with open(args.train_labels, "r") as f:
        train_labels = f.read().split()
    with open(args.val_labels, "r") as f:
        val_labels = f.read().split()
    with open(args.test_labels, "r") as f:
        test_labels = f.read().split()

    # build unique labels
    labels = set((*train_labels, *val_labels, *test_labels))

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # save labels to disk
    with open(os.path.join(args.output_dir, "labels.txt"), "w+") as f:
        f.write('\n'.join(labels))
