import os
import pickle
from xmlc.tree_utils import index_tree
from treelib import Tree

def build_label_tree(labels, n=0):
    # create tree and add root node
    tree = Tree()
    root = tree.create_node("Root", "Root")
    # use a flat label tree (i.e. no label grouping at all)
    if n == 0:
        # add each label as a direct child of the root node
        for label in labels:
            tree.create_node(label, label, parent=root)
    else:
        # group the labels using their first n characters
        label_groups = {}
        for label in labels:
            # get the label group
            group = label[:n]
            # check if there already exists a node for that group
            if group not in label_groups:
                group_node = tree.create_node(group, group, parent=root)
                label_groups[group] = group_node
            # add the label to the group
            group_node = label_groups[group]
            tree.create_node(label, label, parent=group_node)

    return tree
        

if __name__ == '__main__':

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Build the label tree.")
    parser.add_argument("--labels-file", type=str, help="Path to file containing the labels.")
    parser.add_argument("--group-id-chars", type=int, help="Group labels by their first n characters")
    parser.add_argument("--output-file", type=str, help="File to save the label tree at.")
    # parser arguments
    args = parser.parse_args()

    # group the labels by their first n characters
    # e.g. 3-0355.1 -> 3-0 for n=3
    n = args.group_id_chars

    # load labels
    with open(args.labels_file, "r") as f:
        labels = f.read().splitlines()

    # build the label tree and index it
    tree = build_label_tree(labels, n=n)
    tree = index_tree(tree)

    # create output dir if necessary
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    # save the tree
    with open(args.output_file, "wb+") as f:
        pickle.dump(tree, f)
