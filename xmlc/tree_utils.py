from treelib import Tree, Node
from itertools import chain
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Set, Callable, Any

def yield_tree_levels(
    tree:Tree,
    max_depth:int =None,
    key:Callable[[Node], Any] =None
) -> Iterator[Tuple[Node]]:
    """ Yield the levels of the tree as tuple of nodes """
    # yield all levels if no maximum depth is provided
    max_depth = (tree.depth() + 1) if max_depth is None else max_depth
    # start with root node
    cur_level = (tree[tree.root],)
    for _ in range(max_depth):
        # yield current level
        yield cur_level
        # gather all chidlren of all nodes in the current level
        # and sort them if a sorting key is provided
        cur_level = tuple(chain(*(tree.children(n.identifier) for n in cur_level)))
        cur_level = sorted(cur_level, key=key) if key is not None else cur_level


@dataclass
class NodeIndex(object):
    """ Stores indices of a node.
        This defines a fixed ordering over nodes w.r.t different scopes.

        * level_index: enumerates nodes in the same level (scope: level)
        * global_index: enumerates all nodes (scope: global)
    """
    level_index:int =-1
    global_index:int =-1


def index_tree(tree:Tree) -> Tree:
    """ parse tree and index each node """
    # iterate over all levels and index nodes
    global_index = 0
    for level in yield_tree_levels(tree):
        # process each node in the current level
        for j, node in enumerate(level):
            # index node
            node.data = NodeIndex(
                level_index=j,
                global_index=global_index
            )
            # update global index
            global_index += 1
    # return the indexed tree
    return tree


def propagate_labels_to_level(
    tree:Tree,
    labels:List[Set[int]],
    level:int
) -> List[Set[int]]:
    """ propagate the list of label-sets up to the given level
        assuming that the labels are currently at lowest level
    """
    for _ in range(tree.depth() - 1 - level):
        labels = [
            set((tree.parent(i).identifier for i in ls))
            for ls in labels
        ]
    # return the propagated labels
    return labels

def convert_labels_to_ids(
    tree:Tree,
    labels:List[Set[str]]
) -> List[Set[int]]:
    """ Convert labels to label-ids given by the level-indices of the corresponding nodes """
    return [
        set((tree[i].data.level_index for i in ls))
        for ls in labels
    ]    
