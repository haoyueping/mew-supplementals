from copy import deepcopy
from typing import Dict, Any, Sequence, Iterator

import networkx as nx
from numpy.random import Generator as NumpyRNG
from numpy.random import default_rng

from ppref.preferences.linear import Linear


class Poset(object):

    def __init__(self, parent_to_children: Dict[Any, set], item_set: set):
        self.parent_to_children: Dict[Any, set] = deepcopy(parent_to_children)
        self.item_set: set = deepcopy(item_set)
        self.item_set_tuple = tuple(sorted(item_set))

        self.dag: nx.DiGraph = nx.DiGraph()
        for parent, children in parent_to_children.items():
            for child in children:
                self.dag.add_edge(parent, child)

        self.items_in_poset = set(self.dag.nodes)
        self.dag_tc: nx.DiGraph = nx.DiGraph()

    def __repr__(self):
        class_name = self.__class__.__name__
        return f'{class_name}(parent_to_children={self.parent_to_children}, item_set={self.item_set})'

    def __eq__(self, other):
        return isinstance(other, Poset) and (other.item_set == self.item_set) and \
               (other.parent_to_children == self.parent_to_children)

    def __hash__(self):
        return hash((self.to_string_of_sorted_tree(), self.item_set_tuple))

    def to_string_of_sorted_tree(self):
        sorted_tree = [[parent, sorted(self.parent_to_children[parent])] for parent in sorted(self.parent_to_children)]
        return str(sorted_tree)

    def get_full_item_set(self):
        return self.item_set

    def has_item(self, item):
        return self.dag.has_node(item)

    def has_such_linear_extension(self, permutation: Sequence):
        item_to_rank = {item: rank for rank, item in enumerate(permutation)}
        return self.has_such_linear_extension_of_item_to_rank(item_to_rank)

    def has_such_linear_extension_of_item_to_rank(self, item_to_rank):
        """Is the input ranking a linear extension of the item preference.

        :param item_to_rank: represent a ranking by a mapping of item to its rank
        :return:
        """
        for parent, children in self.parent_to_children.items():
            parent_rank = item_to_rank[parent]
            for child in children:
                if item_to_rank[child] < parent_rank:
                    return False

        return True

    def calculate_tc(self):
        self.dag_tc: nx.DiGraph = nx.transitive_closure_dag(self.dag)

    def get_all_ancestors(self, item):
        if nx.is_empty(self.dag_tc):
            self.calculate_tc()

        return self.dag_tc.predecessors(item)

    def get_all_descendants(self, item):
        if nx.is_empty(self.dag_tc):
            self.calculate_tc()

        return self.dag_tc.successors(item)

    def get_number_of_ancestors(self, item):
        return len(list(self.get_all_ancestors(item)))

    def get_number_of_descendants(self, item):
        return len(list(self.get_all_descendants(item)))

    def get_range_of_possible_ranks(self, item):
        # between 0 and m-1
        if item in self.items_in_poset:
            left = self.get_number_of_ancestors(item)
            right = len(self.item_set) - 1 - self.get_number_of_descendants(item)
            return left, right
        else:
            return 0, len(self.item_set) - 1

    def to_linears(self) -> set[Linear]:
        return set(self.get_generator_of_linears())

    def get_generator_of_linears(self) -> Iterator[Linear]:
        for linear in nx.all_topological_sorts(self.dag):
            yield Linear(ranking=tuple(linear))

    @classmethod
    def generate_a_random_instance(cls, m=20, cardinality=5, edge_prob=0.3, rng: NumpyRNG = None):
        rng = rng or default_rng()

        items = rng.choice(m, cardinality, replace=False)
        pref = {}
        for i in range(cardinality - 1):
            for j in range(i + 1, cardinality):
                if rng.random() < edge_prob:
                    pref.setdefault(items[i], set()).add(items[j])

        if pref:
            return Poset(pref, set(range(m)))
        else:
            return cls.generate_a_random_instance(m, cardinality, edge_prob)
