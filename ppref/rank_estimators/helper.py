from dataclasses import dataclass
from typing import Dict, Any


def calculate_amp_insertion_range_iter(r: list, inserted_ancestors: set, inserted_descendants: set):
    item_to_rank_in_r = {e: rank for rank, e in enumerate(r)}

    l_indices = [item_to_rank_in_r[ancestor] for ancestor in inserted_ancestors]
    h_indices = [item_to_rank_in_r[descendant] for descendant in inserted_descendants]

    low = max(l_indices, default=-1)
    high = min(h_indices, default=len(r))

    # low + 1 because low is the position after which current item must be inserted.
    # high + 1 because range(x, y) is [x, y-1], which only reaches y-1.
    return range(low + 1, high + 1)


def calculate_amp_insertion_range(r: list, inserted_ancestors: set, inserted_descendants: set):
    return list(calculate_amp_insertion_range_iter(r, inserted_ancestors, inserted_descendants))


@dataclass(frozen=True, slots=True)
class Item2Pos:
    val: Dict[Any, int]

    def __hash__(self):
        return hash(frozenset(self.val.items()))

    def get_pos(self, item) -> int:
        return self.val[item]

    def is_tracking(self, item):
        return item in self.val

    def size(self):
        return len(self.val)
