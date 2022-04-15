from bisect import bisect_right
from random import choices
from typing import Dict, Any, Tuple

from ppref.models.mallows import Mallows
from ppref.preferences.poset import Poset
from ppref.rank_estimators.helper import calculate_amp_insertion_range


def amp_sampler(mallows: Mallows, poset: Poset) -> Tuple[list, float, int]:
    r, inserted_items = [], set()
    distance, prob = 0, 1.0
    for rank_i, item in enumerate(mallows.reference):

        if poset.has_item(item):
            inserted_ancestors = inserted_items.intersection(poset.get_all_ancestors(item))
            inserted_descendants = inserted_items.intersection(poset.get_all_descendants(item))

            insertion_range = calculate_amp_insertion_range(r, inserted_ancestors, inserted_descendants)
            inserted_items.add(item)
        else:
            insertion_range = list(range(len(r) + 1))

        weights = [mallows.get_prob_i_j(rank_i, j) for j in insertion_range]

        pos_idx = choices(list(range(len(weights))), weights=weights, k=1)[0]  # choices() returns a list
        pos = insertion_range[pos_idx]

        distance += len(r) - pos
        prob *= weights[pos_idx] / sum(weights)

        r.insert(pos, item)

    return r, prob, distance


def prob_of_amp_drawing_ranking(ranking_by_item2rank: Dict[Any, int], poset: Poset, mallows: Mallows) -> float:
    r, ranks, inserted, prob = [], [], set(), 1
    for i, item in enumerate(mallows.reference):
        final_rank = ranking_by_item2rank[item]
        pos_j = bisect_right(ranks, final_rank)

        if poset.has_item(item):
            inserted_ancestors = inserted.intersection(poset.get_all_ancestors(item))
            inserted_descendants = inserted.intersection(poset.get_all_descendants(item))

            insertion_range = calculate_amp_insertion_range(r, inserted_ancestors, inserted_descendants)
            inserted.add(item)
        else:
            insertion_range = list(range(len(r) + 1))

        if pos_j in insertion_range:
            idx = insertion_range.index(pos_j)

            probs = [mallows.get_prob_i_j(i, j) for j in insertion_range]
            prob *= probs[idx] / sum(probs)

            r.insert(pos_j, item)
            ranks.insert(pos_j, final_rank)
        else:
            return 0

    return prob
