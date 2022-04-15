from time import time

from ppref.models.mallows import Mallows
from ppref.preferences.poset import Poset
from ppref.profile_solvers.posets.helper import prune_candidates_w_too_low_upper_bound


def solver_w_smart_pruning(posets: list[Poset], rule_vector: list[int]):
    # TODO undone
    candidates = posets[0].item_set
    mallows = Mallows(reference=tuple(candidates), phi=1.0)
    max_rank = rule_vector.index(0) - 1

    t1 = time()

    # quickly calculate upper and lower bounds
    candidate2upper = {c: 0.0 for c in candidates}
    candidate2lower = {c: 0.0 for c in candidates}
    candidate2estimation = {c: 0.0 for c in candidates}
    # cand_pair_to_ordered_voters = {(c1, c2) for c1 in }

    for poset in posets:
        for c in candidates:
            rank_left, rank_right = poset.get_range_of_possible_ranks(c)
            candidate2upper[c] += rule_vector[rank_left]
            candidate2lower[c] += rule_vector[rank_right]
            candidate2estimation[c] += sum(rule_vector[rank_left:rank_right + 1]) / (rank_right - rank_left + 1)

    t_quick_bounds = time() - t1

    candidate2upper, candidate2lower = prune_candidates_w_too_low_upper_bound(candidate2upper, candidate2lower)
