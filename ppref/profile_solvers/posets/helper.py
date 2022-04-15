from math import isclose

from ppref.preferences.poset import Poset


def prune_candidates_w_too_low_upper_bound(candidate2upper, candidate2lower):
    c_lower_max = max(candidate2lower, key=lambda x: candidate2lower[x])
    lower_max = candidate2lower[c_lower_max]

    losing_candidates = set()
    for c, ub in candidate2upper.copy().items():
        if c != c_lower_max and (not isclose(ub, lower_max)) and ub < lower_max:
            losing_candidates.add(c)

    for c in losing_candidates:
        candidate2upper.pop(c)
        candidate2lower.pop(c)

    return candidate2upper, candidate2lower


def quickly_compute_upper_lower_bounds(posets: list[Poset], rule_vector: tuple[int]):
    candidates = posets[0].item_set
    candidate2upper = {c: 0.0 for c in candidates}
    candidate2lower = {c: 0.0 for c in candidates}

    # quickly calculate upper and lower bounds

    for poset in posets:
        for c in candidates:
            rank_left, rank_right = poset.get_range_of_possible_ranks(c)
            candidate2upper[c] += rule_vector[rank_left]
            candidate2lower[c] += rule_vector[rank_right]

    return candidate2upper, candidate2lower
