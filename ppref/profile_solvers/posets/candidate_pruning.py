import heapq
from _bisect import bisect_right, bisect_left
from math import isclose
from time import time

from ppref.models.mallows import Mallows
from ppref.preferences.poset import Poset
from ppref.profile_solvers.posets.helper import quickly_compute_upper_lower_bounds, prune_candidates_w_too_low_upper_bound
from ppref.rank_estimators.rim_w_poset import calculate_rank_probs_for_item_given_rim_w_poset_by_sequential


def solver_w_candidate_pruning(posets: list[Poset], rule_vector: list[int]):
    """Algorithm:
    1. Quickly estimate upper and lower bounds of scores of each candidate.
    2. Top-k optimization:
        - LB_heap (higher to lower) - order of processing of candidates
        - UB_list (lower to higher) - order of pruning of candidates
    """
    candidates = posets[0].item_set
    mallows = Mallows(reference=tuple(candidates), phi=1.0)
    max_rank = rule_vector.index(0) - 1

    t1 = time()
    candidate2upper, candidate2lower = quickly_compute_upper_lower_bounds(posets, rule_vector)
    t_quick_bounds = time() - t1

    candidate2upper, candidate2lower = prune_candidates_w_too_low_upper_bound(candidate2upper, candidate2lower)

    lb_heap = []
    for c, lb in candidate2lower.items():
        heapq.heappush(lb_heap, (-1 * lb, c))

    ub_list = sorted(candidate2upper, key=lambda x: candidate2upper[x])

    losing_candidates = set()
    pruned_candidates = set()
    while lb_heap:
        lb, c = heapq.heappop(lb_heap)

        if c in losing_candidates:
            pruned_candidates.add(c)
        else:
            c_exact = 0
            if isclose(-1 * lb, candidate2upper[c]):
                c_exact = -1 * lb
            else:
                for i, poset in enumerate(posets):
                    rank_left, rank_right = poset.get_range_of_possible_ranks(c)
                    score_left, score_right = rule_vector[rank_left], rule_vector[rank_right]

                    if isclose(score_left, score_right):
                        c_exact += score_left
                    else:
                        if c in poset.items_in_poset:
                            answer = calculate_rank_probs_for_item_given_rim_w_poset_by_sequential(item=c, rim=mallows,
                                                                                                   poset=poset,
                                                                                                   max_rank=max_rank)
                            probs = answer['probabilities']
                        else:
                            probs = [1 / mallows.num_items for _ in range(max_rank + 1)]

                        exact_score = sum([probs[i] * rule_vector[i] for i in range(max_rank + 1)])
                        c_exact += exact_score

                ub_list.remove(c)
                candidate2upper[c] = c_exact
                new_position = bisect_right(ub_list, c_exact, key=lambda x: candidate2upper[x])
                ub_list.insert(new_position, c)

            split_rank = bisect_left(ub_list, c_exact, key=lambda x: candidate2upper[x])

            if split_rank > 0:
                losing_candidates.update(set(ub_list[:split_rank]))
                ub_list = ub_list[split_rank:]

    winner_score = candidate2upper[ub_list[-1]]
    winners = {ub_list[-1]}
    for c in reversed(ub_list[:-1]):
        if isclose(candidate2upper[c], winner_score):
            winners.add(c)
        else:
            break

    return {'winners': tuple(winners), 'winner_score': winner_score, 'num_pruned_candidates': len(pruned_candidates),
            't_quick_bounds': t_quick_bounds, 't_total_sec': time() - t1}
