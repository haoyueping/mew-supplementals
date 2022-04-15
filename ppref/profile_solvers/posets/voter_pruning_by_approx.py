from math import isclose
from time import time

from ppref.helper import normalize_weights
from ppref.models.mallows import Mallows
from ppref.preferences.poset import Poset
from ppref.profile_solvers.posets.helper import quickly_compute_upper_lower_bounds, prune_candidates_w_too_low_upper_bound
from ppref.rank_estimators.approximate.mallows_w_poset.misamp import misamp_ready_w_fixed_sample_size


def solver_w_voter_pruning(posets: list[Poset], rule_vector: list[int]):
    """Algorithm:
    1. Quickly estimate upper and lower bounds of scores of each candidate.
    2. For each voter, refine the candidates' upper and lower bounds, and prune candidates with too low upper bounds.
    """
    candidates = posets[0].item_set
    mallows = Mallows(reference=tuple(candidates), phi=1.0)
    max_rank = rule_vector.index(0) - 1

    t1 = time()
    candidate2upper, candidate2lower = quickly_compute_upper_lower_bounds(posets, rule_vector)
    t_quick_bounds = time() - t1

    candidate2upper, candidate2lower = prune_candidates_w_too_low_upper_bound(candidate2upper, candidate2lower)

    if len(candidate2upper) == 1:
        t3 = time()
        winner, upper = candidate2upper.popitem()
        lower = candidate2lower[winner]
        return {'winners': (winner,), 'score_upper': upper, 'score_lower': lower,
                'num_pruned_voters': len(posets), 't_quick_bounds_sec': t_quick_bounds, 't_total_sec': t3 - t1}

    for i, poset in enumerate(posets):
        t_ms, item2counts = misamp_ready_w_fixed_sample_size(mallows, poset, [], single_core_workload=10)[-3:-1]

        for c in candidate2upper:
            if candidate2upper[c] != candidate2lower[c]:
                rank_left, rank_right = poset.get_range_of_possible_ranks(c)
                score_left, score_right = rule_vector[rank_left], rule_vector[rank_right]
                if not isclose(score_left, score_right):
                    if c in poset.items_in_poset:
                        probs = normalize_weights(item2counts[c])
                    else:
                        probs = [1 / mallows.num_items for _ in range(max_rank + 1)]

                    exact_score = sum([prob * val for prob, val in zip(probs[:max_rank + 1], rule_vector[: max_rank + 1])])

                    candidate2upper[c] += exact_score - score_left
                    candidate2lower[c] += exact_score - score_right

        candidate2upper, candidate2lower = prune_candidates_w_too_low_upper_bound(candidate2upper, candidate2lower)

        if len(candidate2upper) == 1:
            t4 = time()
            winner, upper = candidate2upper.popitem()
            lower = candidate2lower[winner]
            return {'winners': (winner,), 'score_upper': upper, 'score_lower': lower,
                    'num_pruned_voters': len(posets) - i - 1, 't_quick_bounds': t_quick_bounds, 't_total_sec': t4 - t1}

    t5 = time()
    winner_score = max(candidate2upper.values())
    return {'winners': tuple(candidate2upper.keys()), 'score_upper': winner_score, 'score_lower': winner_score,
            'num_pruned_voters': 0, 't_quick_bounds': t_quick_bounds, 't_total_sec': t5 - t1}


def main():
    import pandas as pd

    from experiments.helper import get_path_to_poset_profile
    from ppref.profile_solvers.posets.voter_pruning import solver_w_voter_pruning

    k_approval, num_candidates, num_voters, phi, rsm_pmax, batch = 1, 10, 1000, 0.1, 0.2, 0
    # generate_partial_voting_profile(num_candidates, num_voters, phi, rsm_pmax, batch)
    filename = get_path_to_poset_profile(num_candidates, num_voters, phi, rsm_pmax, batch)
    df = pd.read_csv(filename, delimiter='\t', comment='#')

    posets: list[Poset] = [eval(po) for po in df['poset']]
    rule_vector = [1 for _ in range(k_approval)] + [0 for _ in range(len(posets[0].item_set) - k_approval)]

    print(solver_w_voter_pruning(posets, rule_vector=rule_vector))


if __name__ == '__main__':
    main()
