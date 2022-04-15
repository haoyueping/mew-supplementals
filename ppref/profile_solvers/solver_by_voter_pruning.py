from math import isclose
from multiprocessing import cpu_count, Pool
from time import time

from ppref.models.mallows import Mallows
from ppref.models.rim import RepeatedInsertionModel
from ppref.models.rsm import RepeatedSelectionModelRV
from ppref.preferences.combined import MallowsWithPP, RimWithTR, MallowsWithPoset
from ppref.preferences.poset import Poset
from ppref.preferences.special import PartitionedWithMissing, PartitionedPreferences, PartialChain, TruncatedRanking
from ppref.profile_solvers.posets.helper import prune_candidates_w_too_low_upper_bound
from ppref.rank_estimators.mallows_w_pp import calculate_rank_probs_for_item_given_mallows_w_pp
from ppref.rank_estimators.rim import calculate_rank_probs_for_item_given_rim
from ppref.rank_estimators.rim_w_poset import calculate_rank_probs_for_item_given_rim_w_poset_by_sequential
from ppref.rank_estimators.rim_w_trun import calculate_rank_probs_for_item_given_rim_w_trun
from ppref.rank_estimators.rsm import calculate_rank_prob_for_item_given_rsm


def quickly_compute_upper_lower_bounds(profile: list, rule_vector: tuple[int]):
    candidates = profile[0].get_full_item_set()
    candidate2upper = {c: 0.0 for c in candidates}
    candidate2lower = {c: 0.0 for c in candidates}

    # quickly calculate upper and lower bounds

    for pref in profile:
        for c in candidates:
            rank_left, rank_right = pref.get_range_of_possible_ranks(c)
            candidate2upper[c] += rule_vector[rank_left]
            candidate2lower[c] += rule_vector[rank_right]

    return candidate2upper, candidate2lower


def general_abstract_solver(pref, item, max_rank=None):
    if isinstance(pref, Poset):
        if pref.has_item(item):
            mallows = Mallows(reference=pref.item_set_tuple, phi=1.0)
            answer = calculate_rank_probs_for_item_given_rim_w_poset_by_sequential(item=item, rim=mallows, poset=pref,
                                                                                   max_rank=max_rank)
            return answer['probabilities']
        else:
            return [1 / len(pref.item_set) for _ in range(max_rank + 1)]
    elif isinstance(pref, PartitionedWithMissing):
        return pref.calculate_rank2prob_for_item(item)
    elif isinstance(pref, PartitionedPreferences):
        return pref.calculate_rank_probabilities_of_item(item)
    elif isinstance(pref, PartialChain):
        return pref.calculate_rank2prob_for_item(item)
    elif isinstance(pref, TruncatedRanking):
        return pref.calculate_rank_probs_for_item(item)
    elif isinstance(pref, RepeatedInsertionModel):
        return calculate_rank_probs_for_item_given_rim(item, pref, max_rank)
    elif isinstance(pref, RepeatedSelectionModelRV):
        return calculate_rank_prob_for_item_given_rsm(item, pref, max_rank)
    elif isinstance(pref, MallowsWithPP):
        return calculate_rank_probs_for_item_given_mallows_w_pp(item, pref, max_rank)
    elif isinstance(pref, RimWithTR):
        return calculate_rank_probs_for_item_given_rim_w_trun(item, pref, max_rank)
    elif isinstance(pref, MallowsWithPoset):
        poset = pref.poset
        if poset.has_item(item):
            answer = calculate_rank_probs_for_item_given_rim_w_poset_by_sequential(item=item, rim=pref.mallows, poset=poset,
                                                                                   max_rank=max_rank)
            return answer['probabilities']
        else:
            return [1 / len(poset.item_set) for _ in range(max_rank + 1)]


def sequential_solver_of_voter_pruning(profile: list, rule_vector: tuple[int]):
    max_rank = rule_vector.index(0) - 1

    t_pruning_sec = 0
    t_solver_sec = 0

    t1 = time()
    candidate2upper, candidate2lower = quickly_compute_upper_lower_bounds(profile, rule_vector)
    t2 = time()
    candidate2upper, candidate2lower = prune_candidates_w_too_low_upper_bound(candidate2upper, candidate2lower)
    t3 = time()

    t_pruning_sec += t3 - t2

    if len(candidate2upper) == 1:
        winner, upper = candidate2upper.popitem()
        lower = candidate2lower[winner]
        return {'winners': (winner,), 'score_upper': upper, 'score_lower': lower, 'num_pruned_voters': len(profile),
                't_quick_bounds_sec': t2 - t1, 't_pruning_sec': t_pruning_sec, 't_solver_sec': 0, 't_total_sec': t3 - t1}

    while profile:
        pref = profile.pop()
        is_refined = False

        for c in candidate2upper:
            rank_left, rank_right = pref.get_range_of_possible_ranks(c)
            score_left, score_right = rule_vector[rank_left], rule_vector[rank_right]
            if not isclose(score_left, score_right):
                t4 = time()
                probs = general_abstract_solver(pref, c, max_rank)
                t5 = time()
                t_solver_sec += t5 - t4

                exact_score = sum([prob * val for prob, val in zip(probs[:max_rank + 1], rule_vector[: max_rank + 1])])
                candidate2upper[c] += exact_score - score_left
                candidate2lower[c] += exact_score - score_right
                is_refined = True

        if is_refined:
            t6 = time()
            candidate2upper, candidate2lower = prune_candidates_w_too_low_upper_bound(candidate2upper, candidate2lower)
            t7 = time()
            t_pruning_sec += t7 - t6

            if len(candidate2upper) == 1:
                t8 = time()
                winner, upper = candidate2upper.popitem()
                lower = candidate2lower[winner]
                return {'winners': (winner,), 'score_upper': upper, 'score_lower': lower, 'num_pruned_voters': len(profile),
                        't_quick_bounds_sec': t2 - t1, 't_pruning_sec': t_pruning_sec, 't_solver_sec': t_solver_sec,
                        't_total_sec': t8 - t1}

    t9 = time()
    winner_score = max(candidate2upper.values())
    winners = tuple(candidate2upper.keys())
    return {'winners': winners, 'score_upper': winner_score, 'score_lower': winner_score, 'num_pruned_voters': 0,
            't_quick_bounds_sec': t2 - t1, 't_pruning_sec': t_pruning_sec, 't_solver_sec': t_solver_sec,
            't_total_sec': t9 - t1}


def parallel_solver_of_voter_pruning(profile: list, rule_vector: tuple[int], threads=None):
    threads = threads or cpu_count()
    max_rank = rule_vector.index(0) - 1

    t1 = time()
    candidate2upper, candidate2lower = quickly_compute_upper_lower_bounds(profile, rule_vector)
    t_quick_bounds = time() - t1

    candidate2upper, candidate2lower = prune_candidates_w_too_low_upper_bound(candidate2upper, candidate2lower)

    if len(candidate2upper) == 1:
        t3 = time()
        winner, upper = candidate2upper.popitem()
        lower = candidate2lower[winner]
        return {'winners': (winner,), 'score_upper': upper, 'score_lower': lower,
                'num_pruned_voters': len(profile), 't_quick_bounds_sec': t_quick_bounds, 't_total_sec': t3 - t1}

    tasks = []
    while profile:
        print(f'#voters = {len(profile)}')
        tasks.clear()
        while profile and len(tasks) < threads * 10:
            pref = profile.pop()
            for c in candidate2upper:
                if candidate2upper[c] != candidate2lower[c]:
                    rank_left, rank_right = pref.get_range_of_possible_ranks(c)
                    score_left, score_right = rule_vector[rank_left], rule_vector[rank_right]
                    if not isclose(score_left, score_right):
                        candidate2upper[c] -= score_left
                        candidate2lower[c] -= score_right

                        if pref.has_item(c):
                            tasks.append((pref, c, max_rank))
                        else:
                            probs = [1 / len(pref.item_set) for _ in range(max_rank + 1)]
                            exact_score = sum([prob * val for prob, val in zip(
                                probs[:max_rank + 1], rule_vector[: max_rank + 1])])
                            candidate2upper[c] += exact_score
                            candidate2lower[c] += exact_score

        if tasks:
            with Pool(threads) as pool:
                all_probs = pool.starmap(general_abstract_solver, tasks)

            for (_, c, _), probs in zip(tasks, all_probs):
                exact_score = sum([prob * val for prob, val in zip(probs[:max_rank + 1], rule_vector[: max_rank + 1])])
                candidate2upper[c] += exact_score
                candidate2lower[c] += exact_score

            candidate2upper, candidate2lower = prune_candidates_w_too_low_upper_bound(candidate2upper, candidate2lower)

            if len(candidate2upper) == 1:
                t4 = time()
                winner, upper = candidate2upper.popitem()
                lower = candidate2lower[winner]
                return {'winners': (winner,), 'score_upper': upper, 'score_lower': lower,
                        'num_pruned_voters': len(profile), 't_quick_bounds': t_quick_bounds, 't_total_sec': t4 - t1}

    t5 = time()
    winner_score = max(candidate2upper.values())
    return {'winners': tuple(candidate2upper.keys()), 'score_upper': winner_score, 'score_lower': winner_score,
            'num_pruned_voters': 0, 't_quick_bounds': t_quick_bounds, 't_total_sec': t5 - t1}


def sequential_baseline_solver(profile: list, rule_vector: tuple[int]):
    t1 = time()

    max_rank = rule_vector.index(0) - 1
    candidates = profile[0].get_full_item_set()
    candidate2score = {c: 0.0 for c in candidates}

    for pref in profile:
        for c in candidates:
                probs = general_abstract_solver(pref, c, max_rank)
                exact_score = sum([prob * val for prob, val in zip(probs[:max_rank + 1], rule_vector[: max_rank + 1])])
                candidate2score[c] += exact_score

    winner_score = max(candidate2score.values())
    winners = tuple([c for c, s in candidate2score.items() if s == winner_score])
    t2 = time()
    return {'winners': winners, 'winner_score': winner_score, 't_sec': t2 - t1}


def main():
    from experiments.synthetic.posets.experiment import generate_profile_df

    k_approval, num_candidates, num_voters, phi, rsm_pmax, batch = 1, 12, 10_000, 0.1, 0.2, 0

    df = generate_profile_df(num_candidates, num_voters, phi, rsm_pmax, batch)
    posets: list[Poset] = df['poset'].tolist()
    rule_vector = tuple([1 for _ in range(k_approval)] + [0 for _ in range(len(posets[0].item_set) - k_approval)])

    print(sequential_solver_of_voter_pruning(posets.copy(), rule_vector=rule_vector))


if __name__ == '__main__':
    main()
