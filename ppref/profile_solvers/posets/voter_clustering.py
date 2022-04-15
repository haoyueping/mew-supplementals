import heapq
from bisect import bisect_right, bisect_left
from math import isclose
from time import time

from ppref.models.mallows import Mallows
from ppref.preferences.poset import Poset
from ppref.rank_estimators.rim_w_poset import calculate_rank_probs_for_item_given_rim_w_poset_by_sequential


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

    return candidate2upper, candidate2lower, losing_candidates


def candidate_pruning(posets: list[Poset], rule_vector: list[int], candidate2upper, candidate2lower):
    """Algorithm:
    1. Quickly estimate upper and lower bounds of scores of each candidate.
    2. Top-k optimization:
        - LB_heap (higher to lower) - order of processing of candidates
        - UB_list (lower to higher) - order of pruning of candidates
    """
    candidates = posets[0].item_set
    mallows = Mallows(reference=tuple(candidates), phi=1.0)
    max_rank = rule_vector.index(0) - 1

    lb_heap = []
    for c, lb in candidate2lower.items():
        heapq.heappush(lb_heap, (-1 * lb, c))

    ub_list = sorted(candidate2upper, key=lambda x: candidate2upper[x])

    candidate_to_exact = {}
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
                candidate_to_exact[c] = c_exact
                new_position = bisect_right(ub_list, c_exact, key=lambda x: candidate2upper[x])
                ub_list.insert(new_position, c)

            split_rank = bisect_left(ub_list, c_exact, key=lambda x: candidate2upper[x])

            if split_rank > 0:
                losing_candidates.update(set(ub_list[:split_rank]))
                ub_list = ub_list[split_rank:]

    for c in candidate_to_exact:
        if c in candidate2upper:
            del candidate2upper[c]

        if c in candidate2lower:
            del candidate2lower[c]

    return candidate_to_exact, candidate2upper, candidate2lower


def solver_w_voter_clustering(posets: list[Poset], rule_vector: list[int], d=1):
    """
    @paras
    d: voter clustering by top-d items
    """
    t1 = time()

    candidates = posets[0].item_set.copy()

    poset_to_candidate_to_upper_lower: dict[Poset, dict[int, tuple[int, int]]] = {}
    cluster_to_voters: dict[tuple, list[Poset]] = {}

    candidate_to_uniform_score_tmp = {}
    for poset in posets:
        poset_to_candidate_to_upper_lower[poset] = {}
        candidate_to_uniform_score_tmp.clear()
        for c in poset.item_set:
            # estimate candidate scores by assuming uniform rank distributions between extremal ranks
            rank_left, rank_right = poset.get_range_of_possible_ranks(c)
            poset_to_candidate_to_upper_lower[poset][c] = (rule_vector[rank_left], rule_vector[rank_right])
            candidate_to_uniform_score_tmp[c] = sum(rule_vector[rank_left:rank_right + 1]) / (rank_right + 1 - rank_left)

        ranking = sorted(candidate_to_uniform_score_tmp, key=lambda x: (candidate_to_uniform_score_tmp[x], x), reverse=True)
        cluster = tuple(ranking[:d])

        if cluster in cluster_to_voters:
            cluster_to_voters[cluster].append(poset)
        else:
            cluster_to_voters[cluster] = [poset]

    clusters = set(cluster_to_voters)
    cluster_to_undone_candidates: dict[tuple, set] = {clu: candidates.copy() for clu in clusters}

    cluster_to_candidate_to_exact: dict[tuple, dict[int, float]] = {clu: {} for clu in clusters}
    cluster_to_candidate_to_upper: dict[tuple, dict[int, float]] = {clu: {c: 0.0 for c in candidates} for clu in clusters}
    cluster_to_candidate_to_lower: dict[tuple, dict[int, float]] = {clu: {c: 0.0 for c in candidates} for clu in clusters}

    candidate_to_upper_score: dict[int, float] = {c: 0.0 for c in candidates}
    candidate_to_lower_score: dict[int, float] = {c: 0.0 for c in candidates}

    for cluster, voters in cluster_to_voters.items():
        for voter in voters:
            c2ul = poset_to_candidate_to_upper_lower[voter]
            for c, ul in c2ul.items():
                upper_score, lower_score = ul

                cluster_to_candidate_to_upper[cluster][c] += upper_score
                cluster_to_candidate_to_lower[cluster][c] += lower_score

    for cluster in clusters:
        for c in candidates:
            candidate_to_upper_score[c] += cluster_to_candidate_to_upper[cluster][c]
            candidate_to_lower_score[c] += cluster_to_candidate_to_lower[cluster][c]

    answers = prune_candidates_w_too_low_upper_bound(candidate_to_upper_score, candidate_to_lower_score)
    candidate_to_upper_score, candidate_to_lower_score, losing_candidates = answers

    for cluster in clusters:
        for loser in losing_candidates:
            if loser in cluster_to_undone_candidates[cluster]:
                cluster_to_undone_candidates[cluster].remove(loser)

            if loser in cluster_to_candidate_to_lower[cluster]:
                del cluster_to_candidate_to_lower[cluster][loser]

            if loser in cluster_to_candidate_to_upper[cluster]:
                del cluster_to_candidate_to_upper[cluster][loser]

    while True:
        is_undone = False
        for cluster in clusters:
            if cluster_to_undone_candidates[cluster]:
                is_undone = True

                voters = cluster_to_voters[cluster]
                c2upper_cluster = cluster_to_candidate_to_upper[cluster]
                c2lower_cluster = cluster_to_candidate_to_lower[cluster]

                for c in c2upper_cluster:
                    candidate_to_upper_score[c] -= c2upper_cluster[c]
                    candidate_to_lower_score[c] -= c2lower_cluster[c]

                c2exact_cluster, c2upper_cluster, c2lower_cluster = candidate_pruning(voters, rule_vector, c2upper_cluster,
                                                                                      c2lower_cluster)

                for c, c_exact in c2exact_cluster.items():
                    candidate_to_upper_score[c] += c_exact
                    candidate_to_lower_score[c] += c_exact
                    cluster_to_candidate_to_exact[cluster][c] = c_exact

                for c in c2upper_cluster:
                    candidate_to_upper_score[c] += c2upper_cluster[c]
                    candidate_to_lower_score[c] += c2lower_cluster[c]

                answers = prune_candidates_w_too_low_upper_bound(candidate_to_upper_score, candidate_to_lower_score)
                candidate_to_upper_score, candidate_to_lower_score, losing_candidates = answers

                if len(candidate_to_upper_score) == 1:
                    winner, upper = candidate_to_upper_score.popitem()
                    lower = candidate_to_lower_score[winner]
                    return {'winners': (winner,), 'score_upper': upper, 'score_lower': lower, 't_total_sec': time() - t1}

                cluster_to_candidate_to_exact[cluster] = c2exact_cluster
                cluster_to_candidate_to_upper[cluster] = c2upper_cluster
                cluster_to_candidate_to_lower[cluster] = c2lower_cluster

                for cluster in clusters:
                    for loser in losing_candidates:
                        if loser in cluster_to_undone_candidates[cluster]:
                            cluster_to_undone_candidates[cluster].remove(loser)

                        if loser in cluster_to_candidate_to_upper[cluster]:
                            del cluster_to_candidate_to_upper[cluster][loser]

                        if loser in cluster_to_candidate_to_lower[cluster]:
                            del cluster_to_candidate_to_lower[cluster][loser]

        if not is_undone:
            winner_score = max(candidate_to_upper_score.values())
            return {'winners': tuple(candidate_to_upper_score.keys()), 'score_upper': winner_score,
                    'score_lower': winner_score, 't_total_sec': time() - t1}


def main():
    import pandas as pd

    from experiments.helper import get_path_to_poset_profile
    from ppref.profile_solvers.posets.voter_pruning import solver_w_voter_pruning

    k_approval, num_candidates, num_voters, phi, rsm_pmax, batch = 2, 10, 1000, 1.0, 0.2, 1
    filename = get_path_to_poset_profile(num_candidates, num_voters, phi, rsm_pmax, batch)
    df = pd.read_csv(filename, delimiter='\t', comment='#')

    posets: list[Poset] = [eval(po) for po in df['poset']]
    rule_vector = [1 for _ in range(k_approval)] + [0 for _ in range(len(posets[0].item_set) - k_approval)]
    print(solver_w_voter_pruning(posets, rule_vector=rule_vector))
    print(solver_w_voter_clustering(posets, rule_vector=rule_vector, d=1))
    print(solver_w_voter_clustering(posets, rule_vector=rule_vector, d=2))
    print(solver_w_voter_clustering(posets, rule_vector=rule_vector, d=3))


if __name__ == '__main__':
    main()
