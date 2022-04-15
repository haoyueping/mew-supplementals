import random
from decimal import Decimal
from multiprocessing import Pool, cpu_count
from statistics import median
from typing import Sequence, List, Dict, Any

import numpy as np

from ppref.helper import calculate_kendall_tau_distance, normalize_weights
from ppref.models.mallows import Mallows
from ppref.preferences.poset import Poset
from ppref.rank_estimators.approximate.amp import amp_sampler


def calculate_mahonian_triangle(m):
    """
    Calculate the triangle of Mahonian numbers T(m, d), the number of permutations of length m at distance d.

    https://oeis.org/A008302
    """
    triangle = np.zeros((m + 1, m * (m - 1) // 2 + 1), dtype=Decimal)
    triangle[:, 0] = 1

    for i in range(1, m + 1):
        for j in range(1, i * (i - 1) // 2 + 1):
            if j >= i:
                triangle[i, j] = triangle[i, j - 1] + triangle[i - 1, j] - triangle[i - 1, j - i]
            else:
                triangle[i, j] = triangle[i, j - 1] + triangle[i - 1, j]

    return triangle


def calculate_insertion_probabilities(m, verbose=False):
    sum_mt = np.cumsum(calculate_mahonian_triangle(m), axis=1)
    ws = {}
    for mi in range(1, m + 1):

        if verbose:
            print(f'calculating row of m = {mi}')

        for dist_full in range(mi * (mi - 1) // 2 + 1):
            dist_max_by_insertion = min(dist_full, mi - 1)
            if dist_full == dist_max_by_insertion:
                weights = (sum_mt[mi - 1, dist_full::-1] / sum_mt[mi - 1, dist_full]).astype(np.float16)
                ws[(mi, dist_full)] = weights / weights.sum()
            else:
                weights = (sum_mt[mi - 1, dist_full:dist_full - dist_max_by_insertion - 1:-1] /
                           sum_mt[mi - 1, dist_full]).astype(np.float16)
                ws[(mi, dist_full)] = weights / weights.sum()

    return ws


def sample_ranking_within_distance(ranking_origin, distance, dist_vector_probs):
    m = len(ranking_origin)
    distance_options = np.arange(m, dtype=np.uint8)
    distance_vector = np.zeros(m, dtype=np.uint8)  # distances contributed by each item
    for e_i in range(m - 1, -1, -1):
        mi = min(distance, e_i)
        if (e_i + 1, distance) in dist_vector_probs:
            distance_val = random.choices(distance_options[:mi + 1], weights=dist_vector_probs[(e_i + 1, distance)])[0]
        else:
            distance_val = random.choices(distance_options[:mi + 1])[0]

        distance_vector[e_i] = distance_val
        distance -= distance_val

        if distance == 0:
            break

    r_new = []
    for i, e in enumerate(ranking_origin):
        r_new.insert(i - distance_vector[i], e)
    return r_new


def get_a_random_linear_extension(center: Sequence, poset: Poset):
    center_new = list(center)
    random.shuffle(center_new)
    r, _, _ = amp_sampler(Mallows(tuple(center_new), 1), poset)
    return r


def slice_sampler(mallows: Mallows, poset: Poset, r_old: List, dist_vector_probs, strategy='midpoint', verbose=False):
    y = random.random() * mallows.calculate_prob_of_permutation(r_old)

    r_new = list(mallows.reference)
    random.shuffle(r_new)

    if mallows.calculate_prob_of_permutation(r_new) >= y and poset.has_such_linear_extension(r_new):
        if verbose:
            print(f'Return ranking immediately.')

        return r_new

    distance = mallows.num_items * (mallows.num_items - 1) // 2
    distance_tenth = distance // 10
    while True:
        if strategy == 'backoff':
            distance = calculate_kendall_tau_distance(r_old, r_new) - 1
        elif strategy == 'midpoint':
            distance //= 2
        else:
            distance -= distance_tenth
            distance = max(0, distance)

        r_new = sample_ranking_within_distance(r_old, distance, dist_vector_probs)

        if mallows.calculate_prob_of_permutation(r_new) >= y and poset.has_such_linear_extension(r_new):
            if verbose:
                print(f'Return ranking within distance boundary {distance}')

            return r_new


def slice_sampling_for_explicit_distribution_by_single_thread(mid: int, mallows: Mallows, poset: Poset,
                                                              dist_vector_probs, strategy='midpoint', warm_up_steps=10,
                                                              verbose=False, seed=0):
    random.seed(seed + mid)

    ranking = get_a_random_linear_extension(mallows.reference, poset)
    for _ in range(warm_up_steps):
        ranking = slice_sampler(mallows, poset, ranking, dist_vector_probs, strategy, False)

    item_to_rank_counts = {item: [0 for _ in mallows.reference] for item in mallows.reference}
    num_samples, distance_sum, distance_square_sum, distance_var_old, convergence_count = 0, 0, 0, 0, 0
    while True:
        num_samples += 1
        ranking = slice_sampler(mallows, poset, ranking, dist_vector_probs, strategy, False)

        for rank, item in enumerate(ranking):
            item_to_rank_counts[item][rank] += 1

        distance_new = mallows.calculate_kendall_tau_distance(ranking)
        distance_sum += distance_new
        distance_square_sum += distance_new ** 2

        if num_samples % 1000 == 0:
            distance_var_new = (distance_square_sum / num_samples) - ((distance_sum / num_samples) ** 2)
            has_converged = (abs(distance_var_new - distance_var_old) < 0.01 * distance_var_old)

            if has_converged:
                convergence_count += 1
            else:
                convergence_count = 0

            if verbose:
                print(f'Machine-{mid}, {num_samples} samples, {convergence_count} convergences, VarOld(dist)'
                      f'={distance_var_old:.3f}, VarNew(dist)={distance_var_new:.3f}, has_converged={has_converged}')

            distance_var_old = distance_var_new

            if convergence_count == 3:
                break

    return {item: normalize_weights(rank_counts) for item, rank_counts in item_to_rank_counts.items()}, num_samples


def slice_sampling_for_explicit_distribution_by_multi_thread(mallows: Mallows, poset: Poset, dist_vector_probs,
                                                             strategy='midpoint', num_threads=None, warm_up_steps=10,
                                                             verbose=False, seed=0):
    num_threads = num_threads or cpu_count() - 2

    with Pool(num_threads) as pool:
        tasks = [(mid, mallows, poset, dist_vector_probs,
                  strategy, warm_up_steps, verbose, seed) for mid in range(num_threads)]
        res_all = pool.starmap(slice_sampling_for_explicit_distribution_by_single_thread, tasks)

    item_to_rank_to_probs: Dict[Any, Dict[int, list]] = {item: {rank: [] for rank in range(mallows.num_items)} for item
                                                         in
                                                         mallows.reference}
    num_samples = 0
    for (res_i, size_i) in res_all:
        num_samples += size_i
        for item, probs in res_i.items():
            for rank, prob in enumerate(probs):
                item_to_rank_to_probs[item][rank].append(prob)

    item_to_probs = {}
    for item in mallows.reference:
        probs = []
        for rank in range(mallows.num_items):
            probs.append(median(item_to_rank_to_probs[item][rank]))

        item_to_probs[item] = normalize_weights(probs)

    return item_to_probs, num_samples


def get_distance_vector_probabilities(m=100):
    import pickle
    from experiments.helper import get_project_root_path

    data_file = get_project_root_path() / f'data/distance_vector_probabilities_for_{m}_items.pkl'
    with open(data_file, 'rb') as f:
        return pickle.load(f)


def run_ss_for_distribution(strategy='midpoint'):
    from itertools import permutations

    m, phi = 9, 1
    poset = Poset({8: {0, 5}, 7: {5, 2}, 5: {4, 3}}, item_set=set(range(m)))
    # poset = Poset.generate_a_random_instance(m, m // 2)

    mallows = Mallows(tuple(range(m)), phi)
    dist_vector_probs = get_distance_vector_probabilities(100)
    print(mallows, '\n', poset)

    from time import time
    t0 = time()
    item_to_probs, num_samples = slice_sampling_for_explicit_distribution_by_multi_thread(mallows, poset,
                                                                                          dist_vector_probs,
                                                                                          strategy, num_threads=10,
                                                                                          warm_up_steps=10,
                                                                                          verbose=True)
    t1 = time()
    print(f'\n=== t_ss = {t1 - t0: .3f} sec, #samples={num_samples}')

    if m <= 12:
        answer = {item: np.zeros(m, dtype=int) for item in mallows.reference}
        for r in permutations(mallows.reference):
            if poset.has_such_linear_extension(r):
                for rank, item in enumerate(r):
                    answer[item][rank] += 1

        for item in mallows.reference:
            print(f'For item {item}, res={np.round(normalize_weights(item_to_probs[item]), decimals=3)}')
            print(f'For item {item}, ans={np.round(normalize_weights(answer[item]), decimals=3)}')
    else:
        for item in mallows.reference:
            print(
                f'For item {item}, t={t1 - t0:.3f} s, res={np.round(normalize_weights(item_to_probs[item]), decimals=5)}')


def main():
    run_ss_for_distribution()


if __name__ == '__main__':
    main()
