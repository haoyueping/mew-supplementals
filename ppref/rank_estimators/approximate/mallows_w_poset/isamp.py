import random
from copy import deepcopy
from multiprocessing.pool import Pool
from os import cpu_count
from typing import Dict, Any, Tuple

import numpy as np

from ppref.helper import normalize_weights
from ppref.models.mallows import Mallows
from ppref.preferences.poset import Poset
from ppref.rank_estimators.approximate.amp import amp_sampler


def worker(poset: Poset, mallows: Mallows, single_core_workload: int, seed_worker: int) -> Dict[Any, np.ndarray]:
    random.seed(seed_worker)

    item_to_rank_counts = {item: np.zeros_like(mallows.reference, dtype=float) for item in mallows.reference}
    for _ in range(single_core_workload):
        r, prob_proposal, distance = amp_sampler(mallows, poset)
        prob_origin = mallows.calculate_prob_given_distance(distance)
        weight_i = prob_origin / prob_proposal

        for rank, item in enumerate(r):
            item_to_rank_counts[item][rank] += weight_i

    return item_to_rank_counts


def estimate_poset_by_isamp(poset: Poset, mallows: Mallows, convergence_threshold=0.01, single_core_workload=50,
                            num_cores: int = None, verbose=False, seed=0) -> Tuple[Dict[Any, list], int]:
    num_cores = num_cores or cpu_count()
    random.seed(seed)
    base_seed = random.randint(0, 1_000_000_000)

    poset = deepcopy(poset)
    poset.calculate_tc()

    dummy_item = min(poset.items_in_poset)

    if verbose:
        print(f'Running\n- {poset}\n- {mallows}\n')

    single_round_sample_size = single_core_workload * num_cores

    item2counts = {item: np.zeros_like(mallows.reference, dtype=float) for item in mallows.reference}
    item2probs_old = {item: np.zeros_like(mallows.reference, dtype=float) for item in mallows.reference}
    probs_diff_max, round_i, convergence_count = 0., 0, 0
    while True:
        round_i += 1
        num_samples = round_i * single_round_sample_size
        base_seed_i = base_seed + (round_i * num_cores)

        with Pool(processes=num_cores) as pool:

            tasks = [(poset, mallows, single_core_workload, base_seed_i + wid) for wid in range(num_cores)]
            res_list = pool.starmap(worker, tasks)

        for item_to_rank_counts_i in res_list:
            for item, rank_counts in item_to_rank_counts_i.items():
                item2counts[item] += rank_counts

        item2probs_new = {item: normalize_weights(item2counts[item]) for item in mallows.reference}

        is_converged = False
        probs_diff_max = max([abs(item2probs_old[e] - item2probs_new[e]).sum() for e in mallows.reference])
        if probs_diff_max < mallows.num_items * convergence_threshold:
            convergence_count += 1
            is_converged = (convergence_count == 3)
        else:
            convergence_count = 0

        if is_converged:
            return {e: probs.tolist() for e, probs in item2probs_new.items()}, num_samples
        elif verbose:
            print(f'#samples={num_samples}, {convergence_count=}, {probs_diff_max=}, '
                  f'probs({dummy_item})={item2probs_new[dummy_item].round(3)}')

        item2probs_old = item2probs_new.copy()


def get_answer(mallows: Mallows, poset: Poset):
    from itertools import permutations
    item2probs = {item: np.zeros_like(mallows.reference, dtype=float) for item in mallows.reference}
    for r in permutations(mallows.reference):
        if poset.has_such_linear_extension(r):
            prob = mallows.calculate_prob_of_permutation(r)

            for k, e in enumerate(r):
                item2probs[e][k] += prob

    return {e: normalize_weights(item2probs[e]) for e in mallows.reference}


def main():
    from time import time
    from numpy.random import default_rng
    rng = default_rng(0)

    m = 20
    mallows = Mallows(tuple(range(m)), 1)
    poset = Poset.generate_a_random_instance(m, cardinality=m // 2, rng=rng)

    t0 = time()
    item2probs, num_samples = estimate_poset_by_isamp(poset, mallows, convergence_threshold=0.01, single_core_workload=1000,
                                                      num_cores=10, verbose=True, seed=0)
    t1 = time()

    print(f'\n=== Summary ({num_samples=}, {int(t1 - t0)}) seconds ===')
    for e in mallows.reference:
        print(f'For item {e}, estimation = {[round(p, ndigits=3) for p in item2probs[e]]}')


if __name__ == '__main__':
    main()
