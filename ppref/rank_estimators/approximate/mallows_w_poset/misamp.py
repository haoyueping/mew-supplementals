from bisect import bisect_right
from copy import deepcopy
from functools import partial
from math import ceil, factorial
from multiprocessing import Pool
from random import choices
from time import time

import numpy as np

from ppref.helper import normalize_weights
from ppref.models.mallows import Mallows
from ppref.preferences.linear import Linear
from ppref.preferences.poset import Poset
from ppref.rank_estimators.approximate.amp import amp_sampler, prob_of_amp_drawing_ranking


class Proposal(object):

    def __init__(self, mallows: Mallows, linear: Linear):
        self.mallows = mallows
        self.linear = linear

        self.uniform_ranking_probability = factorial(len(self.linear)) / self.mallows.normalization_constant

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        class_name = self.__class__.__name__
        return f'{class_name}({self.mallows}, {self.linear})'

    def get_insertion_range(self, item, r: list, inserted: set) -> list[int]:
        if not self.linear.has_item(item):
            return list(range(len(r) + 1))

        ancestors = self.linear.get_ancestors(item).intersection(inserted)
        descendants = self.linear.get_descendants(item).intersection(inserted)
        left, right = 0, len(r)
        for idx, e in enumerate(r):
            if e in ancestors:
                left = idx + 1
            elif e in descendants:
                right = idx
                break

        return list(range(left, right + 1))

    def sample_a_permutation(self):
        r, inserted, prob = [], set(), 1
        for rank_i, item in enumerate(self.mallows.reference):
            insertion_range = self.get_insertion_range(item, r, inserted)
            weights = [self.mallows.get_prob_i_j(rank_i, j) for j in insertion_range]
            pos_idx = choices(list(range(len(weights))), weights=weights)[0]

            pos = insertion_range[pos_idx]
            r.insert(pos, item)
            inserted.add(item)
            prob *= weights[pos_idx] / sum(weights)

        return r, prob

    def prob_of_permutation(self, ranking_by_item_to_rank: dict[int, int]) -> float:
        if not self.linear.is_compatible_with(ranking_by_item_to_rank):
            return 0

        r, ranks, inserted, prob = [], [], set(), 1
        for i, item in enumerate(self.mallows.reference):
            item_final_rank = ranking_by_item_to_rank[item]
            pos_j = bisect_right(ranks, item_final_rank)

            probs = [self.mallows.get_prob_i_j(i, j) for j in self.get_insertion_range(item, r, inserted)]
            prob *= self.mallows.get_prob_i_j(i, pos_j) / sum(probs)

            r.insert(pos_j, item)
            inserted.add(item)
            ranks.insert(pos_j, item_final_rank)

        return prob


def calculate_greedy_modals(linear: Linear, mallows: Mallows, max_num=100) -> set[tuple]:
    """
    Algorithm description
    1. It starts with the short linear.
    2. For each item in the Mallows reference, insert it into (likely multiple) positions that minimize the distance.
    """
    ranks = [mallows.get_rank_in_reference(e) for e in linear.get_items()]
    rank_min, rank_max = min(ranks), max(ranks)

    modals, modals_tmp = [linear.to_list()], []
    for item in mallows.reference[rank_min + 1: rank_max]:
        if not linear.has_item(item):
            item_rank_in_center = mallows.get_rank_in_reference(item)
            while modals and len(modals_tmp) < max_num:
                r = modals.pop()
                is_later_item = [mallows.get_rank_in_reference(item_i) > item_rank_in_center for item_i in r]
                dists = [sum(is_later_item[:j]) + len(r) - j - sum(is_later_item[j:]) for j in range(len(r) + 1)]
                insertion_positions = np.random.permutation(np.where(dists == np.amin(dists))[0])
                for pos in insertion_positions:
                    r_new = r.copy()
                    r_new.insert(pos, item)
                    modals_tmp.append(r_new)

            modals.clear()
            modals, modals_tmp = modals_tmp, modals

    return {mallows.reference[:rank_min] + tuple(modal) + mallows.reference[rank_max + 1:] for modal in modals}


def misamp_worker(mallows: Mallows, poset: Poset, proposals: list[Proposal], single_core_workload: int):
    """
    Invoke MIS-AMP on a single core with specified proposal distributions and sample size.
    """
    num_proposals = len(proposals) + 1
    num_rounds = ceil(single_core_workload / num_proposals)
    sample_size = num_proposals * num_rounds

    weight_sum, weight_sq_sum, weight_max, item2counts = 0, 0, 0, {e: np.zeros(mallows.num_items) for e in mallows.reference}
    for _ in range(num_rounds):

        # Sample 1
        while True:
            sample, pr_q_1, distance = amp_sampler(mallows, poset)
            pr_p = mallows.calculate_prob_given_distance(distance)

            item_to_rank = {item: rank for rank, item in enumerate(sample)}
            pr_q_total = pr_q_1
            for prop_id_2, proposal_2 in enumerate(proposals):
                pr_q_2 = proposal_2.prob_of_permutation(item_to_rank)
                pr_q_total += pr_q_2

            if pr_q_total > 0:
                weight_i = num_proposals * pr_p / pr_q_total

                for e, pos in item_to_rank.items():
                    item2counts[e][pos] += weight_i

                weight_sum += weight_i
                weight_sq_sum += weight_i ** 2
                weight_max = max(weight_max, weight_i)

                break
            else:
                print(f'[WARNING] get invalid sample. {sample}')

        # Samples [2, ..., N-1]
        for prop_id_1, proposal_1 in enumerate(proposals):

            while True:
                sample, pr_q_1 = proposal_1.sample_a_permutation()
                pr_p = mallows.calculate_prob_of_permutation(sample)

                item_to_rank = {item: rank for rank, item in enumerate(sample)}
                pr_q_total = prob_of_amp_drawing_ranking(item_to_rank, poset, mallows)
                for prop_id_2, proposal_2 in enumerate(proposals):
                    if prop_id_2 == prop_id_1:
                        pr_q_total += pr_q_1
                    else:
                        pr_q_2 = proposal_2.prob_of_permutation(item_to_rank)
                        pr_q_total += pr_q_2

                if pr_q_total > 0:
                    weight_i = num_proposals * pr_p / pr_q_total

                    for e, pos in item_to_rank.items():
                        item2counts[e][pos] += weight_i

                    weight_sum += weight_i
                    weight_sq_sum += weight_i ** 2
                    weight_max = max(weight_max, weight_i)

                    break
                else:
                    print(f'[WARNING] get invalid sample. {sample}')

    return weight_sum, weight_sq_sum, weight_max, item2counts, sample_size


def misamp_for_weight_max_convergence(mallows: Mallows,
                                      poset: Poset,
                                      proposals: list[Proposal],
                                      convergence_threshold=0.01,
                                      num_cores=None,
                                      timeout_s=1800,
                                      min_rounds=10):
    t_start = time()
    t_boundary = t_start + timeout_s

    worker = partial(misamp_worker, mallows, poset, proposals)
    weight_max, weight_sum, weight_sq_sum, weight_var_old, round_i, num_samples = 0, 0, 0, 0, 0, 0
    item2counts = {e: np.zeros(mallows.num_items) for e in mallows.reference}
    sample_size_list, t_ms_list, item2counts_list = [], [], []
    weight_max_converge_list, weight_var_converge_list = [], []
    while time() < t_boundary:
        round_i += 1

        with Pool(processes=num_cores) as pool:
            res_list = pool.map(worker, [min(2 ** round_i, 100_000) for _ in range(num_cores)])

        for (weight_sum_i, weight_sq_sum_i, weight_max_i, item2counts_i, sample_size_i) in res_list:
            weight_sum += weight_sum_i
            weight_sq_sum += weight_sq_sum_i
            weight_max = max(weight_max, weight_max_i)
            num_samples += sample_size_i

            for e in item2counts_i:
                item2counts[e] += item2counts_i[e]

        sample_size_list.append(num_samples)
        t_ms_list.append(int((time() - t_start) * 1000))
        weight_max_converge_list.append(weight_max / weight_sum)
        item2counts_list.append(deepcopy(item2counts))

        weight_var_new = (weight_sq_sum / num_samples) - ((weight_sum / num_samples) ** 2)
        is_var_converged = (abs(weight_var_new - weight_var_old) < 0.01 * weight_var_old)

        weight_var_convergence_now = None
        if weight_var_old != 0:
            weight_var_convergence_now = abs(weight_var_new - weight_var_old) / weight_var_old

        weight_var_converge_list.append(weight_var_convergence_now)

        if round_i >= min_rounds and is_var_converged and weight_max_converge_list[-1] < convergence_threshold:
            return sample_size_list, t_ms_list, weight_max_converge_list, weight_var_converge_list, item2counts_list, False
        else:
            weight_var_old = weight_var_new

    return sample_size_list, t_ms_list, weight_max_converge_list, weight_var_converge_list, item2counts_list, True


def misamp_ready_w_fixed_sample_size(mallows: Mallows,
                                     poset: Poset,
                                     proposals: list[Proposal],
                                     num_cores=None,
                                     single_core_workload=1000):
    t_start = time()

    worker = partial(misamp_worker, mallows, poset, proposals)
    weight_max, weight_sum, weight_sq_sum, weight_var_old, round_i, num_samples = 0, 0, 0, 0, 0, 0
    item2counts = {e: np.zeros(mallows.num_items) for e in mallows.reference}

    with Pool(processes=num_cores) as pool:
        res_list = pool.map(worker, [single_core_workload for _ in range(num_cores)])

    for (weight_sum_i, weight_sq_sum_i, weight_max_i, item2counts_i, sample_size_i) in res_list:
        weight_sum += weight_sum_i
        weight_sq_sum += weight_sq_sum_i
        weight_max = max(weight_max, weight_max_i)
        num_samples += sample_size_i

        for e in item2counts_i:
            item2counts[e] += item2counts_i[e]

    t_ms = int((time() - t_start) * 1000)
    return weight_sum, weight_sq_sum, num_samples, t_ms, item2counts, False


def misamp_ready(mallows: Mallows,
                 poset: Poset,
                 proposals: list[Proposal],
                 convergence_threshold=0.01,
                 num_cores=None,
                 verbose=False,
                 timeout_s=1800):
    t_start = time()
    t_boundary = t_start + timeout_s

    worker = partial(misamp_worker, mallows, poset, proposals)
    weight_max, weight_sum, weight_sq_sum, weight_var_old, round_i, num_samples = 0, 0, 0, 0, 0, 0
    item2counts = {e: np.zeros(mallows.num_items) for e in mallows.reference}
    while time() < t_boundary:
        round_i += 1

        with Pool(processes=num_cores) as pool:
            res_list = pool.map(worker, [min(2 ** round_i, 100_000) for _ in range(num_cores)])

        for (weight_sum_i, weight_sq_sum_i, weight_max_i, item2counts_i, sample_size_i) in res_list:
            weight_sum += weight_sum_i
            weight_sq_sum += weight_sq_sum_i
            weight_max = max(weight_max, weight_max_i)
            num_samples += sample_size_i

            for e in item2counts_i:
                item2counts[e] += item2counts_i[e]

        if verbose:
            convergence = weight_max / weight_sum
            weight_var_new = (weight_sq_sum / num_samples) - ((weight_sum / num_samples) ** 2)
            weight_var_convergence = 0 if weight_var_old == 0 else abs(weight_var_new - weight_var_old) / weight_var_old

            print(f'    misamp_ready() {round_i=}, sample_size={min(2 ** round_i, 100_000)}, weight_max_{convergence=:.4f}, '
                  f'{weight_var_convergence=:.4f}, {timeout_s=}, t_remain={int(t_boundary - time())}')

        weight_var_new = (weight_sq_sum / num_samples) - ((weight_sum / num_samples) ** 2)
        is_var_converged = (abs(weight_var_new - weight_var_old) < convergence_threshold * weight_var_old)
        if round_i >= 10 and is_var_converged and weight_max < convergence_threshold * weight_sum:
            t_ms = int((time() - t_start) * 1000)
            return weight_sum, weight_sq_sum, num_samples, t_ms, item2counts, False
        else:
            weight_var_old = weight_var_new

    t_ms = int((time() - t_start) * 1000)
    return weight_sum, weight_sq_sum, num_samples, t_ms, item2counts, True


def misamp_for_k_proposals(mallows: Mallows, poset: Poset, delta_k=1, num_cores=10, single_core_workload=1000,
                           verbose=False):
    weight_sums, samples, ks, times_ms, proposals, modals, item2probs_list = [], [], [], [], [], [], []
    linear_generator = poset.get_generator_of_linears()
    modal_to_linear: dict[tuple, Linear] = {}

    for k in range(0, 6, delta_k):
        while k > len(modal_to_linear):
            new_linear = next(linear_generator, None)
            if new_linear is None:
                break
            else:
                for modal in calculate_greedy_modals(new_linear, mallows):
                    if modal not in modal_to_linear:
                        modal_to_linear[modal] = new_linear
                        modals.append(modal)

        for modal in modals[max(0, k - delta_k):k]:
            mallows_i = mallows.get_new_mallows_by_changing_reference(modal)
            proposals.append(Proposal(mallows_i, modal_to_linear[modal]))

        if verbose:
            print(f'  - k = {k + 1}, get {len(proposals) + 1} proposal distributions.')

        weight_sum, weight_sq_sum, num_samples, t_ms, item2counts, is_timeout = misamp_ready_w_fixed_sample_size(
            mallows=mallows, poset=poset, proposals=proposals, num_cores=num_cores,
            single_core_workload=single_core_workload)

        weight_sums.append(weight_sum)
        samples.append(num_samples)
        times_ms.append(t_ms)
        ks.append(len(proposals) + 1)

        item2probs = {e: list(normalize_weights(item2counts[e])) for e in mallows.reference}
        item2probs_list.append(item2probs)

    return item2probs_list, ks, samples, times_ms, weight_sums, f'finished-in-time'


def misamp_adaptive_for_poset(mallows: Mallows, poset: Poset, delta_k=1, convergence_threshold=0.01, num_cores=10,
                              timeout_min=30, verbose=False):
    """
    Algorithm description
    1. Generator of unordered short linears from the poset.
    2. If a short linear is invoked, calculate its greedy modals.
    3. Generate k proposal distributions, and invoke misamp_ready()
    """
    if verbose:
        print(f'Run misamp_adaptive_for_poset({poset})')

    k = -1  # -1 because (mallows, poset) itself is a proposal distribution
    weight_var_old = 0
    weight_sums, samples, ks, times_ms, proposals, modals, item2probs_list = [], [], [], [], [], [], []
    linear_generator = poset.get_generator_of_linears()
    modal_to_linear: dict[tuple, Linear] = {}

    t_remaining_s = timeout_min * 60
    while t_remaining_s > 0:
        k += delta_k

        while k > len(modal_to_linear):
            new_linear = next(linear_generator, None)
            if new_linear is None:
                break
            else:
                for modal in calculate_greedy_modals(new_linear, mallows):
                    if modal not in modal_to_linear:
                        modal_to_linear[modal] = new_linear
                        modals.append(modal)

        for modal in modals[max(0, k - delta_k):k]:
            mallows_i = mallows.get_new_mallows_by_changing_reference(modal)
            proposals.append(Proposal(mallows_i, modal_to_linear[modal]))

        if verbose:
            print(f'  - k = {k + 1}, get {len(proposals) + 1} proposal distributions.')

        weight_sum, weight_sq_sum, num_samples, t_ms, item2counts, is_timeout = misamp_ready(
            mallows=mallows, poset=poset, proposals=proposals, convergence_threshold=convergence_threshold,
            num_cores=num_cores, verbose=verbose, timeout_s=t_remaining_s)

        weight_var_new = (weight_sq_sum / num_samples) - ((weight_sum / num_samples) ** 2)

        weight_sums.append(weight_sum)
        samples.append(num_samples)
        times_ms.append(t_ms)
        t_remaining_s -= t_ms / 1000
        ks.append(len(proposals) + 1)

        try:
            item2probs = {e: list(normalize_weights(item2counts[e])) for e in mallows.reference}
        except AssertionError:
            item2probs_list.append({e: [1 / mallows.num_items for _ in mallows.reference] for e in mallows.reference})
            return item2probs_list, ks, samples, times_ms, weight_sums, f'divide-by-zero'

        item2probs_list.append(item2probs)

        if verbose:
            item = min(poset.items_in_poset)
            print(f'  -> Pr(c@j|{item=}, MISAMP) = {[round(x, ndigits=3) for x in item2probs[item]]}')

        if is_timeout:
            break
        elif abs(weight_var_new - weight_var_old) < convergence_threshold * weight_var_old:
            return item2probs_list, ks, samples, times_ms, weight_sums, f'finished-in-time'
        else:
            weight_var_old = weight_var_new

    return item2probs_list, ks, samples, times_ms, weight_sums, f'unfinished-in-{timeout_min}-min'


def main(m=15, cardinality=10, verbose=True):
    from numpy.random import default_rng
    from ppref.rank_estimators.rim_w_poset import calculate_rank_probs_for_item_given_rim_w_poset_by_sequential

    mallows = Mallows(tuple(range(m)), 1.0)
    poset = Poset.generate_a_random_instance(m=m, cardinality=cardinality, edge_prob=0.1, rng=default_rng())
    item = min(poset.items_in_poset)

    print('=====================')
    print(mallows)
    print(poset)
    print(f'Evaluating {item=}')
    print('=====================')

    t1 = time()
    ans = calculate_rank_probs_for_item_given_rim_w_poset_by_sequential(item, mallows, poset)
    t2 = time()
    probs_answer = ans.get('possibilities', [])

    probs_answer_print = [round(x, ndigits=3) for x in probs_answer]
    print(f'ans_squential = {probs_answer_print} t_sequential = {t2 - t1: .2f}')
    print('=====================')

    t2 = time()
    item2probs_list, ks, samples, times, weight_sums, is_timeout = misamp_adaptive_for_poset(
        mallows=mallows, poset=poset, delta_k=5, convergence_threshold=0.01, num_cores=10, verbose=verbose, timeout_min=3)
    t3 = time()

    probs_acc = item2probs_list[-1][item]
    tv_acc = sum([abs(pi - pj) for pi, pj in zip(probs_acc, probs_answer)])
    print(f'* === TOTAL VARIANCE (acc) is {tv_acc} ===')

    probabilities_print = [round(x, ndigits=3) for x in probs_acc]
    print(f'* {is_timeout}, Pr(c@j|{item=}, MIS-AMP) = {probabilities_print} t_misamp = {t3 - t2: .2f}')


def case_1(verbose=True):
    m = 120
    mallows = Mallows(tuple(range(m)), 1.0)
    poset = Poset(parent_to_children={83: {115}, 103: {98}, 98: {3}, 102: {104}, 51: {19}, 65: {60}, 11: {69}},
                  item_set=set(range(120)))

    item = 3
    probs_answer = [0.0, 0.0, 3.5607463324309572e-06, 1.0682238997294078e-05, 2.1364477994588356e-05, 3.560746332431711e-05,
                    5.341119498647442e-05, 7.477567298105843e-05, 9.970089730808192e-05, 0.00012818686796754012,
                    0.00016023358495939553, 0.00019584104828371355, 0.00023500925794047093, 0.00027773821392967674,
                    0.0003240279162512783, 0.0003738783649053032, 0.00042728955989178655, 0.00048426150121066915,
                    0.0005447941888620342, 0.0006088876228458188, 0.0006765418031619292, 0.0007477567298105219,
                    0.0008225324027916083, 0.0009008688221051395, 0.0009827659877509913, 0.0010682238997293525,
                    0.0011572425580402236, 0.0012498219626834554, 0.0013459621136590143, 0.0014456630109670585,
                    0.001548924654607614, 0.001655747044580612, 0.0017661301808860254, 0.0018800740635238212,
                    0.001997578692494068, 0.0021186440677968144, 0.0022432701894319683, 0.0023714570573993825,
                    0.0025032046716992767, 0.0026385130323316806, 0.002777382139296411, 0.0029198119925935248,
                    0.0030658025922231766, 0.003215353938185394, 0.0033684660304801385, 0.003525138869107289,
                    0.003685372454066778, 0.0038491667853585948, 0.004016521862982822, 0.004187437686939598,
                    0.004361914257228784, 0.004539951573850302, 0.004721549636804401, 0.0049067084460910695,
                    0.005095428001709935, 0.005287708303660899, 0.005483549351944291, 0.0056829511465604065,
                    0.005885913687509141, 0.006092436974790209, 0.00630252100840362, 0.006516165788349633,
                    0.006733371314628208, 0.006954137587239063, 0.007178464606182285, 0.007406352371458077,
                    0.007637800883066182, 0.00787281014100646, 0.008111380145279297, 0.008353510895884837,
                    0.008599202392822611, 0.008848454636092426, 0.009101267625694817, 0.009357641361630174,
                    0.009617575843898066, 0.009881071072497723, 0.010148127047429092, 0.010418743768693178,
                    0.010692921236290679, 0.010970659450220839, 0.011251958410482866, 0.011536818117077564,
                    0.011825238570005054, 0.012117219769264198, 0.012412761714855771, 0.01271186440678037,
                    0.01301452784503685, 0.013320752029625585, 0.013630536960547257, 0.013943882637801297,
                    0.014260789061387788, 0.014581256231306777, 0.014905284147558285, 0.015232872810141776,
                    0.015564022219057115, 0.01589873237430552, 0.016237003275887327, 0.016578834923800204,
                    0.01692422731804634, 0.017273180458623444, 0.017625694345534435, 0.01798176897877937, 0.0183414043583536,
                    0.01870460048426017, 0.01907135735650042, 0.019441674975073136, 0.019815553339976956,
                    0.02019299245121483, 0.02057399230878481, 0.020958552912688246, 0.02134667426292383,
                    0.021738356359490105, 0.022133599202389716, 0.022532402791624242, 0.022934767127187207,
                    0.023340692209086647, 0.02375017803731715, 0.0241632246118764, 0.024579831932772165,
                    0.025000000000002263]

    t2 = time()
    item2probs_list, k, samples, times_ms, weight_sum, is_timeout = misamp_adaptive_for_poset(
        mallows=mallows, poset=poset, delta_k=5, convergence_threshold=0.01, num_cores=10,
        verbose=verbose, timeout_min=1)
    t3 = time()

    probs = item2probs_list[-1][item]
    tv = sum([abs(pi - pj) for pi, pj in zip(probs_answer, probs)])
    print(f'* === TOTAL VARIANCE is {tv} ===')
    print(f'* t_misamp = {t3 - t2: .2f} seconds, weights = {weight_sum}')


def calculate_benchmark():
    import pandas as pd
    from experiments.helper import get_path_to_poset_benchmark

    df_posets = pd.read_csv(get_path_to_poset_benchmark(), delimiter='\t', comment='#')

    for rid, row in df_posets.query('m == 30').sample(n=10).iterrows():
        m = row['m']
        p_max = row['p_max']
        ith = row['ith_poset']
        poset: Poset = eval(row['poset'])
        item = min(poset.items_in_poset)

        print(f'{rid=}, {m=}, {p_max=}, {ith=}, cardinality={len(poset.items_in_poset)}')

        mallows = Mallows(tuple(range(m)), 1.0)

        ans = misamp_for_weight_max_convergence(mallows, poset, [], 0.01, 10)
        sample_size_list, t_ms_list, weight_max_convergence_list, weight_var_converge_list, item2probs_list, is_timeout = ans
        print(f'{is_timeout=}, {sample_size_list=}, {t_ms_list=}\n{weight_max_convergence_list=}')
        for e2ps in item2probs_list:
            print(f'  {normalize_weights(e2ps[item])}')


if __name__ == '__main__':
    main()
