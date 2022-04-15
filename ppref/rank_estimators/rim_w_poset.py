from collections import defaultdict
from functools import partial
from math import ceil
from multiprocessing import cpu_count, Pool
from random import random
from time import time
from typing import List, Dict, Any

import networkx as nx

from ppref.helper import is_running_out_of_memory, get_ram_gb_size
from ppref.models.rim import RepeatedInsertionModel
from ppref.preferences.poset import Poset
from ppref.rank_estimators.helper import Item2Pos


def get_dag_from_poset(poset: Poset) -> nx.DiGraph:
    parent_to_children = poset.parent_to_children

    dag = nx.DiGraph()
    for parent in parent_to_children:
        for child in parent_to_children[parent]:
            dag.add_edge(parent, child)

    return dag


def calculate_item_to_expiration_step(rim: RepeatedInsertionModel, temporal_hasse_dags: List[nx.DiGraph]) -> Dict[Any, int]:
    e2stop = {e: i for i, e in enumerate(rim.reference)}
    for i in range(rim.num_items - 1):
        e_i = rim.reference[i]
        for j in range(i + 1, rim.num_items):
            e_j = rim.reference[j]
            hasse = temporal_hasse_dags[j]
            if hasse.has_edge(e_i, e_j) or hasse.has_edge(e_j, e_i):
                e2stop[e_i] = j

    return e2stop


def calculate_temporal_hasse_dags(rim: RepeatedInsertionModel, poset: Poset) -> List[nx.DiGraph]:
    dag = get_dag_from_poset(poset)
    dag_tc: nx.DiGraph = nx.transitive_closure_dag(dag)

    hasse_dags: List[nx.DiGraph] = []
    sub_items = set()
    for e in rim.reference:
        sub_items.add(e)
        hasse_i = nx.transitive_reduction(dag_tc.subgraph(sub_items))
        hasse_dags.append(hasse_i)

    return hasse_dags


def calculate_rank_probs_for_item_given_rim_w_poset_by_parallel(item, rim: RepeatedInsertionModel, poset: Poset,
                                                                threads: int = None, t_max_min=30, verbose=False,
                                                                max_rank=None):
    threads = threads or cpu_count()

    if max_rank is None:
        max_rank = rim.num_items - 1

    if verbose:
        print(f'\n{rim}')
        print(f'Evaluate {item=} from posset={poset.parent_to_children}')

    if threads == 1:
        return calculate_rank_probs_for_item_given_rim_w_poset_by_sequential(item, rim, poset, t_max_min, verbose, max_rank)

    ram_gb = get_ram_gb_size()
    round_size_max = 1_000_000 if ram_gb < 50 else 10_000_000  # whether RAM is less than 50 GB

    cover_width = 0
    num_states = 0

    t0 = time()

    temporal_hasse_dags = calculate_temporal_hasse_dags(rim, poset)
    e2stop = calculate_item_to_expiration_step(rim, temporal_hasse_dags)
    e2stop[item] = rim.num_items

    state2prob: Dict[Item2Pos, float] = defaultdict(float)
    state2prob_new: Dict[Item2Pos, float] = defaultdict(float)
    state2prob[Item2Pos(val={})] = 1.0

    for step_i, e_i in enumerate(rim.reference):
        hasse = temporal_hasse_dags[step_i]

        t1 = time()
        if t1 - t0 > t_max_min * 60:
            return {'has_answer': False, 'error': f'unfinished-in-{t_max_min}-min'}

        worker_partial = partial(worker, rim, hasse, e2stop, step_i, e_i, item)
        while state2prob:
            single_worker_size = ceil(min(round_size_max, len(state2prob)) / threads)
            batches = []
            for _ in range(threads):
                batch = [state2prob.popitem() for _ in range(min(single_worker_size, len(state2prob)))]
                batches.append(batch)

            if verbose:
                print([len(b) for b in batches])

            with Pool(threads) as pool:
                try:
                    all_res = pool.map(worker_partial, batches)
                except Exception as e:
                    pool.terminate()
                    print(f'OutOfMemory, {e.__class__}')
                    return {'has_answer': False, 'error': 'out-of-memory'}

            if {'has_answer': False} in all_res:
                nstates = 0
                for res in all_res:
                    if res['has_answer']:
                        nstates += (len(res['answer']))
                print(f'OutOfMemory when mapping producing at least {nstates} states.')
                return {'has_answer': False, 'error': 'out-of-memory'}
            else:
                if len(state2prob_new) == 0:
                    state2prob_new = all_res.pop()['answer']

                while all_res:
                    answer = all_res.pop()['answer']
                    while answer:
                        s, p = answer.popitem()
                        state2prob_new[s] += p

        state2prob.clear()
        state2prob, state2prob_new = state2prob_new, state2prob

        if verbose:
            t2 = time()
            print(f'    - t_{step_i}  = {t2 - t1: .2f} s')

        cover_width = max(cover_width, next(iter(state2prob)).size())
        num_states += len(state2prob)

    probabilities = [0.0 for _ in range(rim.num_items)]
    for state, prob in state2prob.items():
        probabilities[state.get_pos(item)] += prob

    return {'has_answer': True, 'probabilities': probabilities, 'cover_width': cover_width, 'num_states': num_states}


def worker(rim, hasse, e2stop, step_i, e_i, item, state_prob_list) -> dict:
    state2prob_new: Dict[Item2Pos, float] = defaultdict(float)
    for ith, (state, prob) in enumerate(state_prob_list):

        if random() < 0.001 and is_running_out_of_memory(verbose=False):
            return {'has_answer': False}

        if hasse.has_node(e_i):
            left, right = 0, step_i
            for predecessor in hasse.predecessors(e_i):
                if state.is_tracking(predecessor):
                    left = max(left, state.val[predecessor] + 1)
            for successor in hasse.successors(e_i):
                if state.is_tracking(successor):
                    right = min(right, state.val[successor])

            for j in range(left, right + 1):
                e2k = {}
                if e2stop[e_i] > step_i:
                    e2k[e_i] = j

                for e, k in state.val.items():
                    if e2stop[e] > step_i:
                        if j <= k:
                            e2k[e] = k + 1
                        else:
                            e2k[e] = k

                state_new = Item2Pos(val=e2k)
                prob_new = prob * rim.pij_triangle[step_i][j]
                state2prob_new[state_new] += prob_new
        elif e_i == item:
            for j in range(step_i + 1):
                e2k = {e_i: j}

                for e, k in state.val.items():
                    if e2stop[e] > step_i:
                        if j <= k:
                            e2k[e] = k + 1
                        else:
                            e2k[e] = k

                state_new = Item2Pos(val=e2k)
                prob_new = prob * rim.pij_triangle[step_i][j]
                state2prob_new[state_new] += prob_new
        else:
            seq = sorted(state.val.keys(), key=lambda x: state.get_pos(x))
            if len(seq) == 0:
                state2prob_new[state] = prob
            elif len(seq) == 1:
                item_in_seq = seq[0]
                rank = state.val[item_in_seq]

                state_1 = Item2Pos(val={item_in_seq: rank + 1})
                prob_1 = prob * sum(rim.pij_triangle[step_i][:rank + 1])
                state2prob_new[state_1] += prob_1

                prob_2 = prob * sum(rim.pij_triangle[step_i][rank + 1:])
                state2prob_new[state] += prob_2
            else:
                state_1 = Item2Pos(val={e: k + 1 for e, k in state.val.items()})
                prob_1 = prob * sum(rim.pij_triangle[step_i][:state.val[seq[0]] + 1])
                state2prob_new[state_1] += prob_1

                prob_last = prob * sum(rim.pij_triangle[step_i][state.val[seq[-1]] + 1:])
                state2prob_new[state] += prob_last

                for idx_seq in range(len(seq) - 1):
                    e2k = {e: state.get_pos(e) for e in seq[:idx_seq + 1]}
                    for e in seq[idx_seq + 1:]:
                        e2k[e] = state.get_pos(e) + 1

                    state_new = Item2Pos(val=e2k)

                    pos_left = state.get_pos(seq[idx_seq])
                    pos_right = state.get_pos(seq[idx_seq + 1])
                    prob_new = prob * sum(rim.pij_triangle[step_i][pos_left + 1:pos_right + 1])

                    state2prob_new[state_new] += prob_new

    return {'has_answer': True, 'answer': state2prob_new}


def calculate_rank_probs_for_item_given_rim_w_poset_by_sequential(item, rim: RepeatedInsertionModel, poset: Poset,
                                                                  t_max_min=30, verbose=False, max_rank=None):
    t0 = time()

    if max_rank is None:
        max_rank = rim.num_items - 1

    cover_width = 0
    num_states = 0

    temporal_hasse_dags = calculate_temporal_hasse_dags(rim, poset)
    e2stop = calculate_item_to_expiration_step(rim, temporal_hasse_dags)
    e2stop[item] = rim.num_items

    state2prob: Dict[Item2Pos, float] = defaultdict(float)
    state2prob_new: Dict[Item2Pos, float] = defaultdict(float)
    state2prob[Item2Pos(val={})] = 1.0

    for step_i, e_i in enumerate(rim.reference):
        hasse = temporal_hasse_dags[step_i]
        t1 = time()

        if t1 - t0 > t_max_min * 60:
            return {'has_answer': False, 'error': f'unfinished-in-{t_max_min}-min'}

        while state2prob:
            state, prob = state2prob.popitem()

            if random() < 0.001 and is_running_out_of_memory(verbose=False):
                return {'has_answer': False, 'error': 'out-of-memory'}

            if hasse.has_node(e_i):
                left, right = 0, step_i
                for predecessor in hasse.predecessors(e_i):
                    if state.is_tracking(predecessor):
                        left = max(left, state.val[predecessor] + 1)
                for successor in hasse.successors(e_i):
                    if state.is_tracking(successor):
                        right = min(right, state.val[successor])

                for j in range(left, right + 1):
                    e2k = {}
                    if e2stop[e_i] > step_i:
                        e2k[e_i] = j

                    for e, k in state.val.items():
                        if e2stop[e] > step_i:
                            if j <= k:
                                e2k[e] = k + 1
                            else:
                                e2k[e] = k

                    state_new = Item2Pos(val=e2k)
                    prob_new = prob * rim.pij_triangle[step_i][j]
                    if not (state_new.is_tracking(item) and state_new.get_pos(item) > max_rank):
                        state2prob_new[state_new] += prob_new
            elif e_i == item:
                for j in range(min(step_i + 1, max_rank + 1)):
                    e2k = {e_i: j}

                    for e, k in state.val.items():
                        if e2stop[e] > step_i:
                            if j <= k:
                                e2k[e] = k + 1
                            else:
                                e2k[e] = k

                    state_new = Item2Pos(val=e2k)
                    prob_new = prob * rim.pij_triangle[step_i][j]
                    state2prob_new[state_new] += prob_new
            else:
                seq = sorted(state.val.keys(), key=lambda x: state.get_pos(x))
                if len(seq) == 0:
                    state2prob_new[state] = prob
                elif len(seq) == 1:
                    item_in_seq = seq[0]
                    rank = state.val[item_in_seq]

                    state_1 = Item2Pos(val={item_in_seq: rank + 1})
                    prob_1 = prob * sum(rim.pij_triangle[step_i][:rank + 1])
                    if not (item_in_seq == item and state_1.get_pos(item) > max_rank):
                        state2prob_new[state_1] += prob_1

                    prob_2 = prob * sum(rim.pij_triangle[step_i][rank + 1:])
                    state2prob_new[state] += prob_2
                else:
                    state_1 = Item2Pos(val={e: k + 1 for e, k in state.val.items()})
                    prob_1 = prob * sum(rim.pij_triangle[step_i][:state.val[seq[0]] + 1])
                    if not (state_1.is_tracking(item) and state_1.get_pos(item) > max_rank):
                        state2prob_new[state_1] += prob_1

                    prob_last = prob * sum(rim.pij_triangle[step_i][state.val[seq[-1]] + 1:])
                    state2prob_new[state] += prob_last

                    for idx_seq in range(len(seq) - 1):
                        e2k = {e: state.get_pos(e) for e in seq[:idx_seq + 1]}
                        for e in seq[idx_seq + 1:]:
                            e2k[e] = state.get_pos(e) + 1

                        state_new = Item2Pos(val=e2k)

                        if not (state_new.is_tracking(item) and state_new.get_pos(item) > max_rank):
                            pos_left = state.get_pos(seq[idx_seq])
                            pos_right = state.get_pos(seq[idx_seq + 1])
                            prob_new = prob * sum(rim.pij_triangle[step_i][pos_left + 1:pos_right + 1])

                            state2prob_new[state_new] += prob_new

        state2prob.clear()
        state2prob, state2prob_new = state2prob_new, state2prob

        if verbose:
            print(f't_{step_i} = {time() - t1: .2f} s')

        if not state2prob:
            return {'has_answer': True, 'probabilities': [0 for _ in rim.reference], 'cover_width': cover_width,
                    'num_states': num_states}

        cover_width = max(cover_width, next(iter(state2prob)).size())
        num_states += len(state2prob)

    probabilities = [0.0 for _ in rim.reference]
    for state, prob in state2prob.items():
        probabilities[state.get_pos(item)] += prob

    return {'has_answer': True, 'probabilities': probabilities, 'cover_width': cover_width, 'num_states': num_states}


def calculate_cover_width(poset: Poset, rim: RepeatedInsertionModel, target_item):
    temporal_hasse_dags = calculate_temporal_hasse_dags(rim, poset)
    e2stop = calculate_item_to_expiration_step(rim, temporal_hasse_dags)
    e2stop[target_item] = rim.num_items

    widths = [0 for _ in rim.reference]
    for e in e2stop:
        start = rim.get_rank_in_reference(e)
        stop = e2stop[e]
        for i in range(start, stop):
            widths[i] += 1

    return max(widths)


def show_scalability(m=10, cardinality=5, max_rank=None, verbose=True):
    from ppref.models.mallows import Mallows
    from numpy.random import default_rng

    rng = default_rng(m)
    mallows = Mallows(tuple(range(m)), 1)
    poset = Poset.generate_a_random_instance(m=m, cardinality=cardinality, edge_prob=0.1, rng=rng)
    item = min(poset.item_set - poset.items_in_poset)

    t1 = time()
    _ = calculate_rank_probs_for_item_given_rim_w_poset_by_sequential(item, mallows, poset)
    t2 = time()

    if verbose:
        print(f'\nEvaluating item {item}')
        print(mallows)
        print(poset)
        print(f'when {m=} and {cardinality=}, time_sequential={t2 - t1} seconds')


def main():
    for k in range(10, 20, 2):
        show_scalability(m=k, cardinality=k)


if __name__ == '__main__':
    main()
