from collections import defaultdict
from copy import deepcopy
from typing import Dict, List

from ppref.models.rim import RepeatedInsertionModel


def calculate_rank_probs_for_item_given_rim(item, rim: RepeatedInsertionModel, max_rank=None) -> List[float]:
    if max_rank is None:
        max_rank = rim.num_items - 1

    item_rank = rim.get_rank_in_reference(item)

    pos2prob: Dict[int, float] = {j: rim.pij_triangle[item_rank][j] for j in range(item_rank + 1)}
    pos2prob_new: Dict[int, float] = defaultdict(float)

    for i in range(item_rank + 1, rim.num_items):
        for pos, prob in pos2prob.items():
            pos2prob_new[pos] += prob * sum(rim.pij_triangle[i][pos + 1:])

            if pos + 1 <= max_rank:
                pos2prob_new[pos + 1] += prob * sum(rim.pij_triangle[i][:pos + 1])

        pos2prob = deepcopy(pos2prob_new)
        pos2prob_new.clear()

    return [pos2prob[k] for k in range(rim.num_items)]


def show_scalability(m=10):
    from time import time
    item = m // 2

    t1 = time()
    rim = RepeatedInsertionModel.generate_a_random_instance(m=m)
    t2 = time()
    calculate_rank_probs_for_item_given_rim(item, rim)
    t3 = time()

    print(f'Time(s) to generate RIM with {m=} : {t2 - t1}')
    print(f'Time(s) to compute  REP with {m=} : {t3 - t2}')


def main():
    for m in [10, 100, 1000]:
        show_scalability(m)


if __name__ == '__main__':
    main()
