from collections import defaultdict
from copy import deepcopy
from typing import Dict, Tuple

from ppref.models.rsm import RepeatedSelectionModelRV


def calculate_rank_prob_for_item_given_rsm(item, rsm: RepeatedSelectionModelRV, max_rank=None) -> list[float]:
    if max_rank is None:
            max_rank = rsm.num_items - 1

    state2prob: Dict[Tuple[int, int], float] = defaultdict(float)
    state2prob_new: Dict[Tuple[int, int], float] = defaultdict(float)

    idx_item = rsm.get_rank_in_reference(item)
    alpha0, beta0 = idx_item, rsm.num_items - idx_item - 1
    state2prob[(alpha0, beta0)] = 1.0

    probs = []
    for i in range(max_rank):

        prob_i = 0
        for (alpha, beta), prob in state2prob.items():
            prob_i += prob * rsm.pij_triangle[i][alpha]

            if alpha > 0:
                state2prob_new[(alpha - 1, beta)] += prob * sum(rsm.pij_triangle[i][:alpha])
            if beta > 0:
                state2prob_new[(alpha, beta - 1)] += prob * sum(rsm.pij_triangle[i][alpha + 1:alpha + beta + 2])

        probs.append(prob_i)
        state2prob = deepcopy(state2prob_new)
        state2prob_new.clear()

    probs.append(sum([prob * rsm.pij_triangle[max_rank][state[0]] for state, prob in state2prob.items()]))

    return probs
