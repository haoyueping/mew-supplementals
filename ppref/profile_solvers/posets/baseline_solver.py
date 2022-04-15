from time import time

from ppref.models.mallows import Mallows
from ppref.preferences.poset import Poset
from ppref.rank_estimators.rim_w_poset import calculate_rank_probs_for_item_given_rim_w_poset_by_sequential


def solver_baseline(posets: list[Poset], rule_vector: list[int]):
    candidates = posets[0].item_set
    mallows = Mallows(reference=tuple(candidates), phi=1.0)
    max_rank = rule_vector.index(0) - 1
    candidate2score = {c: 0.0 for c in candidates}

    t1 = time()

    for poset in posets:
        for c in candidates:
            if c in poset.items_in_poset:
                answer = calculate_rank_probs_for_item_given_rim_w_poset_by_sequential(item=c, rim=mallows, poset=poset,
                                                                                       max_rank=max_rank)
                probs = answer['probabilities']
            else:
                probs = [1 / mallows.num_items for _ in range(max_rank + 1)]

            exact_score = sum([prob * val for prob, val in zip(probs[:max_rank + 1], rule_vector[: max_rank + 1])])
            candidate2score[c] += exact_score

    winner_score = max(candidate2score.values())
    winners = tuple([c for c, s in candidate2score.items() if s == winner_score])
    t2 = time()
    return {'winners': winners, 'winner_score': winner_score, 't_sec': t2 - t1}
