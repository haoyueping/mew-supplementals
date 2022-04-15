from typing import List

from ppref.models.mallows import Mallows
from ppref.preferences.combined import MallowsWithPP
from ppref.rank_estimators.rim import calculate_rank_probs_for_item_given_rim


def calculate_rank_probs_for_item_given_mallows_w_pp(item, mallows_pp: MallowsWithPP, max_rank=None) -> List[float]:
    mallows = mallows_pp.get_mallows()
    partitioned = mallows_pp.get_partitioned_preferences()

    if max_rank is None:
        max_rank = mallows.num_items - 1

    bucket = partitioned.item_to_bucket[item]
    items_in_bucket = partitioned.bucket_to_items[bucket]
    sorted_bucket = sorted(items_in_bucket, key=lambda x: mallows.get_rank_in_reference(x))

    bucket_rank = partitioned.bucket_to_rank[bucket]
    offset_left = sum([len(partitioned.bucket_to_items[b]) for b in partitioned.bucket_order[:bucket_rank]])

    mallows_tiny = Mallows(reference=tuple(sorted_bucket), phi=mallows.phi)
    probs_tiny = calculate_rank_probs_for_item_given_rim(item, mallows_tiny, max_rank=max_rank - offset_left)

    probs = [0.0 for _ in mallows.reference]
    for rank_sub, prob in enumerate(probs_tiny):
        probs[rank_sub + offset_left] = prob

    return probs
