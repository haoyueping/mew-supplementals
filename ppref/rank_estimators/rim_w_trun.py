from copy import deepcopy
from typing import Dict, List

from ppref.preferences.combined import RimWithTR
from ppref.rank_estimators.helper import Item2Pos


def calculate_rank_probs_for_item_given_rim_w_trun(item, rim_tr: RimWithTR, max_rank=None) -> List[float]:
    rim = rim_tr.get_rim()
    truncated = rim_tr.get_truncated_ranking()

    if max_rank is None:
        max_rank = rim.num_items - 1

    if item in truncated.truncated_item_to_rank:
        return truncated.calculate_rank_probs_for_item(item)
    else:
        state2prob: Dict[Item2Pos, float] = {Item2Pos(val={}): 1.0}
        state2prob_new: Dict[Item2Pos, float] = {}
        for i, e_i in enumerate(rim.reference):
            for state, prob in state2prob.items():

                if e_i in truncated.truncated_item_to_rank:
                    full_rank_i = truncated.truncated_item_to_rank[e_i]

                    l, r = 0, i
                    if e_i in truncated.items_of_top:
                        r = min(r, full_rank_i)
                    else:
                        l = max(l, i + 1 - (rim.num_items - full_rank_i))

                    for e, pos in state.val.items():
                        if e in truncated.truncated_item_to_rank:
                            if full_rank_i < truncated.truncated_item_to_rank[e]:
                                r = min(r, state.get_pos(e))
                            else:
                                l = max(l, state.get_pos(e) + 1)
                        else:
                            if e_i in truncated.items_of_top:
                                r = min(r, state.get_pos(e))
                            else:
                                l = max(l, state.get_pos(e) + 1)

                    j = l if e_i in truncated.items_of_top else r
                    item2pos = state.val.copy()
                    for e, pos in state.val.items():
                        if pos >= j:
                            item2pos[e] += 1

                    item2pos[e_i] = j

                    state_new = Item2Pos(val=item2pos)
                    prob_new = prob * rim.pij_triangle[i][j]
                    if not (state_new.is_tracking(item) and state_new.get_pos(item) > max_rank):
                        state2prob_new[state_new] = state2prob_new.get(state_new, 0) + prob_new
                elif e_i == item:
                    l, r = 0, min(i, max_rank)
                    item2pos = state.val.copy()
                    for e, pos in state.val.items():
                        if e in truncated.items_of_top:
                            l = max(l, pos + 1)
                        elif e in truncated.items_of_bottom:
                            r = min(r, pos)
                            item2pos[e] += 1

                    for j in range(l, r + 1):
                        item2pos_new = item2pos.copy()
                        item2pos_new[e_i] = j
                        state_new = Item2Pos(val=item2pos_new)
                        prob_new = prob * rim.pij_triangle[i][j]
                        state2prob_new[state_new] = state2prob_new.get(state_new, 0) + prob_new
                else:
                    item2pos = state.val.copy()
                    inserted_top = set(item2pos.keys()).intersection(truncated.items_of_top)
                    inserted_bottom = set(item2pos.keys()).intersection(truncated.items_of_bottom)

                    for e in inserted_bottom:
                        item2pos[e] += 1

                    if state.is_tracking(item):
                        pos_item = state.get_pos(item)

                        if state.get_pos(item) <= max_rank - 1:
                            item2pos_left = item2pos.copy()
                            item2pos_left[item] += 1
                            state_left = Item2Pos(val=item2pos_left)
                            prob_left = prob * sum(rim.pij_triangle[i][len(inserted_top):pos_item + 1])
                            state2prob_new[state_left] = state2prob_new.get(state_left, 0) + prob_left

                        state_right = Item2Pos(val=item2pos)
                        prob_right = prob * sum(rim.pij_triangle[i][pos_item + 1:i + 1 - len(inserted_bottom)])
                        state2prob_new[state_right] = state2prob_new.get(state_right, 0) + prob_right
                    else:
                        state_new = Item2Pos(val=item2pos)
                        prob_new = prob * sum(rim.pij_triangle[i][len(inserted_top):i + 1 - len(inserted_bottom)])
                        state2prob_new[state_new] = state2prob_new.get(state_new, 0) + prob_new

            state2prob = deepcopy(state2prob_new)
            state2prob_new.clear()

        probs = [0.0 for _ in rim.reference]
        for state, prob in state2prob.items():
            probs[state.get_pos(item)] = prob

        return probs
