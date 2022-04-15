from typing import Tuple, Sequence

from numpy.random import Generator as NumpyRNG, default_rng

from ppref.helper import normalize_weights
from ppref.models.mallows import Mallows


class RepeatedSelectionModelRV(object):
    """Ranking version (RV) of RSM."""

    def __init__(self, reference: tuple, pij_triangle: Tuple[Tuple[float]]):
        self.reference: tuple = reference
        self.pij_triangle: Tuple[Tuple[float]] = pij_triangle

        self.num_items = len(reference)
        self.item_to_rank = {item: rank for rank, item in enumerate(reference)}

    def __str__(self):
        lines = ['[Repeated Selection Model (ranking version)]', f'  *reference = {self.reference}',
                 f'  *Pi        = {self.pij_triangle[0]}']
        for i in range(1, len(self.reference)):
            lines.append(f'               {self.pij_triangle[i]}')

        return '\n'.join(lines)

    def __repr__(self):
        return f'RepeatedSelectionModelRV(reference={self.reference}, pij_triangle={self.pij_triangle})'

    def __eq__(self, other):
        return isinstance(other, RepeatedSelectionModelRV) and other.reference == self.reference \
               and other.pij_triangle == self.pij_triangle

    def get_full_item_set(self):
        return set(self.item_to_rank)

    def get_probability_of_selecting_position_j_at_step_i(self, i, j) -> float:
        return self.pij_triangle[i][j]

    def get_rank_in_reference(self, item):
        return self.item_to_rank[item]

    def get_range_of_possible_ranks(self, item=None):
        return 0, self.num_items - 1

    def sample_a_ranking(self, rng: NumpyRNG = None) -> list:
        rng = rng or default_rng()
        reference = list(self.reference)
        selection_range = list(range(self.num_items))
        ranking = []
        for step in range(self.num_items):
            sample_index = rng.choice(a=selection_range, p=self.pij_triangle[step], shuffle=False)

            ranking.insert(sample_index, reference.pop(sample_index))
            selection_range.pop()

        return ranking

    def sample_a_permutation(self, rng: NumpyRNG = None) -> list:
        return self.sample_a_ranking(rng)

    def calculate_prob_of_permutation(self, permutation: Sequence):
        assert len(permutation) == self.num_items

        reference = list(self.reference)
        prob = 1
        for step in range(len(self.reference)):
            item = permutation[step]
            rank = reference.index(item)
            prob *= self.pij_triangle[step][rank]
            del reference[rank]

        return prob

    @classmethod
    def generate_a_random_instance(cls, m: int, rng: NumpyRNG = None):
        rng = rng or default_rng()

        reference = tuple(range(m))
        pij_matrix: list[tuple[float]] = [tuple(normalize_weights(rng.random(size=i))) for i in range(m, 0, -1)]

        return cls(reference, tuple(pij_matrix))

    @classmethod
    def generate_instance_from_mallows(cls, mallows: Mallows):
        m = mallows.num_items
        reference = mallows.reference
        pij_matrix: list[tuple[float]] = []
        for i in range(m):
            probs = []
            for j in range(m - i):
                probs.append(mallows.phi ** j)

            prob_sum = sum(probs)
            probs = [prob / prob_sum for prob in probs]

            pij_matrix.append(tuple(probs))

        return cls(reference, tuple(pij_matrix))
