from typing import List, Tuple, Sequence

from numpy.random import Generator as NumpyRNG, default_rng

from ppref.helper import normalize_weights


class RepeatedInsertionModel(object):
    def __init__(self, reference: tuple, pij_triangle: Tuple[Tuple[float]]):
        assert len(reference) > 0

        self.reference = reference
        self.pij_triangle: Tuple[Tuple[float]] = pij_triangle

        self.num_items = len(reference)
        self.item_to_rank = {item: rank for rank, item in enumerate(reference)}

    def __str__(self):
        lines = [f'[Repeated Insertion Model]\n  *reference = {self.reference}\n  *Pi        = {self.pij_triangle[0]}']
        for i in range(1, len(self.reference)):
            lines.append(f'               {self.pij_triangle[i]}')

        return '\n'.join(lines)

    def __repr__(self):
        return f'RepeatedInsertionModel(reference={self.reference}, pij_triangle={self.pij_triangle})'

    def __eq__(self, other):
        return isinstance(other, RepeatedInsertionModel) and other.reference == self.reference \
               and other.pij_triangle == self.pij_triangle

    def get_full_item_set(self):
        return set(self.item_to_rank)

    def get_rank_in_reference(self, item):
        return self.item_to_rank[item]

    def get_prob_i_j(self, rank_i, j):
        return self.pij_triangle[rank_i][j]

    def get_range_of_possible_ranks(self, item=None):
        return 0, self.num_items - 1

    def sample_a_ranking(self, rng: NumpyRNG = None) -> list:
        rng = rng or default_rng()

        ranking = []
        insertion_range = []

        for step, item in enumerate(self.reference):
            insertion_range.append(step)
            sample_index = rng.choice(a=insertion_range, p=self.pij_triangle[step], shuffle=False)

            ranking.insert(sample_index, item)

        return ranking

    def sample_a_permutation(self, rng: NumpyRNG = None) -> list:
        return self.sample_a_ranking(rng)

    def calculate_prob_of_permutation(self, permutation: Sequence):
        assert len(permutation) == self.num_items

        ranking = list(permutation)
        prob = 1
        for step in range(len(self.reference) - 1, 0, -1):
            item = self.reference[step]
            rank = ranking.index(item)
            prob *= self.pij_triangle[step][rank]
            del ranking[rank]

        return prob

    @classmethod
    def generate_a_random_instance(cls, m: int, rng: NumpyRNG = None):
        rng = rng or default_rng()

        reference = tuple(range(m))
        pij_matrix: List[Tuple[float]] = [tuple(normalize_weights(rng.random(size=i))) for i in range(1, m + 1)]

        return cls(reference, tuple(pij_matrix))


def show_scalability(m=10):
    from time import time

    t1 = time()
    rim = RepeatedInsertionModel.generate_a_random_instance(m=m)
    t2 = time()

    print(f'{m=}, t_RIM_generation = {t2 - t1}, len(RIM) = {len(repr(rim))}')

    with open(f'rim_output_of_m_{m}.txt', 'w') as handle:
        handle.write(repr(rim))


def main():
    for m in [10, 100, 200, 500, 1000]:
        show_scalability(m)


if __name__ == '__main__':
    main()
