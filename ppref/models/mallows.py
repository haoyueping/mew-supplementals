from copy import deepcopy
from math import factorial, prod
from typing import List, Tuple, Sequence

from numpy.random import Generator as NumpyRNG, default_rng

from ppref.helper import calculate_kendall_tau_distance, normalize_weights
from ppref.models.rim import RepeatedInsertionModel


class Mallows(RepeatedInsertionModel):

    def __init__(self, reference: tuple, phi: float):
        pij_matrix = self.calculate_pij_matrix(len(reference), phi)
        normalization_constant = self.calculate_normalization_constant(len(reference), phi)

        super().__init__(reference, pij_matrix)
        self.phi = phi
        self.normalization_constant = normalization_constant

        if phi == 1:
            self.uniform_ranking_probability = 1 / normalization_constant

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        class_name = self.__class__.__name__
        return f'{class_name}(reference={self.reference}, phi={self.phi})'

    def calculate_kendall_tau_distance(self, permutation: Sequence) -> int:
        return calculate_kendall_tau_distance(self.reference, permutation)

    def calculate_prob_given_distance(self, distance):
        if self.phi == 1:
            return self.uniform_ranking_probability

        return (self.phi ** distance) / self.normalization_constant

    def calculate_prob_of_permutation(self, permutation: Sequence):
        assert len(permutation) == self.num_items

        if self.phi == 1:
            return self.uniform_ranking_probability

        dist = self.calculate_kendall_tau_distance(permutation)
        return self.calculate_prob_given_distance(dist)

    def get_new_mallows_by_changing_reference(self, new_reference: tuple):
        mallows = deepcopy(self)
        mallows.reference = new_reference
        mallows.item_to_rank = {item: rank for rank, item in enumerate(new_reference)}
        return mallows

    @staticmethod
    def calculate_pij_matrix(num_items: int, phi: float) -> Tuple[Tuple[float]]:
        pij: List[Tuple[float]] = []
        for i in range(num_items):
            pi = [phi ** (i - j) for j in range(i + 1)]
            pi = normalize_weights(pi)
            pij.append(tuple(pi))

        return tuple(pij)

    @staticmethod
    def calculate_normalization_constant(num_items: int, phi: float) -> float:
        if phi == 1:
            return factorial(num_items)
        else:
            ps = [(1 - phi ** i) / (1 - phi) for i in range(1, num_items + 1)]
            return prod(ps)

    @classmethod
    def generate_a_random_instance(cls, m: int, rng: NumpyRNG = None):
        rng = rng or default_rng()

        reference = tuple(range(m))
        phi = rng.uniform(0.01, 0.99)

        return cls(reference, phi)
