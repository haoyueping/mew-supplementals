from numpy.random import Generator as NumpyRNG, default_rng

from ppref.models.mallows import Mallows
from ppref.models.rim import RepeatedInsertionModel
from ppref.preferences.poset import Poset
from ppref.preferences.special import PartitionedPreferences, TruncatedRanking


class RimWithTR(object):
    """RIM combined with a Truncated Ranking (TR)."""

    def __init__(self, rim: RepeatedInsertionModel, truncated_ranking: TruncatedRanking):
        self.rim: RepeatedInsertionModel = rim
        self.truncated_ranking: TruncatedRanking = truncated_ranking

    def __repr__(self):
        return f'RimWithTR(rim={self.rim}, truncated_ranking={repr(self.truncated_ranking)})'

    def get_full_item_set(self):
        return self.truncated_ranking.get_full_item_set()

    def get_rim(self):
        return self.rim

    def get_truncated_ranking(self):
        return self.truncated_ranking

    def get_range_of_possible_ranks(self, item):
        return self.truncated_ranking.get_range_of_possible_ranks(item)

    @classmethod
    def generate_a_random_instance(cls, num_items, top_k: int = None, bottom_k: int = None, rng: NumpyRNG = None):
        rng = rng or default_rng()
        top_k = top_k or max(num_items // 3, 1)
        bottom_k = bottom_k or max(num_items // 3, 1)

        rim = RepeatedInsertionModel.generate_a_random_instance(m=num_items, rng=rng)
        tr = TruncatedRanking.generate_a_random_instance(top_k=top_k, bottom_k=bottom_k, items=set(rim.reference), rng=rng)

        return cls(rim=rim, truncated_ranking=tr)


class MallowsWithPP(object):
    """Mallows combined with Partitioned Preferences (PP)."""

    def __init__(self, mallows: Mallows, partitioned_preferences: PartitionedPreferences):
        self.mallows = mallows
        self.partitioned_preferences = partitioned_preferences

    def __repr__(self):
        class_name = self.__class__.__name__
        mallows_str = repr(self.mallows)
        partitioned_str = repr(self.partitioned_preferences)
        return f'{class_name}(mallows={mallows_str}, partitioned_preferences={partitioned_str})'

    def get_mallows(self):
        return self.mallows

    def get_partitioned_preferences(self):
        return self.partitioned_preferences

    def get_full_item_set(self):
        return self.partitioned_preferences.get_full_item_set()

    def get_range_of_possible_ranks(self, item):
        return self.partitioned_preferences.get_range_of_possible_ranks(item)

    @classmethod
    def generate_a_random_instance(cls, num_items, num_buckets, rng: NumpyRNG = None):
        rng = rng or default_rng()

        mallows = Mallows.generate_a_random_instance(m=num_items, rng=rng)
        pp = PartitionedPreferences.generate_a_random_instance(num_items=num_items, num_buckets=num_buckets, rng=rng)

        return cls(mallows=mallows, partitioned_preferences=pp)


class MallowsWithPoset(object):
    """Mallows combined with a poset."""

    def __init__(self, mallows: Mallows, poset: Poset):
        self.mallows = mallows
        self.poset = poset

    def __repr__(self):
        class_name = self.__class__.__name__
        mallows_str = repr(self.mallows)
        poset_str = repr(self.poset)
        return f'{class_name}(mallows={mallows_str}, poset={poset_str})'

    def get_mallows(self):
        return self.mallows

    def get_poset(self):
        return self.poset

    def get_full_item_set(self):
        return self.poset.get_full_item_set()

    def get_range_of_possible_ranks(self, item):
        return self.poset.get_range_of_possible_ranks(item)

    @classmethod
    def generate_a_random_instance(cls, m=20, num_items=5, edge_prob=0.3, rng: NumpyRNG = None):
        rng = rng or default_rng()

        mallows = Mallows.generate_a_random_instance(m=m, rng=rng)
        poset = Poset.generate_a_random_instance(m=m, cardinality=num_items, edge_prob=edge_prob, rng=rng)

        return cls(mallows=mallows, poset=poset)
