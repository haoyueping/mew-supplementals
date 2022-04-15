from itertools import permutations, product, combinations
from math import comb, factorial
from typing import Dict, List, Any, Sequence

from numpy.random import Generator as NumpyRNG
from numpy.random import default_rng

from ppref.preferences.poset import Poset


class PartitionedPreferences(object):
    def __init__(self, bucket_order: list = None, item_to_bucket: dict = None):
        self.bucket_order = bucket_order.copy() or []
        self.item_to_bucket = item_to_bucket.copy() or {}

        self.m = len(item_to_bucket)
        self.bucket_to_rank = {bucket: rank for rank, bucket in enumerate(bucket_order)}
        self.bucket_to_items = {bucket: set() for bucket in bucket_order}
        for item, bucket in item_to_bucket.items():
            self.bucket_to_items[bucket].add(item)

        self.bucket_to_size = {bucket: len(items) for bucket, items in self.bucket_to_items.items()}

    def __repr__(self):
        return f'PartitionedPreferences(bucket_order={self.bucket_order}, item_to_bucket={self.item_to_bucket})'

    def __eq__(self, other):
        return isinstance(other, PartitionedPreferences) and \
               (self.bucket_order == other.bucket_order) and \
               (self.item_to_bucket == other.item_to_bucket)

    def __hash__(self):
        return hash((tuple(self.bucket_order), tuple(sorted(self.item_to_bucket.items()))))

    def get_full_item_set(self):
        return set(self.item_to_bucket)

    def get_range_of_possible_ranks(self, item):
        bucket = self.item_to_bucket[item]
        bucket_rank = self.bucket_to_rank[bucket]
        left = sum([self.bucket_to_size[self.bucket_order[i]] for i in range(bucket_rank)])
        right_size = sum([self.bucket_to_size[self.bucket_order[i]] for i in range(bucket_rank + 1, len(self.bucket_order))])
        right = self.m - 1 - right_size
        return left, right

    def iterate_linear_extensions(self):
        perm_iters = [permutations(self.bucket_to_items[bucket]) for bucket in self.bucket_order]
        for res in product(*perm_iters):
            yield sum(res, ())

    def has_such_linear_extension(self, permutation):
        item_to_rank = {item: rank for rank, item in enumerate(permutation)}

        for i in range(len(self.bucket_order) - 1):
            bucket_i = self.bucket_order[i]
            bucket_j = self.bucket_order[i + 1]

            for item_i in self.bucket_to_items[bucket_i]:
                rank_i = item_to_rank[item_i]
                for item_j in self.bucket_to_items[bucket_j]:
                    rank_j = item_to_rank[item_j]

                    if rank_i > rank_j:
                        return False

        return True

    def calculate_rank_probabilities_of_item(self, item):
        bucket = self.item_to_bucket[item]
        b_rank = self.bucket_to_rank[bucket]
        b_size = len(self.bucket_to_items[bucket])

        size_left = sum([len(self.bucket_to_items[b]) for b in self.bucket_order[:b_rank]])
        size_right = self.m - size_left - b_size

        return [0. for _ in range(size_left)] + [1 / b_size for _ in range(b_size)] + [0. for _ in range(size_right)]

    def calculate_rank_probabilities_of_all_items(self):
        b_sizes = [len(self.bucket_to_items[b]) for b in self.bucket_order]

        item_to_probs = dict()
        for b_i, bucket in enumerate(self.bucket_order):
            b_size = b_sizes[b_i]
            size_left = sum(b_sizes[:b_i])
            size_right = self.m - size_left - b_size
            probs = [0. for _ in range(size_left)] + [1 / b_size for _ in range(b_size)] + [0. for _ in range(size_right)]
            for item in self.bucket_to_items[bucket]:
                item_to_probs[item] = probs.copy()

        return item_to_probs

    @classmethod
    def generate_a_random_instance(cls, num_items: int, num_buckets: int, rng: NumpyRNG = None):
        assert num_items >= num_buckets
        rng = rng or default_rng()

        item_order = rng.permutation(num_items)

        return cls.generate_instance_from_ranking(num_buckets, item_order, rng)

    @classmethod
    def generate_instance_from_ranking(cls, num_buckets: int, item_order: Sequence, rng: NumpyRNG = None):
        num_items = len(item_order)

        assert num_items >= num_buckets
        rng = rng or default_rng()

        bucket_order = [f'B{i}' for i in range(num_buckets)]

        item_to_bucket = {}

        for i in range(num_buckets):
            item_to_bucket[item_order[i]] = bucket_order[i]

        for i in range(num_buckets, num_items):
            item_to_bucket[item_order[i]] = rng.choice(a=bucket_order)

        return PartitionedPreferences(bucket_order, item_to_bucket)


class PartialChain(object):
    def __init__(self, chain: list, item_set: set):
        self.chain = chain.copy()
        self.item_set = item_set.copy()

        self.missing_items = item_set - set(chain)
        self.item_to_chain_rank = {item: rank for rank, item in enumerate(chain)}

    def __repr__(self):
        return f'PartialChain(chain={self.chain}, item_set={self.item_set})'

    def __eq__(self, other):
        return isinstance(other, PartialChain) and (self.chain == other.chain) and (self.item_set == other.item_set)

    def __hash__(self):
        return hash((tuple(self.chain), tuple(sorted(self.missing_items))))

    def get_full_item_set(self):
        return self.item_set

    def get_range_of_possible_ranks(self, item):
        if item in self.missing_items:
            return 0, len(self.item_set) - 1

        rank = self.item_to_chain_rank[item]
        right_size = len(self.chain) - 1 - rank
        return rank, len(self.item_set) - 1 - right_size

    def iterate_linear_extensions(self):
        chain_size = len(self.chain)
        full_size = len(self.item_set)

        if chain_size == full_size:
            return self.chain

        ranks = list(range(full_size))
        for chain_item_ranks in combinations(ranks, chain_size):
            chain_item_ranks = set(chain_item_ranks)
            for missing_item_permutation in permutations(self.missing_items):
                pointer_chain = 0
                pointer_missing = 0
                ranking = []
                pointer = 0
                while pointer < full_size:
                    if pointer in chain_item_ranks:
                        ranking.append(self.chain[pointer_chain])
                        pointer_chain += 1
                    else:
                        ranking.append(missing_item_permutation[pointer_missing])
                        pointer_missing += 1

                    pointer += 1

                yield ranking

    def has_such_linear_extension(self, permutation):
        item_to_rank = {item: rank for rank, item in enumerate(permutation)}

        for i in range(len(self.chain) - 1):
            left_item = self.chain[i]
            right_item = self.chain[i + 1]

            if item_to_rank[left_item] > item_to_rank[right_item]:
                return False

        return True

    def calculate_rank2prob_for_item(self, item):
        m = len(self.item_set)

        if item in self.missing_items:
            return [1 / m for _ in range(m)]

        item_rank = self.item_to_chain_rank[item]
        k = len(self.chain)
        k_l = item_rank
        k_r = k - 1 - item_rank
        probs = []
        for j in range(m):
            if j < k_l or m - 1 - j < k_r:
                probs.append(0.0)
            else:
                probs.append(comb(j, k_l) * comb(m - 1 - j, k_r))

        prob_sum = sum(probs)
        return [prob / prob_sum for prob in probs]

    def calculate_rank2prob_for_all_items(self):
        return {item: self.calculate_rank2prob_for_item(item) for item in self.item_set}

    @classmethod
    def generate_a_random_instance(cls, chain_size: int, items: set, rng: NumpyRNG = None):
        rng = rng or default_rng()
        ranking = rng.choice(a=list(items), size=chain_size, replace=False)
        return PartialChain(list(ranking), items)


class PartitionedWithMissing(object):
    def __init__(self, bucket_order: list, item_to_bucket: dict, item_set: set):
        self.bucket_order = bucket_order.copy()
        self.item_to_bucket = item_to_bucket.copy()
        self.item_set = item_set.copy()

        self.m = len(item_set)
        self.missing_items = item_set - set(item_to_bucket)
        self.bucket_to_rank = {bucket: rank for rank, bucket in enumerate(bucket_order)}
        self.bucket_to_items = {bucket: set() for bucket in bucket_order}
        for item, bucket in item_to_bucket.items():
            self.bucket_to_items[bucket].add(item)

        self.bucket_to_size = {bucket: len(items) for bucket, items in self.bucket_to_items.items()}

    def __repr__(self):
        class_name = self.__class__.__name__
        return f'{class_name}(bucket_order={self.bucket_order}, item_to_bucket={self.item_to_bucket}, ' \
               f'item_set={self.item_set})'

    def __eq__(self, other):
        return isinstance(other, PartitionedPreferences) and \
               (self.bucket_order == other.bucket_order) and \
               (self.item_to_bucket == other.item_to_bucket)

    def __hash__(self):
        return hash((tuple(self.bucket_order), tuple(sorted(self.item_to_bucket.items())), tuple(sorted(self.item_set))))

    def get_full_item_set(self):
        return self.item_set

    def get_range_of_possible_ranks(self, item):
        if item in self.missing_items:
            return 0, self.m - 1

        bucket = self.item_to_bucket[item]
        bucket_rank = self.bucket_to_rank[bucket]
        left = sum([self.bucket_to_size[self.bucket_order[i]] for i in range(bucket_rank)])
        right_size = sum([self.bucket_to_size[self.bucket_order[i]] for i in range(bucket_rank + 1, len(self.bucket_order))])
        right = self.m - 1 - right_size
        return left, right

    def has_such_linear_extension(self, permutation):
        item_to_rank = {item: rank for rank, item in enumerate(permutation)}

        for i in range(len(self.bucket_order) - 1):
            bucket_i = self.bucket_order[i]
            bucket_j = self.bucket_order[i + 1]

            for item_i in self.bucket_to_items[bucket_i]:
                rank_i = item_to_rank[item_i]
                for item_j in self.bucket_to_items[bucket_j]:
                    rank_j = item_to_rank[item_j]

                    if rank_i > rank_j:
                        return False

        return True

    def calculate_rank2prob_for_item(self, item):
        if item in self.missing_items:
            return [1 / self.m for _ in range(self.m)]
        else:
            bucket = self.item_to_bucket[item]
            b_rank = self.bucket_to_rank[bucket]
            b_size = len(self.bucket_to_items[bucket])

            # num_items in left buckets
            k_l = sum([len(self.bucket_to_items[b]) for b in self.bucket_order[:b_rank]])
            # num_items in right buckets
            k_r = self.m - len(self.missing_items) - k_l - b_size

            probs = []
            for j in range(self.m):
                if k_l <= j <= self.m - 1 - k_r:
                    prob = 0
                    for k_same_bucket_to_left in range(min(j - k_l, b_size - 1) + 1):
                        num_bucket_items_left = k_l + k_same_bucket_to_left
                        num_bucket_items_right = k_r + b_size - 1 - k_same_bucket_to_left
                        prob += comb(j, num_bucket_items_left) * comb(self.m - 1 - j, num_bucket_items_right)

                    probs.append(prob)
                else:
                    probs.append(0.0)

            prob_sum = sum(probs)
            return [prob / prob_sum for prob in probs]

    @classmethod
    def generate_a_random_instance(cls, num_items: int, num_buckets: int, rng: NumpyRNG = None):
        assert num_items >= num_buckets
        rng = rng or default_rng()

        item_order = rng.permutation(num_items)

        bucket_order = [f'B{i}' for i in range(num_buckets)] + ['MISSING']

        item_to_bucket = {}

        for i in range(num_buckets):
            item_to_bucket[item_order[i]] = bucket_order[i]

        for i in range(num_buckets, num_items):
            bucket = rng.choice(a=bucket_order)
            if bucket != 'MISSING':
                item_to_bucket[item_order[i]] = bucket

        return cls(bucket_order[:-1], item_to_bucket, item_set=set(item_order))


class TruncatedRanking(object):
    def __init__(self, top: list, bottom: list, item_set: set):
        self.top = top.copy()
        self.bottom = bottom.copy()
        self.item_set = item_set.copy()

        self.items_of_top = set(top)
        self.items_of_bottom = set(bottom)
        self.items_of_middle = item_set - self.items_of_top - self.items_of_bottom

        self.truncated_item_to_rank = {e: rank for rank, e in enumerate(top)}
        for rank, e in enumerate(reversed(bottom)):
            self.truncated_item_to_rank[e] = len(item_set) - 1 - rank

    def __repr__(self):
        return f'TruncatedRanking(top={self.top}, bottom={self.bottom}, item_set={self.item_set})'

    def __eq__(self, other):
        return isinstance(other, TruncatedRanking) and (self.top == other.top) and (self.bottom == other.bottom) and \
               (self.items_of_middle == other.items_of_middle)

    def __hash__(self):
        return hash((tuple(self.top), tuple(self.bottom), tuple(sorted(self.items_of_middle))))

    def get_full_item_set(self):
        return self.item_set

    def get_items_of_middle(self) -> set:
        return self.items_of_middle

    def get_rank_of_truncated_item(self, item) -> int:
        return self.truncated_item_to_rank[item]

    def get_range_of_possible_ranks(self, item):
        if item in self.items_of_middle:
            return len(self.top), len(self.item_set) - 1 - len(self.items_of_bottom)
        else:
            rank = self.truncated_item_to_rank[item]
            return rank, rank

    def iterate_linear_extensions(self):
        for perm in permutations(self.items_of_middle):
            yield self.top + list(perm) + self.bottom

    def has_such_linear_extension(self, permutation):
        if self.item_set != set(permutation):
            return False

        size_top = len(self.top)
        if size_top and self.top != permutation[:size_top]:
            return False

        size_bottom = len(self.bottom)
        if size_bottom and self.bottom != permutation[-size_bottom:]:
            return False

        return True

    def calculate_rank_probs_for_item(self, item) -> List[float]:
        probs = [0.0 for _ in self.item_set]

        if item in self.truncated_item_to_rank:
            rank = self.truncated_item_to_rank[item]
            probs[rank] = 1.0
        else:
            middle_size = len(self.items_of_middle)
            prob = 1 / middle_size
            for rank in range(len(self.top), len(self.top) + middle_size):
                probs[rank] = prob

        return probs

    def calculate_rank_probs_for_all_items(self) -> Dict[Any, List[float]]:
        item_to_probs = {item: self.calculate_rank_probs_for_item(item) for item in self.truncated_item_to_rank}

        probs = self.calculate_rank_probs_for_item(next(iter(self.items_of_middle)))
        for item in self.items_of_middle:
            item_to_probs[item] = probs.copy()

        return item_to_probs

    def to_poset(self) -> Poset:
        parent_to_children = {}

        if len(self.top) >= 2:
            for i, e_i in enumerate(self.top[:-1]):
                parent_to_children[e_i] = {self.top[i + 1]}

        if len(self.bottom) >= 2:
            for i, e_i in enumerate(self.bottom[:-1]):
                parent_to_children[e_i] = {self.bottom[i + 1]}

        if self.top:
            if self.items_of_middle:
                parent_to_children[self.top[-1]] = self.items_of_middle.copy()
            elif self.bottom:
                parent_to_children[self.top[-1]] = {self.bottom[0]}

        if self.bottom:
            if self.items_of_middle:
                for e in self.items_of_middle:
                    parent_to_children[e] = {self.bottom[0]}
            elif self.top:
                parent_to_children[self.top[-1]] = {self.bottom[0]}

        return Poset(parent_to_children=parent_to_children, item_set=self.item_set.copy())

    @classmethod
    def generate_a_random_instance(cls, top_k: int, bottom_k: int, items: set, rng: NumpyRNG = None):
        rng = rng or default_rng()

        ranking = rng.choice(a=list(items), size=top_k + bottom_k, replace=False).tolist()

        return cls.generate_instance_from_ranking(top_k, bottom_k, ranking)

    @classmethod
    def generate_instance_from_ranking(cls, top_k: int, bottom_k: int, ranking: list):
        top = ranking[:top_k]

        if bottom_k > 0:
            bottom = ranking[-bottom_k:]
        else:
            bottom = []

        return TruncatedRanking(top, bottom, set(ranking))
