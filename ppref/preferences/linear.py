class Linear(object):
    """Equivalent to a partial chain without specifying the missing items."""

    def __init__(self, ranking: tuple):
        self.r = ranking

        self.item_set = set(ranking)
        self.item_to_rank = {item: rank for rank, item in enumerate(ranking)}
        self.item_to_ancestors = {item: set(ranking[:rank]) for rank, item in enumerate(ranking)}
        self.item_to_descendants = {item: set(ranking[rank + 1:]) for rank, item in enumerate(ranking)}

    def __len__(self):
        return len(self.r)

    def __repr__(self):
        return f'Linear(ranking={self.r})'

    def __eq__(self, other):
        return isinstance(other, Linear) and (self.r == other.r)

    def __hash__(self):
        return hash(tuple(self.r))

    def to_list(self):
        return list(self.r)

    def has_item(self, item):
        return item in self.item_set

    def get_items(self):
        return self.item_set

    def get_rank_of(self, item) -> int:
        return self.item_to_rank[item]

    def has_parent(self, item):
        return item != self.r[0]

    def get_parent_of_(self, item):
        rank = self.get_rank_of(item)
        assert rank >= 1
        return self.r[rank - 1]

    def is_compatible_with(self, ranking_by_item_to_rank):
        projected = sorted(self.item_set, key=ranking_by_item_to_rank.get)
        return tuple(projected) == self.r

    def get_ancestors(self, item):
        return self.item_to_ancestors.get(item, set())

    def get_descendants(self, item):
        return self.item_to_descendants.get(item, set())
