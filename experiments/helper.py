from pathlib import Path

from ppref.preferences.poset import Poset


def get_project_root_path() -> Path:
    return get_path_to_experiments().parent


def get_path_to_experiments() -> Path:
    return Path(__file__).parent


def get_dir(path='.') -> Path:
    path = Path(path)

    if path.is_dir():
        return path
    else:
        return path.parent


def get_path_to_rim_benchmark(m: int):
    experiments = get_path_to_experiments()
    return experiments / f'synthetic_data/benchmark_RIM/benchmark_RIM_{m=}.tsv'


def get_path_to_rim_benchmark_new():
    experiments = get_path_to_experiments()
    return experiments / f'synthetic_data/benchmark_RIM.tsv'


def get_path_to_rsm_benchmark(m: int):
    experiments = get_path_to_experiments()
    return experiments / f'synthetic_data/benchmark_rRSM/benchmark_rRSM_{m=}.tsv'


def get_path_to_rsm_benchmark_new():
    experiments = get_path_to_experiments()
    return experiments / f'synthetic_data/benchmark_rRSM.tsv'


def get_path_to_mallows_benchmark(m: int):
    experiments = get_path_to_experiments()
    return experiments / f'synthetic_data/benchmark_Mallows/benchmark_Mallows_{m=}.tsv'


def get_path_to_pp_benchmark():
    experiments = get_path_to_experiments()
    return experiments / 'synthetic_data/benchmark_PartitionedPreferences.tsv'


def get_path_to_pp_benchmark_varying_m():
    experiments = get_path_to_experiments()
    return experiments / 'synthetic_data/benchmark_PartitionedPreferences_varying_m.tsv'


def get_path_to_pp_benchmark_varying_k():
    experiments = get_path_to_experiments()
    return experiments / 'synthetic_data/benchmark_PartitionedPreferences_varying_k.tsv'


def get_path_to_tr_benchmark():
    experiments = get_path_to_experiments()
    return experiments / 'synthetic_data/benchmark_TruncatedRankings.tsv'


def get_path_to_poset_benchmark():
    experiments = get_path_to_experiments()
    return experiments / 'synthetic_data/benchmark_Posets.tsv'


def get_path_to_poset_profile(num_candidates, num_voters, phi, rsm_pmax, batch):
    filename = f'{num_candidates}_candidates_{num_voters}_voters_{phi=:.1f}_pmax={rsm_pmax:.1f}_batch_{batch}.tsv'
    return get_path_to_experiments() / f'synthetic_data/partial_voting_profiles/{filename}'


def get_random_poset_by_rsm(random_permutation, rsm_p_list, rng):
    m = len(random_permutation)
    po = {}
    for i in range(m - 1):
        e_i = random_permutation[i]
        p_i = rsm_p_list[i]
        for j in range(i + 1, m):
            if rng.random() < p_i:
                e_j = random_permutation[j]
                po.setdefault(e_i, set()).add(e_j)

    if po:
        return Poset(parent_to_children=po, item_set=set(range(m)))
    else:
        return get_random_poset_by_rsm(random_permutation, rsm_p_list, rng)


def get_rule_vector(name: str, m: int):
    if name == 'Plurality':
        return tuple([1] + [0 for _ in range(m - 1)])
    elif name == '2-approval':
        return tuple([1, 1] + [0 for _ in range(m - 2)])
    elif name == 'Borda':
        return tuple(range(m - 1, -1, -1))
    else:
        rules = {'Plurality', '2-approval', 'Borda'}
        raise f'The input voting rule {name} is not covered by the supported rules of {rules}.'
