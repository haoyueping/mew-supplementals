import pandas as pd
from numpy.random import default_rng

from experiments.helper import get_dir, get_rule_vector, get_random_poset_by_rsm
from ppref.models.mallows import Mallows
from ppref.preferences.combined import MallowsWithPoset
from ppref.preferences.poset import Poset
from ppref.profile_solvers.solver_by_voter_pruning import sequential_solver_of_voter_pruning


def generate_combs():
    m = 10
    n = 1000
    pmax = 0.9
    batch_list = list(range(10))

    combs = set()
    for phi in [0.1, 0.5, 0.9]:
        for batch in batch_list:
            combs.add((m, n, phi, pmax, batch))

    return combs


def generate_profile_df(m, n, phi, pmax, batch):
    rng = default_rng([m, n, int(1000 * phi), int(1000 * pmax), batch])
    df = pd.DataFrame(columns=['m', 'phi', 'pmax', 'mallows_poset'])
    reference = tuple(range(m))
    for i in range(n):
        mallows = Mallows(reference, phi)
        rsm_p_list = rng.random(size=m) * pmax
        poset: Poset = get_random_poset_by_rsm(mallows.sample_a_ranking(rng), rsm_p_list, rng)
        mp = MallowsWithPoset(mallows, poset)
        df.loc[i] = [m, phi, pmax, repr(mp)]

    return df


def generate_profiles():
    for (m, n, phi, pmax, batch) in generate_combs():
        filename = f'{m}_candidates_{n}_voters_phi_{phi:.1f}_pmax_{pmax:.1f}_batch_{batch}.tsv'
        fullpath = get_dir(__file__) / f'profiles/{filename}'
        fullpath.parent.mkdir(parents=True, exist_ok=True)

        print(f'Generating {filename}')

        df = generate_profile_df(m, n, phi, pmax, batch)
        df.to_csv(fullpath, index=False, sep='\t')

    print('Done.')


def run_experiment():
    sep = '\t'

    rule_list = ['Plurality', '2-approval', 'Borda']

    out_file = get_dir(__file__) / 'experiment_output.tsv'
    open_mode = 'a' if out_file.exists() else 'w'
    with open(out_file, open_mode, buffering=1) as out:
        if open_mode == 'w':
            cols = ['m', 'n', 'phi', 'pmax', 'batch', 'rule', 'winners', 'score_upper', 'score_lower',
                    'num_pruned_voters', 't_quick_bounds_sec', 't_pruning_sec', 't_solver_sec', 't_total_sec']
            out.write(sep.join(cols) + '\n')
        else:
            df_existing = pd.read_csv(out_file, sep=sep)

        for (m, n, phi, pmax, batch) in generate_combs():
            for rule in rule_list:

                condition = f'(m == {m}) and (n == {n}) and (phi == {phi:.1f}) and (pmax == {pmax}) and ' \
                            f'(batch == {batch}) and (rule == "{rule}")'

                is_existing = (open_mode == 'a') and (not df_existing.query(condition).empty)

                if not is_existing:
                    print(f'- Executing {condition}')

                    filename = f'{m}_candidates_{n}_voters_phi_{phi:.1f}_pmax_{pmax:.1f}_batch_{batch}.tsv'
                    fullpath = get_dir(__file__) / f'profiles/{filename}'
                    df_in = pd.read_csv(fullpath, sep=sep)

                    profile: list[MallowsWithPoset] = [eval(pp) for pp in df_in['mallows_poset']]
                    answer = sequential_solver_of_voter_pruning(profile, get_rule_vector(rule, m))

                    record = [m, n, phi, pmax, batch, rule, answer['winners'], answer['score_upper'],
                              answer['score_lower'], answer['num_pruned_voters'], answer['t_quick_bounds_sec'],
                              answer['t_pruning_sec'], answer['t_solver_sec'], answer['t_total_sec']]
                    record = [str(i) for i in record]

                    out.write(sep.join(record) + '\n')

    print('Done.')


if __name__ == '__main__':
    run_experiment()
