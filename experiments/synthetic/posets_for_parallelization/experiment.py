from multiprocessing import Pool

import pandas as pd

from experiments.helper import get_dir, get_rule_vector
from experiments.synthetic.posets.experiment import generate_profile_df
from ppref.preferences.poset import Poset
from ppref.profile_solvers.solver_by_voter_pruning import parallel_baseline_solver


def generate_combs():
    m = 10
    n = 1_000_000
    phi = 0.5
    pmax = 0.1
    batch_list = list(range(10))

    combs = []
    for batch in batch_list:
        combs.append((m, n, phi, pmax, batch))

    return sorted(combs, key=lambda x: x[1])


def worker(m, n, phi, pmax, batch):
    filename = f'{m}_candidates_{n}_voters_{phi=:.1f}_pmax={pmax:.1f}_batch_{batch}.tsv'
    fullpath = get_dir(__file__) / f'profiles/{filename}'
    fullpath.parent.mkdir(parents=True, exist_ok=True)

    if not fullpath.exists():
        print(f'[INFO] Generating {filename}')
        df = generate_profile_df(m, n, phi, pmax, batch)
        df.to_csv(fullpath, index=False, sep='\t')


def generate_profiles():
    with Pool(processes=10) as pool:
        pool.starmap(worker, generate_combs())

    print('Done.')


def run_experiment():
    """Run parallel baseline solver."""

    sep = '\t'
    grouping = True

    rule_list = ['Plurality', '2-approval', 'Borda']

    out_file = get_dir(__file__) / 'experiment_output.tsv'
    open_mode = 'a' if out_file.exists() else 'w'
    with open(out_file, open_mode, buffering=1) as out:
        if open_mode == 'w':
            cols = ['m', 'n', 'phi', 'pmax', 'batch', 'rule', 'threads', 'grouping', 'winners', 't_sec']
            out.write(sep.join(cols) + '\n')
        else:
            df_existing = pd.read_csv(out_file, sep=sep)

        for (m, n, phi, pmax, batch) in generate_combs():
            for rule in rule_list:
                for threads in [1, 5, 10]:

                    condition = f'(m == {m}) and (n == {n}) and (phi == {phi}) and (pmax == {pmax}) and ' \
                                f'(batch == {batch}) and (rule == "{rule}") and (threads == {threads}) and ' \
                                f'(grouping == {grouping})'

                    is_existing = (open_mode == 'a') and (not df_existing.query(condition).empty)

                    if not is_existing:
                        print(f'- Executing {condition}')

                        filename = f'{m}_candidates_{n}_voters_{phi=:.1f}_pmax={pmax:.1f}_batch_{batch}.tsv'
                        fullpath = get_dir(__file__) / f'profiles/{filename}'
                        df_in = pd.read_csv(fullpath, sep=sep)

                        profile: list[Poset] = [eval(po) for po in df_in['poset']]
                        answer = parallel_baseline_solver(profile, get_rule_vector(rule, m), threads=threads,
                                                          grouping=grouping)

                        record = [m, n, f'{phi:.1f}', f'{pmax:.1f}', batch, rule, threads, grouping, answer['winners'],
                                  answer['t_sec']]
                        record = [str(i) for i in record]

                        out.write(sep.join(record) + '\n')

    print('Done.')


if __name__ == '__main__':
    generate_profiles()
    run_experiment()