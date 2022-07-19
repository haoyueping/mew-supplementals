import pandas as pd

from experiments.helper import get_dir, get_rule_vector
from experiments.synthetic.posets.experiment import generate_profile_df
from ppref.preferences.poset import Poset
from ppref.profile_solvers.solver_by_voter_pruning import sequential_solver_of_voter_pruning
from ppref.profile_solvers.solver_for_mpw import sequential_mpw_solver


def generate_combs():
    m_list = list(range(3, 8))
    n_list = list(range(1, 8))
    phi = 0.5
    pmax = 0.1
    batch_list = list(range(10))

    combs = set()

    # varying m
    n = 5
    for m in m_list:
        for batch in batch_list:
            combs.add((m, n, phi, pmax, batch))

    # varying n
    m = 5
    for n in n_list:
        for batch in batch_list:
            combs.add((m, n, phi, pmax, batch))

    return sorted(combs, key=lambda x: sum(x[:2]))


def generate_profiles():
    for (m, n, phi, pmax, batch) in generate_combs():
        filename = f'{m}_candidates_{n}_voters_{phi=:.1f}_pmax={pmax:.1f}_batch_{batch}.tsv'
        fullpath = get_dir(__file__) / f'profiles/{filename}'
        fullpath.parent.mkdir(parents=True, exist_ok=True)

        if not fullpath.exists():
            print(f'[INFO] Generating {filename}')
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
            cols = ['m', 'n', 'phi', 'pmax', 'batch', 'rule', 'mpw', 't_mpw_s', 'mew', 't_mew_s']
            out.write(sep.join(cols) + '\n')
        else:
            df_existing = pd.read_csv(out_file, sep=sep)

        for (m, n, phi, pmax, batch) in generate_combs():
            for rule in rule_list:

                condition = f'(m == {m}) and (n == {n}) and (phi == {phi}) and (pmax == {pmax}) and (batch == {batch}) ' \
                            f'and (rule == "{rule}")'

                is_existing = (open_mode == 'a') and (not df_existing.query(condition).empty)

                if not is_existing:
                    print(f'- Executing {condition}')

                    filename = f'{m}_candidates_{n}_voters_{phi=:.1f}_pmax={pmax:.1f}_batch_{batch}.tsv'
                    fullpath = get_dir(__file__) / f'profiles/{filename}'
                    df_in = pd.read_csv(fullpath, sep=sep)

                    profile: list[Poset] = [eval(po) for po in df_in['poset']]
                    ans_mpw = sequential_mpw_solver(profile, get_rule_vector(rule, m), verbose=True)
                    ans_mew = sequential_solver_of_voter_pruning(profile, get_rule_vector(rule, m))

                    record = [m, n, f'{phi:.1f}', f'{pmax:.1f}', batch, rule, ans_mpw['winners'], ans_mpw['t_total_sec'],
                              ans_mew['winners'], ans_mew['t_total_sec']]
                    record = [str(i) for i in record]

                    out.write(sep.join(record) + '\n')

    print('Done.')


if __name__ == '__main__':
    generate_profiles()
    run_experiment()
