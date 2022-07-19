import pandas as pd

from experiments.helper import get_dir, get_rule_vector
from experiments.synthetic.posets_for_parallelization.experiment import generate_profiles, generate_combs
from ppref.preferences.poset import Poset
from ppref.profile_solvers.solver_by_voter_pruning import parallel_baseline_solver


def run_experiment():
    """Run parallel baseline solver."""

    sep = '\t'
    grouping = True

    rule_list = ['Plurality', '2-approval', 'Borda']

    out_file = get_dir(__file__) / 'experiment_cm_output.tsv'
    open_mode = 'a' if out_file.exists() else 'w'
    with open(out_file, open_mode, buffering=1) as out:
        if open_mode == 'w':
            cols = ['m', 'n', 'phi', 'pmax', 'batch', 'rule', 'threads', 'grouping', 'winners', 't_sec']
            out.write(sep.join(cols) + '\n')
        else:
            df_existing = pd.read_csv(out_file, sep=sep)

        for (m, n, phi, pmax, batch) in generate_combs():
            for rule in rule_list:
                for threads in [15, 20, 25, 30, 35, 40, 45, 48]:

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