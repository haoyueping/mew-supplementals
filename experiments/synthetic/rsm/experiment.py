import pandas as pd

from experiments.helper import get_dir, get_rule_vector
from experiments.synthetic.mallows.experiment import generate_combs
from ppref.models.mallows import Mallows
from ppref.models.rsm import RepeatedSelectionModelRV
from ppref.profile_solvers.solver_by_voter_pruning import sequential_solver_of_voter_pruning


def run_experiment():
    sep = '\t'

    rule_list = ['Plurality', '2-approval', 'Borda']

    out_file = get_dir(__file__) / 'experiment_output.tsv'
    open_mode = 'a' if out_file.exists() else 'w'
    with open(out_file, open_mode, buffering=1) as out:
        if open_mode == 'w':
            cols = ['m', 'n', 'equiv_phi', 'batch', 'rule', 'winners', 'score_upper', 'score_lower', 'num_pruned_voters',
                    't_quick_bounds_sec', 't_pruning_sec', 't_solver_sec', 't_total_sec']
            out.write(sep.join(cols) + '\n')
        else:
            df_existing = pd.read_csv(out_file, sep=sep)

        for (m, n, phi, batch) in generate_combs():
            for rule in rule_list:

                condition = f'(m == {m}) and (n == {n}) and (equiv_phi == {phi}) and (batch == {batch}) ' \
                            f'and (rule == "{rule}")'

                is_existing = (open_mode == 'a') and (not df_existing.query(condition).empty)

                if not is_existing:
                    print(f'- Executing {condition}')

                    filename = f'{m}_candidates_{n}_voters_phi_{phi}_batch_{batch}.tsv'
                    fullpath = get_dir(__file__).parent / f'mallows/profiles/{filename}'
                    df_in = pd.read_csv(fullpath, sep=sep)

                    profile: list[RepeatedSelectionModelRV] = []
                    for r in df_in['mallows']:
                        mal: Mallows = eval(r)
                        profile.append(RepeatedSelectionModelRV.generate_instance_from_mallows(mal))

                    answer = sequential_solver_of_voter_pruning(profile, get_rule_vector(rule, m))

                    record = [m, n, phi, batch, rule, answer['winners'], answer['score_upper'],
                              answer['score_lower'], answer['num_pruned_voters'], answer['t_quick_bounds_sec'],
                              answer['t_pruning_sec'], answer['t_solver_sec'], answer['t_total_sec']]
                    record = [str(i) for i in record]

                    out.write(sep.join(record) + '\n')

    print('Done.')


if __name__ == '__main__':
    run_experiment()
