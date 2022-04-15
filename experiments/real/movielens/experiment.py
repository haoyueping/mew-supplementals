import pandas as pd

from experiments.helper import get_dir, get_rule_vector
from ppref.preferences.special import PartitionedWithMissing
from ppref.profile_solvers.solver_by_voter_pruning import sequential_solver_of_voter_pruning


def calculate_movielens():
    sep = '\t'

    df = pd.read_csv(get_dir(__file__) / 'movielens_input.tsv', sep=sep)
    profile: list[PartitionedWithMissing] = [eval(ppwm) for ppwm in df['ppwm']]

    df_out = pd.DataFrame(columns=['rule', 'winners', 't_sec'])
    for rule in ['Plurality', '2-approval', 'Borda']:
        rule_vector = get_rule_vector(rule, profile[0].m)
        answer = sequential_solver_of_voter_pruning(profile.copy(), rule_vector)

        df_out.loc[df_out.shape[0]] = [rule, answer['winners'], answer['t_total_sec']]

    df_out.to_csv(get_dir(__file__) / 'movielens_output.tsv', index=False, sep=sep)


if __name__ == '__main__':
    calculate_movielens()
