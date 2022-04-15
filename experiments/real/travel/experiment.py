import pandas as pd

from experiments.helper import get_dir, get_rule_vector
from ppref.preferences.special import PartitionedPreferences
from ppref.profile_solvers.solver_by_voter_pruning import sequential_solver_of_voter_pruning


def calculate_travel():
    sep = '\t'

    df = pd.read_csv(get_dir(__file__) / 'travel_input.tsv', sep=sep)
    profile: list[PartitionedPreferences] = [eval(pp) for pp in df['pp']]

    df_out = pd.DataFrame(columns=['rule', 'winners', 't_sec'])
    for rule in ['Plurality', '2-approval', 'Borda']:
        rule_vector = get_rule_vector(rule, profile[0].m)
        answer = sequential_solver_of_voter_pruning(profile.copy(), rule_vector)

        df_out.loc[df_out.shape[0]] = [rule, answer['winners'], answer['t_total_sec']]

    df_out.to_csv(get_dir(__file__) / 'travel_output.tsv', index=False, sep=sep)


if __name__ == '__main__':
    calculate_travel()
