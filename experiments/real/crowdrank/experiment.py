import pandas as pd

from experiments.helper import get_dir, get_rule_vector
from ppref.preferences.special import PartialChain
from ppref.profile_solvers.solver_by_voter_pruning import sequential_solver_of_voter_pruning


def calculate_crowdrank(is_test=False):
    df = pd.read_csv(get_dir(__file__) / 'crowdrank_input.tsv', sep='\t')

    df_out = pd.DataFrame(columns=['hit', 'm', 'n', 'rule', 'winners', 't_sec'])
    for hit in df['hitid'].unique():
        df_sub = df.query(f'hitid == {hit}')
        profile: list[PartialChain] = [eval(pc) for pc in df_sub['pc']]
        m = len(profile[0].get_full_item_set())
        n = len(profile)

        for rule in ['Plurality', '2-approval', 'Borda']:
            rule_vector = get_rule_vector(rule, m)
            answer = sequential_solver_of_voter_pruning(profile.copy(), rule_vector)

            df_out.loc[df_out.shape[0]] = [hit, m, n, rule, answer['winners'], answer['t_total_sec']]

        if is_test:
            break

    df_out.to_csv(get_dir(__file__) / 'crowdrank_output.tsv', index=False, sep='\t')


if __name__ == '__main__':
    calculate_crowdrank(is_test=False)
