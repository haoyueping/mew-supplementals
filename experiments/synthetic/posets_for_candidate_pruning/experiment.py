import pandas as pd

from experiments.helper import get_dir, get_rule_vector
from experiments.synthetic.posets.experiment import generate_profile_df
from ppref.preferences.poset import Poset
from ppref.profile_solvers.solver_by_voter_pruning import sequential_baseline_solver, sequential_solver_of_voter_pruning


def generate_combs():
    m = 10
    n = 10_000
    phi = 0.5
    batch_list = list(range(10))

    combs = set()
    for pmax in [0.1, 0.5, 0.9]:
        for batch in batch_list:
            combs.add((m, n, phi, pmax, batch))

    return combs


def generate_profiles():
    for (m, n, phi, pmax, batch) in generate_combs():
        filename = f'{m}_candidates_{n}_voters_{phi=:.1f}_pmax={pmax:.1f}_batch_{batch}.tsv'
        fullpath = get_dir(__file__) / f'profiles/{filename}'
        fullpath.parent.mkdir(parents=True, exist_ok=True)

        if not fullpath.exists():
            print(f'[INFO] Generating {filename}')
            df = generate_profile_df(m, n, phi, pmax, batch)
            df.to_csv(fullpath, index=False, sep='\t')

    print('[INFO] Profile generation... Done.')


def run_experiment_of_no_grouping_and_no_pruning_sequential_baseline():
    """
    To test the effectiveness of pruning strategy.

    Set grouping=False, invoke the sequential baseline solver.
    """
    sep = '\t'
    pruning = False
    grouping = False

    rule_list = ['Plurality', '2-approval', 'Borda']

    out_file = get_dir(__file__) / 'experiment_output_of_no_grouping_and_no_pruning_sequential_baseline.tsv'
    open_mode = 'a' if out_file.exists() else 'w'
    with open(out_file, open_mode, buffering=1) as out:
        if open_mode == 'w':
            cols = ['m', 'n', 'phi', 'pmax', 'batch', 'rule', 'pruning', 'grouping', 'winners', 'winner_score',
                    't_baseline_sec']
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
                    answer = sequential_baseline_solver(profile, get_rule_vector(rule, m), grouping=grouping)

                    record = [m, n, f'{phi:.1f}', f'{pmax:.1f}', batch, rule, pruning, grouping,
                              answer['winners'], answer['winner_score'], answer['t_sec']]
                    record = [str(i) for i in record]

                    out.write(sep.join(record) + '\n')

    print('[INFO] run_experiment_of_no_grouping_and_no_pruning_sequential_baseline()... Done.')


def run_experiment_of_no_grouping_but_with_pruning():
    """
    To test the effectiveness of pruning strategy.

    Set grouping=False, invoke the sequential solver with candidate pruning.
    """
    sep = '\t'
    pruning = True
    grouping = False

    rule_list = ['Plurality', '2-approval', 'Borda']

    out_file = get_dir(__file__) / 'experiment_output_of_no_grouping_but_with_pruning.tsv'
    open_mode = 'a' if out_file.exists() else 'w'
    with open(out_file, open_mode, buffering=1) as out:
        if open_mode == 'w':
            cols = ['m', 'n', 'phi', 'pmax', 'batch', 'rule', 'pruning', 'grouping', 'winners', 't_total_sec']
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
                    answer = sequential_solver_of_voter_pruning(profile, get_rule_vector(rule, m), grouping=grouping)

                    record = [m, n, f'{phi:.1f}', f'{pmax:.1f}', batch, rule, pruning, grouping,
                              answer['winners'], answer['t_total_sec']]
                    record = [str(i) for i in record]

                    out.write(sep.join(record) + '\n')

    print('[INFO] run_experiment_of_no_grouping_but_with_pruning()... Done.')


if __name__ == '__main__':
    generate_profiles()
    run_experiment_of_no_grouping_and_no_pruning_sequential_baseline()
    run_experiment_of_no_grouping_but_with_pruning()
