from time import time

import pandas as pd
from numpy.random import default_rng

from experiments.helper import get_random_poset_by_rsm, get_dir
from ppref.models.mallows import Mallows
from ppref.preferences.poset import Poset
from ppref.rank_estimators.rim_w_poset import calculate_rank_probs_for_item_given_rim_w_poset_by_sequential


def generate_input_data():
    m = 10
    repeat = 100
    phi = 0.5
    pmax_list = [0.1, 0.2, 0.3, 0.5, 0.9]

    rng = default_rng(0)

    mallows = Mallows(reference=tuple(range(m)), phi=phi)
    df = pd.DataFrame(columns=['m', 'phi', 'pmax', 'ith_such_poset', 'cardinality', 'poset'])

    for pmax in pmax_list:
        for ith in range(repeat):
            ranking = mallows.sample_a_ranking(rng)
            probs = rng.random(size=m) * pmax
            poset = get_random_poset_by_rsm(ranking, probs, rng)
            cardinality = poset.dag.number_of_nodes()

            df.loc[df.shape[0]] = [m, f'{phi:.1f}', f'{pmax:.1f}', ith, cardinality, poset]

    df.to_csv(get_dir(__file__) / f'experiment_input.tsv', index=False, sep='\t')


def run_experiment():
    sep = '\t'

    out_file = get_dir(__file__) / 'experiment_output.tsv'
    open_mode = 'a' if out_file.exists() else 'w'
    with open(out_file, open_mode, buffering=1) as out:
        if open_mode == 'w':
            cols = ['m', 'phi', 'pmax', 'ith_such_poset', 'item', 'cover_width', 'num_states', 't_sec']
            out.write(sep.join(cols) + '\n')
        else:
            df_existing = pd.read_csv(out_file, sep=sep)

        df_input = pd.read_csv(get_dir(__file__) / f'experiment_input.tsv', sep=sep)
        for _, row in df_input.iterrows():
            m = row['m']
            phi = row['phi']
            pmax = row['pmax']
            repeat = row['ith_such_poset']
            poset: Poset = eval(row['poset'])
            item = min(poset.items_in_poset)

            condition = f'(m == {m}) and (phi == {phi}) and (pmax == {pmax}) and (ith_such_poset == {repeat}) and ' \
                        f'(item == {item})'

            is_existing = (open_mode == 'a') and (not df_existing.query(condition).empty)

            if not is_existing:
                print(f'- Executing {condition}')
                mallows = Mallows(reference=poset.item_set_tuple, phi=1.0)
                t0 = time()
                answer = calculate_rank_probs_for_item_given_rim_w_poset_by_sequential(item=item, rim=mallows, poset=poset)
                t_sec = time() - t0

                record = [m, phi, pmax, repeat, item, answer['cover_width'], answer['num_states'], t_sec]
                record = [str(x) for x in record]

                out.write(sep.join(record) + '\n')

    print('Finished!')


if __name__ == '__main__':
    run_experiment()
