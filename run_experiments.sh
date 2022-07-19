#!/usr/bin/env bash


PYTHONPATH=./ python 'experiments/synthetic/posets/experiment.py' >> 'experiments/synthetic/posets/printout.txt'
PYTHONPATH=./ python 'experiments/synthetic/posets_for_cover_width/experiment.py' >> 'experiments/synthetic/posets_for_cover_width/printout.txt'
PYTHONPATH=./ python 'experiments/synthetic/posets_for_voter_grouping/experiment.py' >> 'experiments/synthetic/posets_for_voter_grouping/printout.txt'
PYTHONPATH=./ python 'experiments/synthetic/posets_for_parallelization/experiment.py' >> 'experiments/synthetic/posets_for_parallelization/printout.txt'
PYTHONPATH=./ python 'experiments/synthetic/posets_for_parallelization/experiment_cm.py' >> 'experiments/synthetic/posets_for_parallelization/printout_cm.txt'

PYTHONPATH=./ python 'experiments/synthetic/tr/experiment.py' >> 'experiments/synthetic/tr/printout.txt'
PYTHONPATH=./ python 'experiments/synthetic/pc/experiment.py' >> 'experiments/synthetic/pc/printout.txt'
PYTHONPATH=./ python 'experiments/synthetic/pp/experiment.py' >> 'experiments/synthetic/pp/printout.txt'
PYTHONPATH=./ python 'experiments/synthetic/ppwm/experiment.py' >> 'experiments/synthetic/ppwm/printout.txt'

PYTHONPATH=./ python 'experiments/synthetic/mallows/experiment.py' >> 'experiments/synthetic/mallows/printout.txt'
PYTHONPATH=./ python 'experiments/synthetic/rsm/experiment.py' >> 'experiments/synthetic/rsm/printout.txt'

PYTHONPATH=./ python 'experiments/synthetic/mallows_poset/experiment.py' >> 'experiments/synthetic/mallows_poset/printout.txt'
PYTHONPATH=./ python 'experiments/synthetic/mallows_pp/experiment.py' >> 'experiments/synthetic/mallows_pp/printout.txt'
PYTHONPATH=./ python 'experiments/synthetic/mallows_tr/experiment.py' >> 'experiments/synthetic/mallows_tr/printout.txt'

PYTHONPATH=./ python 'experiments/real/movielens/experiment.py' >> 'experiments/real/movielens/printout.txt'
PYTHONPATH=./ python 'experiments/real/crowdrank/experiment.py' >> 'experiments/real/crowdrank/printout.txt'
PYTHONPATH=./ python 'experiments/real/travel/experiment.py' >> 'experiments/real/travel/printout.txt'

PYTHONPATH=./ python 'experiments/synthetic/mpw/experiment.py' >> 'experiments/synthetic/mpw/printout.txt'
