from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll
from hyperopt.base import miscs_update_idxs_vals
from hyperopt.pyll.base import dfs, as_apply
from hyperopt.pyll.stochastic import implicit_stochastic_symbols
import numpy as np

class ExhaustiveSearchError(Exception):
    pass


def validate_space_exhaustive_search(space):
    supported_stochastic_symbols = ['randint', 'quniform', 'qloguniform', 'qnormal', 'qlognormal', 'categorical']
    for node in dfs(as_apply(space)):
        if node.name in implicit_stochastic_symbols:
            if node.name not in supported_stochastic_symbols:
                raise ExhaustiveSearchError('Exhaustive search is only possible with the following stochastic symbols: ' + ', '.join(supported_stochastic_symbols))


def suggest(new_ids, domain, trials, seed, nbMaxSucessiveFailures=1000, ntrials = 100):
    
    if len(trials.trials)>0:
        print("Trial: {:4d}/{:4d}, Best loss: {:.8f}".format(len(trials.trials)+1, ntrials, np.min([r['loss'] for r in trials.results])))
    else:
        print("Trial: {:4d}/{:4d}, Best loss: {:.8f}".format(1, ntrials, np.inf))
        
    # Build a hash set for previous trials
    hashset = set([hash(frozenset([(key, value[0]) if len(value) > 0 else ((key, None))
                                   for key, value in trial['misc']['vals'].items()])) for trial in trials.trials])

    rng = np.random.RandomState(seed)
    rval = []
    for _, new_id in enumerate(new_ids):
        newSample = False
        nbSucessiveFailures = 0
        while not newSample:
            # -- sample new specs, idxs, vals
            idxs, vals = pyll.rec_eval(
                domain.s_idxs_vals,
                memo={
                    domain.s_new_ids: [new_id],
                    domain.s_rng: rng,
                })
            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)

            # Compare with previous hashes
            h = hash(frozenset([(key, value[0]) if len(value) > 0 else (
                (key, None)) for key, value in vals.items()]))
            if h not in hashset and len(trials.trials) < ntrials:
                newSample = True
            else:
                # Duplicated sample, ignore
                nbSucessiveFailures += 1
            
            if nbSucessiveFailures > nbMaxSucessiveFailures:
                # No more samples to produce
                return []

        rval.extend(trials.new_trial_docs([new_id],
                                          [None], [new_result], [new_misc]))
    return rval


# # Define an objective function
# def objective(args):
#     args.pop('save_file',None)
#     print(args)
#     return 0


# space = hp.choice('a', [{'a': 'x'},
#                         {'a': 'y', 'b': hp.choice('c', [1, 2, 3])},
#                         {'a': 'z', 'b': hp.choice('e', [hp.choice('f', [0.0, 1.0]),
#                                                         hp.quniform('g', 10.0, 100.0, 10)])}
#                         ])

# validate_space_exhaustive_search(space)

# trials = Trials()
# best = fmin(fn=objective,
#             space=space,
#             trials=trials,
#             algo=partial(suggest, nbMaxSucessiveFailures=1000, ntrials=200),
#             max_evals=np.inf,
#             show_progressbar=True)