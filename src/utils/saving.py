import pickle

# filename only accepts pathlib instances.


def load_pickle(filename):
    try:
        return pickle.load(open(filename, 'rb'))
    except:
        return 'Error'


def dump_pickle(obj, filename):
    filename.parents[0].mkdir(parents=True, exist_ok=True)
    pickle.dump(obj, open(filename, 'wb'))