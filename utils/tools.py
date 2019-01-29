import yaml


def load_config(file):
    with open(file, 'r') as stream:
        dict_ = yaml.load(stream)
    return dict_
