import yaml

def to_dotdict(d):
    if isinstance(d, dict):
        return type('DotDict', (dict,), {
            '__getattr__': lambda self, name: to_dotdict(self[name]),
            '__setattr__': lambda self, name, value: self.__setitem__(name, value)
        })(**{k: to_dotdict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [to_dotdict(i) for i in d]
    return d

def load_config(path):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    return to_dotdict(d)

