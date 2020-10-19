import pathlib
import yaml
import glob

REGISTRY = pathlib.Path(__file__).parent.absolute()
registry = None

class Registry(dict):
    ''' Implements access to metadata about registered things
    '''
    def __init__(self, **kwargs):
        ''' Load package yaml files '''
        super().__init__()
        for f in REGISTRY.glob('*.yaml'):
            with open(f) as f:
                data = yaml.safe_load(f)
            self.update(data)

    def update(self, other):
        ''' update registry with data from other, accomodate URLs to codes registries '''
        super().update(other)

registry = Registry()
