from dataclasses import dataclass

@dataclass(init=False)
class Variable(dict):
    '''
    Variable instances hold full variable metadata.
    Variables may range from very generic to very specific.
    specific variables are used for expressing data that is to be selected...
    potentially for passing to procedures
    '''

    name : str
    long_name : str = None
    standard_name : str = None
    data_type : str = None
    units : str = None
    valid_min : Number = None
    valid_max : Number = None
    coordinate_variables : list = None
    OM__observedProperty : str = None
    SOSA__usedProcedure : list = None

    def __init__(self, name):
        super().__init__()
        self.__dict__ = self
