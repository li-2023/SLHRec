PRED_DICT = {}


class Predicate:

    def __init__(self, name, var_types):

        self.name = name
        self.var_types = var_types
        self.num_args = len(var_types)

    def __repr__(self):
        return '%s(%s)' % (self.name, ','.join(self.var_types))
