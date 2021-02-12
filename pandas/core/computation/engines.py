
'\nEngine classes for :func:`~pandas.eval`\n'
import abc
from typing import Dict, Type
from pandas.core.computation.align import align_terms, reconstruct_object
from pandas.core.computation.ops import MATHOPS, REDUCTIONS
import pandas.io.formats.printing as printing
_ne_builtins = frozenset((MATHOPS + REDUCTIONS))

class NumExprClobberingError(NameError):
    pass

def _check_ne_builtin_clash(expr):
    '\n    Attempt to prevent foot-shooting in a helpful way.\n\n    Parameters\n    ----------\n    terms : Term\n        Terms can contain\n    '
    names = expr.names
    overlap = (names & _ne_builtins)
    if overlap:
        s = ', '.join((repr(x) for x in overlap))
        raise NumExprClobberingError(f'Variables in expression "{expr}" overlap with builtins: ({s})')

class AbstractEngine(metaclass=abc.ABCMeta):
    'Object serving as a base class for all engines.'
    has_neg_frac = False

    def __init__(self, expr):
        self.expr = expr
        self.aligned_axes = None
        self.result_type = None

    def convert(self):
        '\n        Convert an expression for evaluation.\n\n        Defaults to return the expression as a string.\n        '
        return printing.pprint_thing(self.expr)

    def evaluate(self):
        '\n        Run the engine on the expression.\n\n        This method performs alignment which is necessary no matter what engine\n        is being used, thus its implementation is in the base class.\n\n        Returns\n        -------\n        object\n            The result of the passed expression.\n        '
        if (not self._is_aligned):
            (self.result_type, self.aligned_axes) = align_terms(self.expr.terms)
        res = self._evaluate()
        return reconstruct_object(self.result_type, res, self.aligned_axes, self.expr.terms.return_type)

    @property
    def _is_aligned(self):
        return ((self.aligned_axes is not None) and (self.result_type is not None))

    @abc.abstractmethod
    def _evaluate(self):
        '\n        Return an evaluated expression.\n\n        Parameters\n        ----------\n        env : Scope\n            The local and global environment in which to evaluate an\n            expression.\n\n        Notes\n        -----\n        Must be implemented by subclasses.\n        '
        pass

class NumExprEngine(AbstractEngine):
    'NumExpr engine class'
    has_neg_frac = True

    def _evaluate(self):
        import numexpr as ne
        s = self.convert()
        env = self.expr.env
        scope = env.full_scope
        _check_ne_builtin_clash(self.expr)
        return ne.evaluate(s, local_dict=scope)

class PythonEngine(AbstractEngine):
    '\n    Evaluate an expression in Python space.\n\n    Mostly for testing purposes.\n    '
    has_neg_frac = False

    def evaluate(self):
        return self.expr()

    def _evaluate(self):
        pass
ENGINES = {'numexpr': NumExprEngine, 'python': PythonEngine}
