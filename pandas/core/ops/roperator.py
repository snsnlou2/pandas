
'\nReversed Operations not available in the stdlib operator module.\nDefining these instead of using lambdas allows us to reference them by name.\n'
import operator

def radd(left, right):
    return (right + left)

def rsub(left, right):
    return (right - left)

def rmul(left, right):
    return (right * left)

def rdiv(left, right):
    return (right / left)

def rtruediv(left, right):
    return (right / left)

def rfloordiv(left, right):
    return (right // left)

def rmod(left, right):
    if isinstance(right, str):
        typ = type(left).__name__
        raise TypeError(f'{typ} cannot perform the operation mod')
    return (right % left)

def rdivmod(left, right):
    return divmod(right, left)

def rpow(left, right):
    return (right ** left)

def rand_(left, right):
    return operator.and_(right, left)

def ror_(left, right):
    return operator.or_(right, left)

def rxor(left, right):
    return operator.xor(right, left)
