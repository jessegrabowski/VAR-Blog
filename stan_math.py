import aesara.tensor as at
from aesara.tensor.nlinalg import matrix_dot


def mdivide_right(B, A):
    '''
    Following the signature of the Stan function. Pay attention that B is first.
    '''
    return at.linalg.solve(A, B.T, assume_a='pos').T

def quad_form_sym(A, B):
    '''
    Compute the quadratic of A and B, B.T @ A @ B, and normalize the off-diagonal to be symmetric
    '''
    ret =  matrix_dot(B.T, A, B)
    return 0.5 * (ret + ret.T)
