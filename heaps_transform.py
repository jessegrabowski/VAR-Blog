from aesara.tensor.nlinalg import matrix_dot
import aesara.tensor as at
import aesara

from stan_math import mdivide_right, quad_form_sym
from scipy import linalg
import numpy as np

def matrix_square_root(A):
    u, w = at.linalg.eigh(A)
    
    root_root_u = at.sqrt(np.sqrt(u))
    eprod = w.dot(at.diag(root_root_u))
    A_sqrt = eprod.dot(eprod.T)
    
    return A_sqrt

def A_to_P(A):
    if A.ndim == 2:
        B = A.dot(A.T) + at.eye(A.shape[0])
        result = at.linalg.solve(matrix_square_root(B), A, assume_a='pos')
    else:
        result, _ = aesara.scan(A_to_P, sequences=A)
    return result

def transform_step(P, S0, Sigma):
    m = P.shape[0]
    S_for = -P.dot(P.T) + at.eye(m)
    S_rev = matrix_square_root(S_for)
    
    quad_term = quad_form_sym(Sigma, S_rev)
    inner = at.linalg.solve(S_rev, matrix_square_root(quad_term), assume_a='pos')
    
    S_for_out = mdivide_right(inner, S_rev)
    Sigma_for = S_for_out.dot(S_for_out.T)

    return S_for_out, Sigma_for

def autocovariance_update_step(S_for, Sigma_for, P, Sigma_rev):
    S_rev = matrix_square_root(Sigma_rev)
    phi_for = mdivide_right(S_for.dot(P), S_rev)
    phi_rev = mdivide_right(S_rev.dot(P.T), S_for)
    Gamma_trans = phi_for.dot(Sigma_rev)
    Sigma_rev_next = Sigma_rev - quad_form_sym(Sigma_for, phi_rev.T)

    return phi_for, phi_rev, Gamma_trans, Sigma_rev_next   

def cross_covariance_update_step(s, k, phi_for, phi_rev, Gamma_trans):
    phi_for = at.set_subtensor(phi_for[s, k], phi_for[s-1, k] - phi_for[s, s] @ phi_rev[s-1, s-k-1])
    phi_rev = at.set_subtensor(phi_rev[s, k], phi_rev[s-1, k] - phi_rev[s, s] @ phi_for[s-1, s-k-1])
    Gamma_trans = at.set_subtensor(Gamma_trans[s+1], Gamma_trans[s + 1] + phi_for[s-1, k] @ Gamma_trans[s-k])

    return phi_for, phi_rev, Gamma_trans    


def batch_diag(A):
    result, _ = aesara.scan(at.diag, sequences=[A])
    return result

def build_A_matrix(diag, offdiag, n_lags, n_eq):
    tril, triu = np.tril_indices(n_eq, k=-1), np.triu_indices(n_eq, k=1)
    corners = (np.r_[tril[0], triu[0]], np.r_[tril[1], triu[1]])
    A = at.zeros((n_lags, n_eq, n_eq))
    A = at.set_subtensor(A[:, corners[0], corners[1]], offdiag)
    A += batch_diag(diag)
    
    return A

def P_to_params(P, Sigma):
    p = P.shape[0]
    m = Sigma.shape[0]
    
    phi_forward = at.zeros((p, p, m, m))
    phi_reverse = at.zeros((p, p, m, m))
    
    # Step 1
    S_for0 = matrix_square_root(Sigma)
    result, update = aesara.scan(transform_step,
                                 sequences=[P],
                                 outputs_info=[S_for0, Sigma],
                                 go_backwards=True)
    S_for, Sigma_for = result
    S_for_list = at.concatenate([at.expand_dims(S_for0, 0), S_for], axis=0)[::-1]
    Sigma_for = at.concatenate([at.expand_dims(Sigma, 0), Sigma_for], axis=0)[::-1]
    
    # Step 2, part 1: Compute the diagonal matrices (autocovariance)
    Sigma_rev = Sigma_for[0]
    Gamma_trans = Sigma_for[0]

    result, update = aesara.scan(autocovariance_update_step,
                                 sequences=[S_for_list, Sigma_for, P],
                                 outputs_info=[None, None, None, Sigma_rev])
    
    phi_for_diag, phi_rev_diag, Gamma_trans, Sigma_rev = result
    
    phi_forward = at.set_subtensor(phi_forward[at.arange(p), at.arange(p)], phi_for_diag)
    phi_reverse = at.set_subtensor(phi_reverse[at.arange(p), at.arange(p)], phi_rev_diag)
    
    Gamma_trans = at.concatenate([at.expand_dims(Sigma_for[0], axis=0), Gamma_trans], axis=0)
    Sigma_rev = at.concatenate([at.expand_dims(Sigma_for[0], axis=0), Sigma_rev], axis=0)
    
    # Step 2, part 2: Compute the off-diagonal matrices (cross-covariance)
    s_idxs, k_idxs = at.tril_indices(p, k=-1)
    result, update = aesara.scan(cross_covariance_update_step,
                                 sequences=[s_idxs, k_idxs],
                                 outputs_info=[phi_forward, phi_reverse, Gamma_trans])
    
    phi_forward, phi_reverse, Gamma_trans = result
    phi_forward = phi_forward[-1]
    phi_reverse = phi_reverse[-1]
    Gamma_trans = Gamma_trans[-1]
    
    
    phiGamma = at.stack([phi_forward[-1], Gamma_trans.transpose(0,2,1)[:-1]], axis=0)
    
    return phiGamma


def np_mdivide_left_spd(A, B):
    assert np.allclose(A, A.T)
    assert np.all(np.linalg.eigvals(A) > 0.0)
    return np.linalg.solve(A, B)

def np_mdivide_right_spd(B, A):
    assert np.allclose(A, A.T)
    assert np.all(np.linalg.eigvals(A) > 0.0)
    return np_mdivide_left_spd(A, B.T).T

def tcrossprod(A):
    return A.dot(A.T)

def np_quad_form_sym(A, B):
    ret = np.linalg.multi_dot([B.T, A, B])
    ret = 0.5 * (ret + ret.T)
    assert np.allclose(ret, ret.T)
    return ret

def np_matrix_square_root(A):
    eig_vals, eig_vecs = np.linalg.eigh(A)
    
    # Stan sorts the eigenvalues, don't know if this is important
    sorted_idx = np.argsort(eig_vals)
    eig_vals = eig_vals[sorted_idx]
    eig_vecs = eig_vecs[:, sorted_idx]
    root_root_vals = np.sqrt(np.sqrt(eig_vals))
    
    eprod = eig_vecs.dot(np.diag(root_root_vals))
    A_sqrt = tcrossprod(eprod)
    
    assert np.allclose(A_sqrt @ A_sqrt, A)
    return A_sqrt

def np_A_to_P(A):
    out = np.empty_like(A)

    for i in range(A.shape[0]):
        _A = A[i].copy()
        B = _A.dot(_A.T) + np.eye(_A.shape[0])
        out[i] = np.linalg.solve(np_matrix_square_root(B), _A)
    return out

def reverse_mapping(P, Sigma):
    p = 1 if P.ndim < 3 else P.shape[0]
    m, _ = Sigma.shape
    Sigma_for = np.zeros((p+1, m, m))
    S_for_list = np.zeros((p+1, m, m))
    Sigma_rev = np.zeros((p+1, m, m))
    Gamma_trans = np.zeros((p+1, m, m))
    phi_for = np.zeros((p, p, m, m))
    phi_rev = np.zeros((p, p, m, m))
    phiGamma = np.zeros((2, p, m, m))
    
    Sigma_for[p] = Sigma
    S_for_list[p] = np_matrix_square_root(Sigma)
        
    for idx in range(p-1, -1, -1):
        S_for = -tcrossprod(P[idx]) + np.eye(m)
        S_rev = np_matrix_square_root(S_for)
        quad = np_quad_form_sym(Sigma_for[idx+1], S_rev)
        inner = np_mdivide_left_spd(S_rev, np_matrix_square_root(quad))
        S_for_list[idx] = np_mdivide_right_spd(inner, S_rev)
                
        Sigma_for[idx] = S_for_list[idx].dot(S_for_list[idx].T)
        
    Sigma_rev[0] = Sigma_for[0]
    Gamma_trans[0] = Sigma_for[0]
    
    for s in range(p):
        S_for = S_for_list[s]
        S_rev = np_matrix_square_root(Sigma_rev[s])
        phi_for[s, s] = np_mdivide_right_spd(S_for @ P[s], S_rev)
        phi_rev[s, s] = np_mdivide_right_spd(S_rev @ P[s].T, S_for)
        Gamma_trans[s+1] = phi_for[s, s] @ Sigma_rev[s]
        
        for k in range(s):
            phi_for[s, k] = phi_for[s-1, k] - phi_for[s, s] @ phi_rev[s-1, s-k-1]
            phi_rev[s, k] = phi_rev[s-1, k] - phi_rev[s, s] @ phi_for[s-1, s-k-1]
        
        for k in range(s):
            Gamma_trans[s+1] = Gamma_trans[s + 1] + phi_for[s-1, k] @ Gamma_trans[s-k]
        Sigma_rev[s+1] = Sigma_rev[s] - np_quad_form_sym(Sigma_for[s], phi_rev[s, s].T)
    
    
#     for s in range(p):
#         S_for = S_for_list[s]
#         S_rev = np_matrix_square_root(Sigma_rev[s])
#         print(Sigma_rev[s, 0, 0])
#         old_value = phi_for[s, s].copy()
#         phi_for[s, s] = np_mdivide_right_spd(S_for @ P[s], S_rev)
# #         print(f'Updating location {s}, {s}: {old_value} -> {phi_for[s, s]}')
#         phi_rev[s, s] = np_mdivide_right_spd(S_rev @ P[s].T, S_for)        
#         Gamma_trans[s+1] = phi_for[s, s] @ Sigma_rev[s]
                    
#         Sigma_rev[s+1] = Sigma_rev[s] - np_quad_form_sym(Sigma_for[s], phi_rev[s, s].T)
    
#     offdiag_idxs = np.tril_indices(p, k=-1)
#     for s, k in zip(*offdiag_idxs):
# #         print(s, k, phi_rev[s-1, k, 0, 0])
#         old_value = phi_for[s, k].copy()
#         phi_for[s, k] = phi_for[s-1, k] - phi_for[s, s] @ phi_rev[s-1, s-k]
# #         print(f'Updating location {s}, {s}: {old_value} -> {phi_for[s, k]}')
#         phi_rev[s, k] = phi_rev[s-1, k] - phi_rev[s, s] @ phi_for[s-1, s-k]
#         Gamma_trans[s+1] = Gamma_trans[s+1] + phi_for[s-1, k] @ Gamma_trans[s-k];

    phiGamma[0] = phi_for[-1]
    phiGamma[1] = np.transpose(Gamma_trans, axes=(0,2,1))[:-1]
    
    return phiGamma