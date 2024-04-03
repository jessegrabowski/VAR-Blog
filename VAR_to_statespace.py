from numba import njit
import numpy as np

@njit
def make_statespace(coefs, intercepts):
    n_obs, n_lags, _ = coefs.shape
    n_states = n_obs * n_lags
    
    T = np.zeros((n_states, n_states))
    T[:n_obs, :] = coefs.reshape(n_obs, n_states)
    T[n_obs:, :-n_obs] = np.eye((n_states - n_obs)) 
    
    c = np.zeros((n_states, 1))
    c[:n_obs, 0] = intercepts
    Z = np.zeros((n_obs, n_states))
    R = np.zeros((n_states, n_obs))
    for i in range(n_obs):
        Z[i, i] = 1.0
        R[i, i] = 1.0
    
    H = np.zeros((n_obs, n_obs))
    Q = np.zeros((n_obs, n_obs))
    d = np.zeros((n_obs, 1))
    
    return T, Z, R, H, Q, c, d

@njit
def numba_MV_normal(cov, n_draws, mu=0):
    k, _ = cov.shape
    L = np.linalg.cholesky(cov)
    standard_shocks = np.random.normal(0, 1, size=(k, n_draws))
    mv_draws = mu + (L @ standard_shocks).T
    
    return np.expand_dims(mv_draws, -1)

@njit
def simulate(x0, T, Z, R, H, Q, c, d, shock_trajectory=None, sim_length=40):
    k_obs, k_hidden = Z.shape
    _, k_posdef = R.shape
    
    # We're going to chop off the t=0 obs, since it's just the inital state,
    # so add 1 step to the end to compensate
    sim_length += 1

    if shock_trajectory is None:
        if np.all(Q == 0):
            shock_trajectory = np.zeros((sim_length, k_posdef, 1))
        else:
            shock_trajectory = np.ascontiguousarray(numba_MV_normal(Q, sim_length))
    if shock_trajectory.shape[0] < sim_length:
        temp = np.zeros((sim_length, k_posdef, 1))
        temp[:shock_trajectory.shape[0], :, :] = shock_trajectory
        shock_trajectory = temp.copy()
    
    if np.all(H == 0):
        observation_noise = np.zeros((sim_length, k_obs, 1))
    else:
        observation_noise = numba_MV_normal(H, sim_length)
    
#     shock_trajectory = np.ascontiguousarray(shock_trajectory)
    
    simulated_data = np.zeros((sim_length, k_obs))
    simulated_states = np.zeros((sim_length, k_hidden))
    
    x = x0.copy()
    for t in range(sim_length):
        simulated_data[t] = (Z @ x + d + observation_noise[t]).ravel()
        x = T @ x + c + R @ shock_trajectory[t]
        simulated_states[t] = x.ravel()
    
    return simulated_states[:-1], simulated_data[1:]

@njit
def statespace_forecast(x0, coefs, intercepts, sigmas, sigmas_are_chol=True, horizon=10):
    n_samples, n_obs, n_lags, _ = coefs.shape
    forecasts = np.zeros((n_samples, horizon, n_obs), dtype='d')
    
    for sample in range(n_samples):
        T, Z, R, H, Q, c, d = make_statespace(coefs[sample], intercepts[sample])
        
        if sigmas_are_chol:
            tril_idxs = np.tril_indices(n_obs)
            #loop because numba can only do one fancy-index at a time, and numba loops are fast anyway
            for i, (j, k) in enumerate(zip(tril_idxs[0], tril_idxs[1])):
                Q[j, k] = sigmas[sample][i]
        else:
            for i in range(n_obs):
                Q[i, i] = sigmas[sample][i]
        
        # This is admittedly silly because I re-cholesky it right away...
        Q = Q @ Q.T
        
        simulated_states, simulated_data = simulate(x0, T, Z, R, H, Q, c, d, shock_trajectory=None, sim_length=horizon)
        forecasts[sample] = simulated_data
        
    return forecasts
    
@njit
def bayesian_impulse_response_function(coefs, sim_length=40, shock_state=0, shock_size=1.0):
    n_samples, n_obs, n_lags, _ = coefs.shape
    n_states = n_obs * n_lags
    
    shock_trajectory = np.zeros((sim_length, n_obs, 1))
    shock_trajectory[0, shock_state] = shock_size
    simulations = np.empty((n_samples, sim_length, n_obs))
    
    for sample in range(n_samples):
        T, Z, R, H, Q, c, d = make_statespace(coefs[sample], np.zeros(n_obs))
        x0 = np.zeros((n_states, 1))
        simulated_states, simulated_data = simulate(x0, T, Z, R, H, Q, c, d, shock_trajectory=shock_trajectory, sim_length=sim_length)
        simulations[sample] = simulated_data
    
    return simulations

@njit
def stability_analysis(coefs):
    n_samples, n_obs, n_lags, _ = coefs.shape
    n_states = n_obs * n_lags
    
    eigs = np.empty((n_samples, n_states), dtype='D')
    
    for sample in range(n_samples):
        T, Z, R, H, Q, c, d = make_statespace(coefs[sample], np.zeros(n_obs))
        eigs[sample] = np.linalg.eigvals(T.astype('D'))
        
    return eigs