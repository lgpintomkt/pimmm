import numpy as np
from scipy.integrate import odeint

def gbm_ode_step(F, t, p, q, g, spend_array, dt):
    """
    Generalized Bass Model (GBM) Ordinary Differential Equation.
    dF/dt = (p + q*F) * (1 - F) * (gamma * sqrt(spend))
    """
    idx = min(int(t / dt), len(spend_array) - 1)
    
    spend_val = np.asarray(spend_array[idx]).squeeze()
    if spend_val.ndim > 0:
        spend_val = np.mean(spend_val)
        
    spend_t = max(0.0, float(spend_val))
    latent_effort = g * np.sqrt(spend_t)
    
    dFdt = (p + q * F[0]) * (1.0 - F[0]) * latent_effort
    return [dFdt]

def simulate_pimmm(params, spend_array, t_eval, F0=0.001, dt=0.5):
    """
    Simulate cumulative adoption trajectory F(t) given system parameters.
    """
    p, q, gamma = params
    spend_array = np.asarray(spend_array).squeeze() # Ensure 1D structure
    
    def ode_func(F, t):
        return gbm_ode_step(F, t, p, q, gamma, spend_array, dt)
    
    F_sim = odeint(ode_func, [F0], t_eval).flatten()
    return F_sim