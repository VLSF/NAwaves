import jax.numpy as jnp

def f(x, y, I, dG, dIP3, diff, params):
    # Izhikevich neurons https://ieeexplore.ieee.org/document/1257420
    # Astrocyte-neuron coupling according to https://arxiv.org/abs/2011.01750
    v_ = 0.04 * x[0]**2  + 5 * x[0] + 140 - x[1] + I
    u_ = params['a']*(params['b']*x[0] - x[1])
    g_ = -params["alpha_glutamate"]*x[2] + dG
    ip3_ = (params["IP_3_star"] - y[0]) / params["tau_IP_3"] + params["nu_4"]*(y[1] + (1 - params["alpha"])*params["k_4"]) / (y[1] + params["k_4"]) + dIP3 + diff[0]
    J_supplementary = params["c0"] - (1 + params["c1"])*y[1]
    ca_ = params["nu1"]*((y[0]*y[1]*y[2])/(y[0] + params["d1"])/(y[1] + params["d5"]))**3*J_supplementary # J_ER
    ca_ -= params["nu3"]*y[1]**2/(params["k3"]**2 + y[1]**2) # J_pump
    ca_ += params["nu2"]*J_supplementary # J_leak
    ca_ += params["nu6"]*y[0]**2/(params["k2"]**2 + y[0]**2) # J_in
    ca_ -= params["k1"]*y[1] # J_out
    ca_ += diff[1]
    h_ = params["a2"]*(params["d2"]*(1 - y[2])*(y[0] + params["d1"])/(y[0] + params["d3"]) - y[1]*y[2])
    return jnp.stack([v_, u_, g_]), jnp.stack([ip3_, ca_, h_])

def timestep(x, y, R, I_applied, firing_patterns, params):
    # reset
    fired = x[0] >= 30
    v = x[0] * (1 - fired) + params["c"] * fired
    u = x[1] * (1 - fired) + (x[1] + params["d"]) * fired
    x = jnp.stack([v, u, x[2]])

    # glutamate fire interaction
    dG = fired * params["k_glutamate"]

    # detect IP3 event
    active_glutamate = x[2] >= params["G_threshold"]
    active_astrocytes = jnp.zeros((R.shape[1],))
    active_astrocytes = active_astrocytes.at[params["ind_tripartite"]].add(active_glutamate[params["ind_neurons"][:, 1]]) / params["synapse_counter"]
    active_astrocytes = active_astrocytes >= params["F_active"]
    R = R.at[0].set(R[0] + active_astrocytes * (R[0] == 0) * params["t_glutamate"])

    # detect Ca event
    active_astrocytes = jnp.zeros((R.shape[1],))
    active_astrocytes = active_astrocytes.at[params["ind_tripartite"]].add(jnp.sum(firing_patterns[:, 1:], axis=0)[params["ind_neurons"][:, 1]]) / params["synapse_counter"]
    active_astrocytes = active_astrocytes >= params["F_astrocyte"]
    R = R.at[1].set(R[1] + active_astrocytes * (R[1] == 0) * params["t_astrocyte"])

    # glutamate IP3 interaction and weight modification
    dIP3 = (R[0] > 0) * params["A_glutamate"]
    dw = (R[1] > 0) * params["nu_star_Ca"] * (y[1] > params["Ca_threshold"])
    y_ = R - params['h'].reshape(1, -1)
    R = y_ * (y_ >= 0)

    # neuron interaction
    I = I_applied.copy()
    I = I.at[params["ind_neurons"][:, 0]].add((params["w_neurons"] + dw[params["ind_tripartite"]])*(params["E_synaptic"] - v[params["ind_neurons"][:, 0]])/(1 + jnp.exp(-v[params["ind_neurons"][:, 1]]/params["k_synaptic"]))) # smooth interaction
    #I = I.at[params["ind_neurons"][:, 0]].add(w*fired[params["ind_neurons"][:, 1]]) # fire interaction # 'fire' interaction

    # diffusive transport
    diff = jnp.zeros_like(y[:2])
    diff = diff.at[:, params["ind_astrocytes"][:, 0]].add(y[:2, params["ind_astrocytes"][:, 1]]*params["w_diff"])

    # rk4
    k1, k1_ = f(x, y, I, dG, dIP3, diff, params)
    k2, k2_ = f(x + params['h']*k1/2, y + params['h']*k1_/2, I/2, dG/2, dIP3/2, diff/2, params)
    k3, k3_ = f(x + params['h']*k2/2, y + params['h']*k2_/2, I/2, dG/2, dIP3/2, diff/2, params)
    k4, k4_ = f(x + params['h']*k3, y + params['h']*k3_, I, dG, dIP3, diff, params)
    x = x + params['h']*(k1 + 2*k2 + 2*k3 + k4) / 6
    y = y + params['h']*(k1_ + 2*k2_ + 2*k3_ + k4_) / 6
    
    return x, y, R, fired

def integrate_scan(carry, ind_, I_applied):
    params, firing_patterns, t, x, y, R, x_neurons = carry
    I_applied_ = I_applied(x, y, R, firing_patterns, t, ind_, params) # external control
    x, y, R, fired = timestep(x, y, R, I_applied_, firing_patterns, params)
    # tracking neurons that fired for last firing_patterns.shape[0] integration steps
    firing_patterns = firing_patterns.at[ind_ % firing_patterns.shape[0]].set(jnp.concatenate([t[ind_].reshape(1,), fired]))
    return [params, firing_patterns, t, x, y, R, x_neurons], [x, y, R, firing_patterns]