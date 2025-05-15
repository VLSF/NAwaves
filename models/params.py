import jax.numpy as jnp

def get_standard_params(connectivity, ind_tripartite, diff_connectivity, w_diff, N_neurons, N_astrocytes, t):
    # counting_number of synapses
    synapse_counter = jnp.zeros((N_astrocytes,))
    synapse_counter = synapse_counter.at[ind_tripartite].add(1)
    synapse_counter = synapse_counter + (synapse_counter == 0)
    
    # diffusion parameters
    diff_IP3 = 0.1 * 1e-3
    diff_Ca = 0.05 * 1e-3
    w_diff = jnp.stack([w_diff*diff_IP3, w_diff*diff_Ca], axis=0)
    
    # main parameters
    weights = jnp.ones((connectivity.shape[0],)) * 0.025
    a = jnp.ones((N_neurons,))*0.1
    b = jnp.ones((N_neurons,))*0.2
    c = jnp.array([-65.0,]*N_neurons)
    d = jnp.array([2.0,]*N_neurons)
    E_synaptic = jnp.array([0.0,])
    k_synaptic = jnp.array([0.2,])
    alpha_glutamate = jnp.array([10.0,]) * 1e-3
    k_glutamate = jnp.array([600.0,]) * 1e-3
    A_glutamate = jnp.array([5.0,]) * 1e-3
    t_glutamate = jnp.array([60.0,])
    F_active = jnp.array([0.1,])
    G_threshold = jnp.array([0.1,])
    IP_3_star = jnp.array([0.16, ])
    tau_IP_3 = jnp.array([7.14,]) * 1e3
    nu_4 = jnp.array([0.3,]) * 1e-3
    k_4 = jnp.array([1.1,])
    alpha = jnp.array([0.8,])
    
    c0 = jnp.array([2.0,])
    c1 = jnp.array([0.185,])
    nu1 = jnp.array([6.0,]) * 1e-3
    nu2 = jnp.array([0.11,]) * 1e-3
    d1 = jnp.array([0.13,])
    d5 = jnp.array([82.0,]) * 1e-3
    nu3 = jnp.array([2.2,]) * 1e-3
    k3 = jnp.array([0.1,])
    nu6 = jnp.array(0.2,) * 1e-3
    k2 = jnp.array([1.0,])
    k1 = jnp.array([0.5,]) * 1e-3
    a2 = jnp.array([0.14,]) * 1e-3 # not given in the article, found in https://pubmed.ncbi.nlm.nih.gov/16330095/
    d2 = jnp.array([1.049,])
    d1 = jnp.array([0.13,])
    d3 = jnp.array([943.4,]) * 1e-3
    
    F_astrocyte = jnp.array([0.375,])
    t_astrocyte = jnp.array([250.0,])
    Ca_threshold = jnp.array([0.15,])
    nu_star_Ca = jnp.array([0.5,])

    h = t[1] - t[0]
    
    # initial conditions
    x = jnp.stack([-65*jnp.ones((N_neurons,)), b*c, jnp.zeros((N_neurons,))])
    y = jnp.zeros((3, N_astrocytes))
    # y = y.at[0].set(IP_3_star)
    R = jnp.zeros((2, N_astrocytes))
    
    params = {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "h": h,
        "E_synaptic": E_synaptic,
        "k_synaptic": k_synaptic,
        "alpha_glutamate": alpha_glutamate,
        "k_glutamate": k_glutamate,
        "A_glutamate": A_glutamate,
        "t_glutamate": t_glutamate,
        "F_active": F_active,
        "G_threshold": G_threshold,
        "IP_3_star": IP_3_star,
        "tau_IP_3": tau_IP_3,
        "nu_4": nu_4,
        "k_4": k_4,
        "alpha": alpha,
        "c0": c0,
        "c1": c1,
        "nu1": nu1,
        "nu2": nu2,
        "d1": d1,
        "d5": d5,
        "nu3": nu3,
        "k3": k3,
        "nu6": nu6,
        "k2": k2,
        "k1": k1,
        "a2": a2,
        "d2": d2,
        "d1": d1,
        "d3": d3,
        "F_astrocyte": F_astrocyte,
        "t_astrocyte": t_astrocyte,
        "Ca_threshold": Ca_threshold,
        "nu_star_Ca": nu_star_Ca,
        "w_neurons": weights,
        "w_diff": w_diff,
        "ind_neurons": connectivity,
        "ind_astrocytes": diff_connectivity,
        "ind_tripartite": ind_tripartite,
        "synapse_counter": synapse_counter
    }
    return params