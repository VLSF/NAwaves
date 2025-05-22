import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import sys

from jax.lax import scan
from jax import random
from models import gg_model, params
from connectivity import plane_I

def constant_DC_current_3_stages(N_neurons, R_neurons, N_astrocytes, R_astrocyte, I, R, coupling, T_max, N_int):
    N_t = int(T_max * 3)
    t = jnp.linspace(0, T_max, N_t)
    h = t[1] - t[0]
    ind_ = jnp.arange(N_t)[:N_int]
    N_firing_track = int(10 // h)

    # connectivities
    connectivity, x_neurons = plane_I.get_neuron_connectivity(N_neurons, R_neurons)
    ind_tripartite = plane_I.get_tripartite_connectivity(N_astrocytes, N_neurons, connectivity)
    diff_connectivity, w_diff, x_astrocytes = plane_I.get_diff_connectivity(N_astrocytes)
    
    # parameters
    params_ = params.get_standard_params(connectivity, ind_tripartite, diff_connectivity, w_diff, N_neurons**2, N_astrocytes**2, t)
    params_["nu_star_Ca"] *= coupling
    
    # initial conditions
    neurons_state = jnp.stack([-65*jnp.ones((N_neurons**2,)), params_["b"]*params_["c"], jnp.zeros((N_neurons**2,))])
    astrocytes_state = jnp.zeros((3, N_astrocytes**2))
    astrocytes_state = astrocytes_state.at[0].set(params_["IP_3_star"])
    events_variables = jnp.zeros((2, N_astrocytes**2))
    firing_patterns = jnp.zeros((N_firing_track, N_neurons**2 + 1))
    
    carry = [params_, firing_patterns, t, neurons_state, astrocytes_state, events_variables, x_neurons]
    
    def I_applied(x, y, R_, firing_patterns, t, ind_, params):
        I_ = 0
        for i, r in zip(I, R):
            I_ += i * (jnp.linalg.norm(x_neurons - 0.5, axis=1) <= r)
        return I_
    
    integrate_scan_ = lambda carry, ind_: gg_model.integrate_scan(carry, ind_, I_applied)
    carry, (neuron_history, astrocyte_history, relaxation_params_history, firing_patterns) = scan(integrate_scan_, carry, ind_)
    return neuron_history, astrocyte_history, relaxation_params_history, firing_patterns, t[:N_int], x_neurons, x_astrocytes

if __name__ == "__main__":
    exp_type = sys.argv[1]

    if exp_type == "1":
        N_neurons = 50
        R_neurons = 4 / N_neurons
        N_astrocytes = 25
        R_astrocyte = 4 / N_astrocytes
        coupling = 1.0
        I = [4.0, 3.0, 2.0]
        R = [0.1, 0.3, 0.4]
        T_max = 75 * 1e3
        N_int = 100000
    
    neuron_history, astrocyte_history, relaxation_params_history, firing_patterns, t, x_neurons, x_astrocytes = constant_DC_current_3_stages(N_neurons, R_neurons, N_astrocytes, R_astrocyte, I, R, coupling, T_max, N_int)
    print("done")
    data = {
        "neuron_history": neuron_history,
        "astrocyte_history": astrocyte_history,
        "relaxation_params_history": relaxation_params_history,
        "firing_patterns": firing_patterns,
        "t": t,
        "x_neurons": x_neurons,
        "x_astrocytes": x_astrocytes
    }
    
    jnp.savez("test_run.npz", **data)