import jax.numpy as jnp
from jax import random

def get_neuron_connectivity(N_neurons, R_neurons):
    # neurons are uniformly distributed on circle; connected if separated by R_neuron
    x_neurons = jnp.linspace(0, 1, N_neurons+1)[:-1]
    ind_neurons = jnp.arange(N_neurons)
    connectivity = []

    for i, x in zip(ind_neurons, x_neurons):
        d1 = jnp.minimum(jnp.abs((x - 1.0 - x_neurons)), jnp.abs((x + 1.0 - x_neurons)))
        d = jnp.minimum(d1, jnp.abs((x - x_neurons)))
        overlap = d <= R_neurons
        overlap = ind_neurons[overlap]
        overlap = overlap[i != overlap]
        ind = jnp.stack([overlap, jnp.ones_like(overlap)*i], axis=1)
        connectivity.append(ind)
    connectivity = jnp.concatenate(connectivity, axis=0)
    return connectivity, x_neurons

def get_diff_connectivity(N_astrocytes):
    # astrocytes are uniformly distributed on the circle; nearest neighbours are connected
    x_astrocytes = jnp.linspace(0, 1, N_astrocytes+1)[:-1]
    ind_astrocytes = jnp.arange(N_astrocytes)
    diff_connectivity = []
    w_diff = []

    for i in ind_astrocytes:
        for s in [-1, 0, +1]:
            diff_connectivity.append([i, ind_astrocytes[(i+s) % N_astrocytes]])
        w_diff += [1, -2, 1]
    diff_connectivity = jnp.array(diff_connectivity)
    w_diff = jnp.array(w_diff)
    return diff_connectivity, w_diff, x_astrocytes

def get_tripartite_connectivity(N_astrocytes, connectivity, R_astrocyte, x_neurons, key=random.PRNGKey(44)):
    # astrocytes are attached to particular synapse if presynaptic element is within range R_astrocyte from the location of astrocyte
    x_astrocytes = jnp.linspace(0, 1, N_astrocytes+1)[:-1]
    ind_astrocytes = jnp.arange(N_astrocytes)
    synapse_affinity = [[] for _ in range(connectivity.shape[0])]
    ind_tripartite = []

    for i, x in zip(ind_astrocytes, x_astrocytes):
        d1 = jnp.minimum(jnp.abs((x - 1.0 - x_neurons[connectivity[:, 1]])), jnp.abs((x + 1.0 - x_neurons[connectivity[:, 1]])))
        d = jnp.minimum(d1, jnp.abs((x - x_neurons[connectivity[:, 1]])))
        overlap = d <= R_astrocyte
        for j, o in enumerate(overlap):
            if o:
                synapse_affinity[j].append(i.item())

    keys = random.split(key, len(synapse_affinity))
    for i, key in enumerate(keys):
        ind_tripartite.append(random.choice(key, jnp.array(synapse_affinity[i])))
    ind_tripartite = jnp.array(ind_tripartite)
    return ind_tripartite