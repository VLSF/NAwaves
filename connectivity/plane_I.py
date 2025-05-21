import jax.numpy as jnp
from jax.lax import scan

def get_neighbours(ind, window):
    j = window + jnp.expand_dims(ind, 1)
    i = jnp.stack([ind,]*j.shape[1], axis=1)
    return i, j

def get_neighbours_scan(carry, i):
    window, ind = carry
    a, b = get_neighbours(ind[:, i], window)
    return carry, (a, b)

def lex(i, j, N_neurons):
    return j + i*N_neurons

def get_mask(ind, N_neurons):
    mask1 = jnp.logical_and(ind[0] < N_neurons, ind[0] >= 0)
    mask2 = jnp.logical_and(ind[1] < N_neurons, ind[1] >= 0)
    mask = jnp.logical_and(mask1, mask2)
    return mask

def get_neuron_connectivity(N_neurons, R_neurons):
    x = jnp.linspace(0, 1, N_neurons)
    N_window = (R_neurons / (x[1] - x[0])).astype(int).item()
    N_window = N_window if N_window != 0 else 1
    coords = jnp.stack(jnp.meshgrid(x, x, indexing="ij")).reshape(2, -1)
    ind = jnp.arange(N_neurons)
    ind = jnp.stack(jnp.meshgrid(ind, ind, indexing="ij")).reshape(2, -1)
    window = jnp.arange(-N_window, N_window+1, 1)
    window = jnp.stack(jnp.meshgrid(window, window, indexing="ij")).reshape(2, -1)
    window = window[:, jnp.logical_not(jnp.all(window == 0, axis=0))]

    r = jnp.arange(ind.shape[1])
    _, (i, j) = scan(get_neighbours_scan, [window, ind], r)
    i, j = jnp.transpose(i, (1, 0, 2)).reshape(2, -1), jnp.transpose(j, (1, 0, 2)).reshape(2, -1)
    mask = get_mask(j, N_neurons)
    i, j = i[:, mask], j[:, mask]
    i_lex = lex(i[0], i[1], N_neurons)
    j_lex = lex(j[0], j[1], N_neurons)

    connectivity = jnp.stack([i_lex, j_lex], axis=1)
    return connectivity, coords.T

def get_diff_connectivity(N_astrocytes):
    x = jnp.linspace(0, 1, N_astrocytes)
    coords = jnp.stack(jnp.meshgrid(x, x, indexing="ij")).reshape(2, -1)
    ind = jnp.arange(N_astrocytes)
    ind = jnp.stack(jnp.meshgrid(ind, ind, indexing="ij")).reshape(2, -1)
    window = jnp.array([
        [0, 0, 0, -1, 1],
        [-1, 0, 1, 0, 0]
    ])
    weights = jnp.array([1, -4, 1, 1, 1])
    
    r = jnp.arange(ind.shape[1])
    _, (i, j) = scan(get_neighbours_scan, [window, ind], r)
    i, j = jnp.transpose(i, (1, 0, 2)).reshape(2, -1), jnp.transpose(j, (1, 0, 2)).reshape(2, -1)
    w = jnp.array([weights]*N_astrocytes**2).reshape(-1,)
    mask = get_mask(j, N_astrocytes)
    w = w[mask]
    i, j = i[:, mask], j[:, mask]
    i_lex = lex(i[0], i[1], N_astrocytes)
    j_lex = lex(j[0], j[1], N_astrocytes)
    diff_connectivity = jnp.stack([i_lex, j_lex], axis=1)
    return diff_connectivity, w, coords.T

def get_tripartite_connectivity(N_astrocytes, N_neurons, connectivity):
    ind = jnp.arange(N_neurons)
    ind = jnp.stack(jnp.meshgrid(ind, ind, indexing="ij")).reshape(2, -1)
    
    x_astrocytes = jnp.linspace(0, 1, N_astrocytes)
    x_neurons = jnp.linspace(0, 1, N_neurons)
    D = ((x_neurons.reshape(-1, 1) - x_astrocytes.reshape(1, -1))**2)
    n_a = jnp.argmin(D, axis=1)
    neuron_to_astrocyte = lex(n_a[ind[0]], n_a[ind[1]], N_astrocytes)
    ind_tripartite = neuron_to_astrocyte[connectivity[:, 1]]
    return ind_tripartite