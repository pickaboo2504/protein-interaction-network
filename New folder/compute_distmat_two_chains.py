import numpy as np

def compute_distmat_two_chains(chain1_atoms, chain2_atoms):
    """
    Compute the distance matrix between atoms of two different chains.

    Parameters:
    ===========
    - chain1_atoms: DataFrame of atoms in chain 1
    - chain2_atoms: DataFrame of atoms in chain 2

    Returns:
    ========
    - distmat: 2D numpy array containing distances between each atom in chain 1 and each atom in chain 2
    """
    coords1 = chain1_atoms[['x_coord', 'y_coord', 'z_coord']].values
    coords2 = chain2_atoms[['x_coord', 'y_coord', 'z_coord']].values

    distmat = np.sqrt(np.sum((coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :])**2, axis=2))
    return distmat
