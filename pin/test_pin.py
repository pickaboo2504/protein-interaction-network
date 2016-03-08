import pandas as pd
from pin import ProteinInteractionNetwork
import sys
import os

pdb_handle = os.path.join(sys.path[0], 'test_data/2VIU.pdb')
net = ProteinInteractionNetwork(pdb_handle)

BOND_TYPES = ['hydrophobic', 'disulfide', 'hbond', 'ionic', 'aromatic',
              'aromatic_sulphur', 'cation_pi', 'backbone']

AROMATIC_RESIS = ['PHE', 'TRP', 'HIS', 'TYR']


def test_data_integrity():
    """
    A series of data and data structure integrity tests.
    """

    for u, v, d in net.edges(data=True):
        assert isinstance(d['kind'], set)
        for kind in d['kind']:
            assert kind in BOND_TYPES

def test_parse_pdb():
    """
    Tests the function parse_pdb
    """

    # Asserts that the number of lines in the dataframe is correct.
    assert len(net.dataframe) == 3892, "Error: Function or data has changed!"

    # Asserts that the number of columns is correct
    assert len(net.dataframe.columns) == 9, \
        "Error: There should be only 9 columns in the DataFrame!"

    # Asserts that the following columns are all present.
    column_types = {'Record name': str,
                    'serial_number': int,
                    'atom': str,
                    'resi_name': str,
                    'chain_id': str,
                    'resi_num': int,
                    'x': float,
                    'y': float,
                    'z': float}
    for c in column_types.keys():
        assert c in net.dataframe.columns, \
            "{0} not present in DataFrame columns!".format(c)


def test_compute_distmat():
    """
    Tests the function compute_distmat, using dummy data.
    """
    data = list()
    for i in range(1, 2):
        d = dict()
        d['idx'] = i
        d['x'] = i
        d['y'] = i
        d['z'] = i
        data.append(d)
    df = pd.DataFrame(data)
    distmat = net.compute_distmat(df)

    # Asserts that the shape is correct.
    assert distmat.shape == (len(data), len(data))


def test_get_interacting_atoms_():
    """
    Tests the function get_interacting_atoms_, using 2VIU data.
    """
    interacting = net.get_interacting_atoms_(6, net.distmat)
    # Asserts that the number of interactions found at 6A for 2VIU.
    assert len(interacting[0]) == 156420

def get_edges_by_bond_type(net, bond_type):
    """
    Parameters:
    ===========
    - net: the protein interaction graph
    - bond_type: (str) one of the elements in the variable BOND_TYPES

    Returns:
    ========
    - resis: (list) a list of tuples, where each tuple is an edge.
    """

    resis = []
    for n1, n2, d in net.edges(data=True):
        if bond_type in d['kind']:
            resis.append((n1, n2))
    return resis

def test_add_hydrophobic_interactions_():
    """
    Tests the function add_hydrophobic_interactions_, using 2VIU data.
    """
    net.add_hydrophobic_interactions_()
    resis = get_edges_by_bond_type(net, 'hydrophobic')
    assert len(resis) == 287


def test_add_disulfide_interactions_():
    """
    Tests the function add_disulfide_interactions_, using 2VIU data.
    """
    net.add_disulfide_interactions_()
    resis = get_edges_by_bond_type(net, 'disulfide')
    assert len(resis) == 6


def test_add_hydrogen_bond_interactions_():
    """
    Tests the function add_hydrogen_bond_interactions_, using 2VIU data.
    """
    net.add_hydrogen_bond_interactions_()
    resis = get_edges_by_bond_type(net, 'hbond')
    assert len(resis) == 86


def test_add_ionic_interactions_():
    """
    Tests the function add_ionic_interactions_, using 2VIU data.
    """
    net.add_ionic_interactions_()
    resis = get_edges_by_bond_type(net, 'ionic')
    assert len(resis) == 96

def test_add_aromatic_interactions_():
    """
    Tests the function add_aromatic_interactions_, using 2VIU data.
    """
    net.add_aromatic_interactions_()
    aromatic_resis = get_edges_by_bond_type(net, 'aromatic')
    for n1, n2 in aromatic_resis:
        assert net.node[n1]['aa'] in AROMATIC_RESIS
        assert net.node[n2]['aa'] in AROMATIC_RESIS

    assert len(aromatic_resis) == 34

def test_get_ring_atoms_():
    """
    Tests the function get_ring_atoms_, using 2VIU data.
    """

    ring_atom_TRP = net.get_ring_atoms_(net.dataframe, 'TRP')
    assert len(ring_atom_TRP) == 63
    ring_atom_HIS = net.get_ring_atoms_(net.dataframe, 'HIS')
    assert len(ring_atom_HIS) == 55


def test_get_ring_centroids():
    """
    Tests the function get_ring_centroids_, using 2VIU data.
    """
    ring_atom_TYR = net.get_ring_atoms_(net.dataframe, 'TYR')
    assert len(ring_atom_TYR) == 96
    centroid_TYR = net.get_ring_centroids_(ring_atom_TYR, 'TYR')
    assert len(centroid_TYR) == 16

    ring_atom_PHE = net.get_ring_atoms_(net.dataframe, 'PHE')
    assert len(ring_atom_PHE) == 108
    centroid_PHE = net.get_ring_centroids_(ring_atom_PHE, 'PHE')
    assert len(centroid_PHE) == 18
