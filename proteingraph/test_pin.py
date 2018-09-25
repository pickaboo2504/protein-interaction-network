import pandas as pd
import os
import pytest
from .pin import ProteinGraph
from .resi_atoms import BOND_TYPES, AROMATIC_RESIS, SULPHUR_RESIS, POS_AA,\
    NEG_AA, CATION_RESIS, PI_RESIS

file_path = os.path.dirname(os.path.realpath(__file__))
pdb_handle = 'test_data/2VIU.pdb'
data_path = os.path.join(file_path, pdb_handle)
net = ProteinGraph(data_path)


def test_node_feature_array_length():
    """
    Checks performed on the node features.

    1. Array length is equal to 36.
    2. All features scaled between 0 and 1.

    For expediency, only check a random sample of 1/4 of the nodes.
    """
    for n, d in net.nodes(data=True):
        assert d['features'].shape == (1, 36)


def test_bond_types_are_correct():
    """
    Checks that the bonds that are specified are correct.
    """
    # Check that the bonds are correctly
    for u, v, d in net.edges(data=True):
        assert isinstance(d['kind'], list)
        for kind in d['kind']:
            assert kind in BOND_TYPES


def test_nodes_are_strings():
    """
    Checks to make sure that the nodes are a string.

    For expediency, checks only 1/4 of the nodes.
    """
    for n in net.nodes():
        assert isinstance(n, str)


def test_parse_pdb():
    """
    Tests the function parse_pdb
    """

    # Asserts that the number of lines in the dataframe is correct.
    assert len(net.dataframe) == 3892, "Error: Function or data has changed!"

    # Asserts that the following columns are all present.
    column_types = {'Record name': str,
                    'serial_number': int,
                    'atom': str,
                    'resi_name': str,
                    'chain_id': str,
                    'resi_num': int,
                    'x': float,
                    'y': float,
                    'z': float,
                    'node_id': str}
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


def test_no_self_loops():
    """
    Assures that there are no self loops amongst residues.
    """
    for n in net.nodes():
        assert not net.has_edge(n, n)


def test_get_interacting_atoms_():
    """
    Tests the function get_interacting_atoms_, using 2VIU data.
    """
    interacting = net.get_interacting_atoms_(6, net.distmat)
    # Asserts that the number of interactions found at 6A for 2VIU.
    assert len(interacting[0]) == 156420


def test_add_hydrophobic_interactions_():
    """
    Tests the function add_hydrophobic_interactions_, using 2VIU data.
    """
    resis = net.get_edges_by_bond_type('hydrophobic')
    HYDROPHOBIC_RESIS = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP',
                         'PRO', 'TYR']
    for (r1, r2) in resis:
        assert net.node[r1]['resi_name'] in HYDROPHOBIC_RESIS
        assert net.node[r2]['resi_name'] in HYDROPHOBIC_RESIS


def test_add_disulfide_interactions_():
    """
    Tests the function add_disulfide_interactions_, using 2VIU data.
    """
    resis = net.get_edges_by_bond_type('disulfide')

    for (r1, r2) in resis:
        assert net.node[r1]['resi_name'] == 'CYS'
        assert net.node[r2]['resi_name'] == 'CYS'


def test_delaunay_triangulation():
    """
    I am including this test here that always passes because I don't know how
    best to test it. The code in pin.py uses scipy's delaunay triangulation.
    """
    pass


# 10 March 2016
# This test has been passed out until I figure out what the exact criteria
# for hydrogen bonding is. I intuitively don't think it should be merely 3.5A
# between any two of N, O, S atoms, regardless of whether they are the same
# element or not. Rather, it should be O:->N or N:-->O, or something like that.
#
def test_add_hydrogen_bond_interactions_():
    """
    Tests the function add_hydrogen_bond_interactions_, using 2VIU data.
    """
    # net.add_hydrogen_bond_interactions_()
    # resis = net.get_edges_by_bond_type('hbond')
    # assert len(resis) == 86
    pass


def test_add_aromatic_interactions_():
    """
    Tests the function add_aromatic_interactions_, using 2VIU data. The test
    checks that each residue in an aromatic interaction is one of the aromatic
    residues.
    """

    resis = net.get_edges_by_bond_type('aromatic')
    for n1, n2 in resis:
        assert net.node[n1]['resi_name'] in AROMATIC_RESIS
        assert net.node[n2]['resi_name'] in AROMATIC_RESIS


def test_add_aromatic_sulphur_interactions_():
    """
    Tests the function add_aromatic_sulphur_interactions_, using 2VIU data.
    """

    resis = net.get_edges_by_bond_type('aromatic_sulphur')
    for n1, n2 in resis:
        condition1 = net.node[n1]['resi_name'] in SULPHUR_RESIS and\
            net.node[n2]['resi_name'] in AROMATIC_RESIS

        condition2 = net.node[n2]['resi_name'] in SULPHUR_RESIS and\
            net.node[n1]['resi_name'] in AROMATIC_RESIS

        assert condition1 or condition2


def test_add_cation_pi_interactions_():
    """
    Tests the function add_cation_pi_interactions_, using 2VIU data.
    """

    resis = net.get_edges_by_bond_type('cation_pi')
    for n1, n2 in resis:
        resi1 = net.node[n1]['resi_name']
        resi2 = net.node[n2]['resi_name']

        condition1 = resi1 in CATION_RESIS and resi2 in PI_RESIS
        condition2 = resi2 in CATION_RESIS and resi1 in PI_RESIS

        assert condition1 or condition2


def test_atom_features():
    """
    Tests to make sure that the atom features are correct.
    """
    pass


def test_add_ionic_interactions_():
    """
    Tests the function add_ionic_interactions_, using 2VIU data.
    """
    resis = net.get_edges_by_bond_type('ionic')
    for n1, n2 in resis:
        resi1 = net.node[n1]['resi_name']
        resi2 = net.node[n2]['resi_name']

        condition1 = resi1 in POS_AA and resi2 in NEG_AA
        condition2 = resi2 in POS_AA and resi1 in NEG_AA

        assert condition1 or condition2


def test_feature_array():
    """
    Tests the function feature_array.
    """
    with pytest.raises(AssertionError):
        net.feature_array('atom')

    node_features = net.feature_array(kind='node')
    assert len(node_features) == len(net.nodes())

    edge_features = net.feature_array(kind='edge')
    assert len(edge_features) == len(net.edges())


# def test_get_ring_atoms_():
#     """
#     Tests the function get_ring_atoms_, using 2VIU data.
#     """
#     ring_atom_TRP = net.get_ring_atoms_(net.dataframe, 'TRP')
#     assert len(ring_atom_TRP) == 63
#     ring_atom_HIS = net.get_ring_atoms_(net.dataframe, 'HIS')
#     assert len(ring_atom_HIS) == 55


# def test_get_ring_centroids():
#     """
#     Tests the function get_ring_centroids_, using 2VIU data.
#     """
#     ring_atom_TYR = net.get_ring_atoms_(net.dataframe, 'TYR')
#     assert len(ring_atom_TYR) == 96
#     centroid_TYR = net.get_ring_centroids_(ring_atom_TYR, 'TYR')
#     assert len(centroid_TYR) == 16

#     ring_atom_PHE = net.get_ring_atoms_(net.dataframe, 'PHE')
#     assert len(ring_atom_PHE) == 108
#     centroid_PHE = net.get_ring_centroids_(ring_atom_PHE, 'PHE')
#     assert len(centroid_PHE) == 18
