import pandas as pd
from pin import ProteinInteractionNetwork
import sys
import os

pdb_handle = os.path.join(sys.path[0], 'test_data/2VIU.pdb')
net = ProteinInteractionNetwork(pdb_handle)

BOND_TYPES = ['hydrophobic', 'disulfide', 'hbond', 'ionic', 'aromatic',
              'aromatic_sulphur', 'cation_pi', 'backbone']

AROMATIC_RESIS = ['PHE', 'TRP', 'HIS', 'TYR']
SULPHUR_RESIS = ['MET', 'CYS']
POS_AA = ['HIS', 'LYS', 'ARG']
NEG_AA = ['GLU', 'ASP']
CATION_RESIS = ['LYS', 'ARG']
PI_RESIS = ['PHE', 'TYR', 'TRP']


def test_bond_types_are_correct():
    """
    Checks that the bonds that are specified are correct.
    """
    # Check that the bonds are correctly
    for u, v, d in net.edges(data=True):
        assert isinstance(d['kind'], set)
        for kind in d['kind']:
            assert kind in BOND_TYPES

def test_nodes_are_strings():
    """
    Checks to make sure that the nodes are a string.
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

# 10 March 2016
# This test has been commented out until I figure out what the exact criteria
# for hydrogen bonding is. I intuitively don't think it should be merely 3.5A
# between any two of N, O, S atoms, regardless of whether they are the same
# element or not. Rather, it should be O:->N or N:-->O, or something like that.
#
# def test_add_hydrogen_bond_interactions_():
#     """
#     Tests the function add_hydrogen_bond_interactions_, using 2VIU data.
#     """
#     net.add_hydrogen_bond_interactions_()
#     resis = net.get_edges_by_bond_type('hbond')
#     assert len(resis) == 86


def test_add_ionic_interactions_():
    """
    Tests the function add_ionic_interactions_, using 2VIU data.
    """
    resis = net.get_edges_by_bond_type('ionic')
    POS_AA = ['HIS', 'LYS', 'ARG']
    NEG_AA = ['GLU', 'ASP']

    for (r1, r2) in resis:
        condition1 = net.node[r1]['resi_name'] in POS_AA and net.node[r2]['resi_name'] in NEG_AA
        condition2 = net.node[r2]['resi_name'] in POS_AA and net.node[r1]['resi_name'] in NEG_AA
        assert condition1 or condition2, print(r1, r2)


def test_add_aromatic_interactions_():
    """
    Tests the function add_aromatic_interactions_, using 2VIU data.
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
