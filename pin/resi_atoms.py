"""
Author: Eric J. Ma

Purpose: This is a set of utility variables and functions that can be used
across the PIN project.
"""

BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']

AMINO_ACIDS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

BOND_TYPES = ['hydrophobic', 'disulfide', 'hbond', 'ionic', 'aromatic',
              'aromatic_sulphur', 'cation_pi', 'backbone']

RESI_NAMES = ['ALA', 'ASX', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE',
              'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR',
              'VAL', 'TRP', 'TYR', 'GLX']

HYDROPHOBIC_RESIS = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO',
                     'TYR']

DISULFIDE_RESIS = ['CYS']

DISULFIDE_ATOMS = ['SG']

IONIC_RESIS = ['ARG', 'LYS', 'HIS', 'ASP', 'GLU']

POS_AA = ['HIS', 'LYS', 'ARG']

NEG_AA = ['GLU', 'ASP']

AA_RING_ATOMS = dict()
AA_RING_ATOMS['PHE'] = ['CG', 'CD', 'CE', 'CZ']
AA_RING_ATOMS['TRP'] = ['CD', 'CE', 'CH', 'CZ']
AA_RING_ATOMS['HIS'] = ['CG', 'CD', 'CE', 'ND', 'NE']
AA_RING_ATOMS['TYR'] = ['CG', 'CD', 'CE', 'CZ']

AROMATIC_RESIS = ['PHE', 'TRP', 'HIS', 'TYR']

CATION_PI_RESIS = ['LYS', 'ARG', 'PHE', 'TYR', 'TRP']

CATION_RESIS = ['LYS', 'ARG']

PI_RESIS = ['PHE', 'TYR', 'TRP']
