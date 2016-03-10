"""
Author: Eric J. Ma
License: MIT

A Python module that computes the protein interaction graph from a PDB file.
"""

import pandas as pd
import numbers
import numpy as np
import networkx as nx

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelBinarizer
from resi_atoms import BACKBONE_ATOMS, AMINO_ACIDS, BOND_TYPES, RESI_NAMES,\
    HYDROPHOBIC_RESIS, DISULFIDE_RESIS, DISULFIDE_ATOMS, AA_RING_ATOMS,\
    IONIC_RESIS, POS_AA, NEG_AA, AROMATIC_RESIS, CATION_PI_RESIS,\
    CATION_RESIS, PI_RESIS


class ProteinInteractionNetwork(nx.Graph):
    """
    The ProteinInteractionNetwork object.

    Inherits from the NetworkX Graph object.

    Implements further functions for automatically computing the graph
    structure.

    Certain functions are available for integration with the
    neural-fingerprint Python package.
    """
    def __init__(self, pdb_handle):
        super(ProteinInteractionNetwork, self).__init__()
        self.pdb_handle = pdb_handle
        self.dataframe = self.parse_pdb()
        self.distmat = self.compute_distmat(self.dataframe)
        self.rgroup_df = self.get_rgroup_dataframe_()
        # Automatically compute the interaction graph upon loading.
        self.compute_interaction_graph()

    def compute_interaction_graph(self):
        """
        Computes the interaction graph.

        Graph definition and metadata:
        ==============================
        - Node: Amino acid position.
            - aa: amino acid identity

        - Edge: Any interaction found by the atomic interaction network.
            - hbond:            BOOLEAN
            - disulfide:        BOOLEAN
            - hydrophobic:      BOOLEAN
            - ionic:            BOOLEAN
            - aromatic:         BOOLEAN
            - aromatic_sulphur: BOOLEAN
            - cation_pi:        BOOLEAN
        """
        # Populate nodes, which are amino acid positions, and have metadata
        nums_and_names = set(zip(self.dataframe['resi_num'],
                                 self.dataframe['resi_name'],
                                 self.dataframe['chain_id']))

        for r, d in self.dataframe.iterrows():
            self.add_node(d['node_id'],
                          chain_id=d['chain_id'],
                          resi_num=d['resi_num'],
                          resi_name=d['resi_name'],
                          features=None)

        # 10 March 2016
        # I currently do not have a good way of telling whether two amino
        # acids are covalently bonded to one another in the backbone sequence.
        # The particular problem I am most concerned with is the scenario
        # where we leave one chain and go to the next.
        #
        # # Add in edges for amino acids that are adjacent in the linear amino
        # # acid sequence.
        # nodes1 = self.nodes()[0:-1]
        # nodes2 = self.nodes()[1:]
        # for n1, n2 in zip(nodes1, nodes2):
        #     self.add_edge(n1, n2, kind='backbone')

        # Define function shortcuts for each of the interactions.
        funcs = dict()
        funcs['hydrophobic'] = self.add_hydrophobic_interactions_
        funcs['disulfide'] = self.add_disulfide_interactions_
        funcs['hbond'] = self.add_hydrogen_bond_interactions_
        funcs['ionic'] = self.add_ionic_interactions_
        funcs['aromatic'] = self.add_aromatic_interactions_
        funcs['aromatic_sulphur'] = self.add_aromatic_sulphur_interactions_
        funcs['cation_pi'] = self.add_cation_pi_interactions_

        # Add in each type of edge, based on the above.
        for k, v in funcs.items():
            v()

    def parse_pdb(self):
        """
        Parses the PDB file as a pandas DataFrame object.

        Backbone chain atoms are ignored for the calculation
        of interacting residues.
        """
        atomic_data = []
        with open(self.pdb_handle, 'r') as f:
            for line in f.readlines():
                data = dict()
                if line[0:4] == 'ATOM':

                    data['Record name'] = line[0:5].strip(' ')
                    data['serial_number'] = int(line[6:11].strip(' '))
                    data['atom'] = line[12:15].strip(' ')
                    data['resi_name'] = line[17:20]
                    data['chain_id'] = line[21]
                    data['resi_num'] = int(line[23:26])
                    data['x'] = float(line[30:37])
                    data['y'] = float(line[38:45])
                    data['z'] = float(line[46:53])

                    atomic_data.append(data)

        atomic_df = pd.DataFrame(atomic_data)
        atomic_df['node_id'] = atomic_df['chain_id'] + \
            atomic_df['resi_num'].map(str) + \
            atomic_df['resi_name']

        return atomic_df

    def compute_distmat(self, dataframe):
        """
        Computes the pairwise euclidean distances between every atom.

        Design choice: passed in a DataFrame to enable easier testing on
        dummy data.
        """

        self.eucl_dists = pdist(dataframe[['x', 'y', 'z']],
                                metric='euclidean')
        self.eucl_dists = pd.DataFrame(squareform(self.eucl_dists))
        self.eucl_dists.index = dataframe.index
        self.eucl_dists.columns = dataframe.index

        return self.eucl_dists

    def get_interacting_atoms_(self, angstroms, distmat):
        """
        Finds the atoms that are within a particular radius of one another.
        """
        return np.where(distmat <= angstroms)

    def add_interacting_resis_(self, interacting_atoms, dataframe, kind):
        """
        Returns a list of 2-tuples indicating the interacting residues based
        on the interacting atoms. This is most typically called after the
        get_interacting_atoms_ function above.

        Also filters out the list such that the residues have to be at least
        two apart.

        Parameters:
        ===========
        - interacting_atoms:    (numpy array) result from
                                get_interacting_atoms_ function.
        - dataframe:            (pandas dataframe) a pandas dataframe that
                                houses the euclidean locations of each atom.
        - kind:                 (list) the kind of interaction. Contains one
                                of :
                                - hydrophobic
                                - disulfide
                                - hbond
                                - ionic
                                - aromatic
                                - aromatic_sulphur
                                - cation_pi

        Returns:
        ========
        - filtered_interacting_resis: (set of tuples) the residues that are in
                                      an interaction, with the interaction kind
                                      specified

        """
        # This assertion/check is present for defensive programming!
        for k in kind:
            assert k in BOND_TYPES

        resi1 = dataframe.ix[interacting_atoms[0]]['node_id'].values
        resi2 = dataframe.ix[interacting_atoms[1]]['node_id'].values

        interacting_resis = set(list(zip(resi1, resi2)))
        filtered_interacting_resis = set()
        for i1, i2 in interacting_resis:
            if i1 != i2:
                if self.has_edge(i1, i2):
                    for k in kind:
                        self.edge[i1][i2]['kind'].add(k)
                else:
                    self.add_edge(i1, i2, {'kind': set(kind)})

        # return filtered_interacting_resis

    def get_rgroup_dataframe_(self):
        """
        Returns just the atoms that are amongst the R-groups and not part of
        the backbone chain.
        """

        rgroup_df = self.filter_dataframe(self.dataframe,
                                          'atom',
                                          BACKBONE_ATOMS,
                                          False)
        return rgroup_df

    def filter_dataframe(self, dataframe, by_column, list_of_values, boolean):
        """
        Filters the [dataframe] such that the [by_column] values have to be
        in the [list_of_values] list if boolean == True, or not in the list
        if boolean == False
        """
        df = dataframe.copy()
        df = df[df[by_column].isin(list_of_values) == boolean]
        df.reset_index(inplace=True, drop=True)

        return df

    # SPECIFIC INTERACTION FUNCTIONS #
    def add_hydrophobic_interactions_(self):
        """
        Finds all hydrophobic interactions between the following residues:
        ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO, TYR

        Criteria: R-group residues are within 5A distance.
        """
        hydrophobics_df = self.filter_dataframe(self.rgroup_df,
                                                'resi_name',
                                                HYDROPHOBIC_RESIS,
                                                True)
        distmat = self.compute_distmat(hydrophobics_df)
        interacting_atoms = self.get_interacting_atoms_(5, distmat)
        interacting_resis = self.add_interacting_resis_(interacting_atoms,
                                                        hydrophobics_df,
                                                        ['hydrophobic'])

        return interacting_resis

    def add_disulfide_interactions_(self):
        """
        Finds all disulfide interactions between CYS residues, such that the
        sulfur atom pairs are within 2.2A of each other.
        """

        disulfide_df = self.filter_dataframe(self.rgroup_df,
                                             'resi_name',
                                             DISULFIDE_RESIS,
                                             True)
        disulfide_df = self.filter_dataframe(disulfide_df,
                                             'atom',
                                             DISULFIDE_ATOMS,
                                             True)
        distmat = self.compute_distmat(disulfide_df)
        interacting_atoms = self.get_interacting_atoms_(2.2, distmat)
        interacting_resis = self.add_interacting_resis_(interacting_atoms,
                                                        disulfide_df,
                                                        ['disulfide'])

        return interacting_resis

    def add_hydrogen_bond_interactions_(self):
        """
        Finds all hydrogen-bond interactions between atoms capable of hydrogen
        bonding.
        """
        # For these atoms, find those that are within 3.5A of one another.
        HBOND_ATOMS = ['ND', 'NE', 'NH', 'NZ', 'OD', 'OE', 'OG', 'OH', 'SD',
                       'SG', 'N', 'O']
        hbond_df = self.filter_dataframe(self.rgroup_df,
                                         'atom',
                                         HBOND_ATOMS,
                                         True)
        distmat = self.compute_distmat(hbond_df)
        interacting_atoms = self.get_interacting_atoms_(3.5, distmat)
        interacting_resis = self.add_interacting_resis_(interacting_atoms,
                                                        hbond_df,
                                                        ['hbond'])

        # For these atoms, find those that are within 4.0A of one another.
        HBOND_ATOMS_SULPHUR = ['SD', 'SG']
        hbond_df = self.filter_dataframe(self.rgroup_df,
                                         'atom',
                                         HBOND_ATOMS_SULPHUR,
                                         True)
        distmat = self.compute_distmat(hbond_df)
        interacting_atoms = self.get_interacting_atoms_(4.0, distmat)
        interacting_resis = self.add_interacting_resis_(interacting_atoms,
                                                        hbond_df,
                                                        ['hbond'])

    def add_ionic_interactions_(self):
        """
        Finds all ionic interactiosn between ARG, LYS, HIS, ASP, and GLU.
        Distance cutoff: 6A.
        """
        ionic_df = self.filter_dataframe(self.rgroup_df,
                                         'resi_name',
                                         IONIC_RESIS,
                                         True)
        distmat = self.compute_distmat(ionic_df)
        interacting_atoms = self.get_interacting_atoms_(6, distmat)

        self.add_interacting_resis_(interacting_atoms, ionic_df, ['ionic'])

        # Check that the interacting residues are of opposite charges
        for r1, r2 in self.get_edges_by_bond_type('ionic'):
            condition1 = self.node[r1]['resi_name'] in POS_AA and \
                self.node[r2]['resi_name'] in NEG_AA

            condition2 = self.node[r2]['resi_name'] in POS_AA and \
                self.node[r1]['resi_name'] in NEG_AA

            if not condition1 or condition2:
                self.remove_edge(r1, r2)

    def add_aromatic_interactions_(self):
        """
        Finds all aromatic-aromatic interactions by looking for phenyl ring
        centroids separated between 4.5A to 7A.

        Phenyl rings are present on PHE, TRP, HIS and TYR.

        Phenyl ring atoms on these amino acids are defined by the following
        atoms:
        - PHE: CG, CD, CE, CZ
        - TRP: CD, CE, CH, CZ
        - HIS: CG, CD, ND, NE, CE
        - TYR: CG, CD, CE, CZ

        Centroids of these atoms are taken by taking:
            (mean x), (mean y), (mean z)
        for each of the ring atoms.

        Notes for future self/developers:
        - Because of the requirement to pre-compute ring centroids, we do not
          use the functions written above (filter_dataframe, compute_distmat,
          get_interacting_atoms), as they do not return centroid atom
          euclidean coordinates.
        """
        dfs = []
        for resi in AROMATIC_RESIS:
            resi_rings_df = self.get_ring_atoms_(self.dataframe, resi)
            resi_centroid_df = self.get_ring_centroids_(resi_rings_df, resi)
            dfs.append(resi_centroid_df)

        aromatic_df = pd.concat(dfs)
        aromatic_df.sort_values(by='node_id', inplace=True)
        aromatic_df.reset_index(inplace=True, drop=True)

        distmat = self.compute_distmat(aromatic_df)
        distmat.set_index(aromatic_df['node_id'], inplace=True)
        distmat.columns = aromatic_df['node_id']
        distmat = distmat[(distmat >= 4.5) & (distmat <= 7)].fillna(0)
        indices = np.where(distmat > 0)

        interacting_resis = []
        for i, (r, c) in enumerate(zip(indices[0], indices[1])):
            interacting_resis.append((distmat.index[r], distmat.index[c]))

        for i, (n1, n2) in enumerate(interacting_resis):
            assert self.node[n1]['resi_name'] in AROMATIC_RESIS
            assert self.node[n2]['resi_name'] in AROMATIC_RESIS
            if self.has_edge(n1, n2):
                self.edge[n1][n2]['kind'].add('aromatic')
            else:
                self.add_edge(n1, n2, kind={'aromatic'})

    def get_ring_atoms_(self, dataframe, aa):
        """
        A helper function for add_aromatic_interactions_.

        Gets the ring atoms from the particular aromatic amino acid.

        Parameters:
        ===========
        - dataframe: the dataframe containing the atom records.
        - aa: the amino acid of interest, passed in as 3-letter string.

        Returns:
        ========
        - dataframe: a filtered dataframe containing just those atoms from the
                     particular amino acid selected. e.g. equivalent to
                     selecting just the ring atoms from a particular amino
                     acid.
        """

        ring_atom_df = self.filter_dataframe(dataframe,
                                             'resi_name',
                                             [aa],
                                             True)

        ring_atom_df = self.filter_dataframe(ring_atom_df,
                                             'atom',
                                             AA_RING_ATOMS[aa],
                                             True)
        return ring_atom_df

    def get_ring_centroids_(self, ring_atom_df, aa):
        """
        A helper function for add_aromatic_interactions_.

        Computes the ring centroids for each a particular amino acid's ring
        atoms.

        Ring centroids are computed by taking the mean of the x, y, and z
        coordinates.

        Parameters:
        ===========
        - ring_atom_df: a dataframe computed using get_ring_atoms_.
        - aa: the amino acid under study

        Returns:
        ========
        - centroid_df: a dataframe containing just the centroid coordinates of
                       the ring atoms of each residue.
        """
        centroid_df = ring_atom_df.groupby('node_id')\
                                  .mean()[['x', 'y', 'z']]\
                                  .reset_index()

        return centroid_df

    def add_aromatic_sulphur_interactions_(self):
        """
        Finds all aromatic-sulphur interactions.
        """
        RESIDUES = ['MET', 'CYS', 'PHE', 'TYR', 'TRP']
        SULPHUR_RESIS = ['MET', 'CYS']
        AROMATIC_RESIS = ['PHE', 'TYR', 'TRP']

        aromatic_sulphur_df = self.filter_dataframe(self.rgroup_df,
                                                    'resi_name',
                                                    RESIDUES,
                                                    True)
        distmat = self.compute_distmat(aromatic_sulphur_df)
        interacting_atoms = self.get_interacting_atoms_(5.3, distmat)
        interacting_atoms = zip(interacting_atoms[0], interacting_atoms[1])

        for (a1, a2) in interacting_atoms:
            resi1 = aromatic_sulphur_df.ix[a1]['node_id']
            resi2 = aromatic_sulphur_df.ix[a2]['node_id']

            condition1 = resi1 in SULPHUR_RESIS and resi2 in AROMATIC_RESIS
            condition2 = resi1 in AROMATIC_RESIS and resi2 in SULPHUR_RESIS

            if (condition1 or condition2) and resi1 != resi2:
                if self.has_edge(resi1, resi2):
                    self.edge[resi1][resi2]['kind'].add('aromatic_sulphur')
                else:
                    self.add_edge(resi1,
                                  resi2,
                                  {'kind': {'aromatic_sulphur'}})

    def add_cation_pi_interactions_(self):
        cation_pi_df = self.filter_dataframe(self.rgroup_df,
                                             'resi_name',
                                             CATION_PI_RESIS,
                                             True)
        distmat = self.compute_distmat(cation_pi_df)
        interacting_atoms = self.get_interacting_atoms_(6, distmat)
        interacting_atoms = zip(interacting_atoms[0], interacting_atoms[1])
        interacting_resis = set()

        for (a1, a2) in interacting_atoms:
            resi1 = cation_pi_df.ix[a1]['node_id']
            resi2 = cation_pi_df.ix[a2]['node_id']

            condition1 = resi1 in CATION_RESIS and resi2 in PI_RESIS
            condition2 = resi1 in PI_RESIS and resi2 in CATION_RESIS

            if (condition1 or condition2) and resi1 != resi2:
                if self.has_edge(resi1, resi2):
                    self.edge[resi1][resi2]['kind'].add('cation_pi')
                else:
                    self.add_edge(resi1, resi2, {'kind': {'cation_pi'}})

    def get_edges_by_bond_type(self, bond_type):
        """
        Parameters:
        ===========
        - bond_type: (str) one of the elements in the variable BOND_TYPES

        Returns:
        ========
        - resis: (list) a list of tuples, where each tuple is an edge.
        """

        resis = []
        for n1, n2, d in self.edges(data=True):
            if bond_type in d['kind']:
                resis.append((n1, n2))
        return resis

    def add_node_features(self, node):
        """
        A function that computes one node's features from the data.

        The features are as such:

        - one-of-K encoding for amino acid identity at that node [23 cells]
        - the molecular weight of the amino acid [1 cell]
        - the pKa of the amino acid [1 cell]
        - the node degree, i.e. the number of other nodes it is connected to
          [1 cell] (#nts: not sure if this is necessary.)
        - the sum of all euclidean distances on each edge connecting those
          nodes [1 cell]

        Parameters:
        ===========
        - node:     A node present in the Protein Interaction Network.
        """

        # A defensive programming assertion!
        assert self.has_node(node)

        # Encode the amino acid as a one-of-K encoding.
        lb = LabelBinarizer()
        lb.fit(RESI_NAMES)
        aa = lb.transform(net.node[node]['resi_name'])

        #
