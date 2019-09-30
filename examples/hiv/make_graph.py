import networkx as nx

from proteingraph import ProteinGraph

"""
Makes a graph out of a PDB file and saves the graph as a NetworkX pickle.
"""

p = ProteinGraph("hiv1_homology_model.pdb")
nx.write_gpickle(p, "hiv1_homology_model.pkl")
