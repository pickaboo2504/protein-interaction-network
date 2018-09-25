from pin import pin
import networkx as nx

"""
Makes a graph out of a PDB file and saves the graph as a NetworkX pickle.
"""

p = pin.ProteinInteractionNetwork("hiv1_homology_model.pdb")
nx.write_gpickle(p, "hiv1_homology_model.pkl")
