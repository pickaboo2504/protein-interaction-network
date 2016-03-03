import pandas as pd
from pin import ProteinInteractionNetwork

pdb_handle = 'test_data/2VIU.pdb'
net = ProteinInteractionNetwork(pdb_handle)
def test_parse_pdb():
	"""
	Tests the function parse_pdb
	"""
	# Asserts that the number of lines in the dataframe is correct.
	assert len(net.dataframe) == 3892, "Error: Function or data has changed!"

	# Asserts that the number of columns is correct
	assert len(net.dataframe.columns) == 9, "Error: There should be only 9 columns in the DataFrame!"

	# Asserts that the following columns are all present.
	column_types = {'Record name':str,
			'serial_number':int,
			'atom':str,
			'resi_name':str, 
			'chain_id':str, 
			'resi_num':int, 
			'x':float, 
			'y':float, 
			'z':float}
	for c in column_types.keys():
		assert c in net.dataframe.columns, "{0} not present in DataFrame columns!".format(c)

def test_compute_distmat():
	"""
	Tests the function compute_distmat, using dummy data.
	"""
	data = list()
	for i in range(1,2):
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

def test_get_interacting_atoms():
	"""
	Tests the function get_interacting_atoms, using 2VIU data.
	"""
	interacting = net.get_interacting_atoms(6, net.distmat)
	# Asserts that the number of interactions found at 6A for 2VIU 
	assert len(interacting[0]) == 156420

def test_get_hydrophobic_interactions():
	"""
	Tests the function get_hydrophobic_interactions, using 2VIU data.
	"""
	resis = net.get_hydrophobic_interactions()
	assert len(resis) == 574

def test_get_disulfide_interactions():
	"""
	Tests the function get_disulfide_interactions, using 2VIU data.
	"""
	resis = net.get_disulfide_interactions()
	assert len(resis) == 12

def test_get_hbond_interactions():
	"""
	Tests the function get_hydrogen_bond_interactions, using 2VIU data.
	"""
	resis = net.get_hydrogen_bond_interactions()
	assert len(resis) == 172

def test_get_ionic_interactions():
	"""
	Tests the function get_ionic_interactions, using 2VIU data.
	"""
	resis = net.get_ionic_interactions()
	assert len(resis) == 192

def test_get_ring_atoms():
	"""
	Tests the function get_ring_atoms, using 2VIU data.
	"""

	ring_atom_TRP = net.get_ring_atoms(net.dataframe, 'TRP')
	assert len(ring_atom_TRP) == 63
	ring_atom_HIS = net.get_ring_atoms(net.dataframe, 'HIS')
	assert len(ring_atom_HIS) == 55

def test_get_ring_centroids_and_centroids():
	"""
	Tests the function get_ring_centroids, using 2VIU data.
	"""
	ring_atom_TYR = net.get_ring_atoms(net.dataframe, 'TYR')
	assert len(ring_atom_TYR) == 96
	centroid_TYR = net.get_ring_centroids(ring_atom_TYR, 'TYR')
	assert len(centroid_TYR) == 16

	ring_atom_PHE = net.get_ring_atoms(net.dataframe, 'PHE')
	assert len(ring_atom_PHE) == 108
	centroid_PHE = net.get_ring_centroids(ring_atom_PHE, 'PHE')
	assert len(centroid_PHE) == 18

def test_get_aromatic_interactions():
	"""
	Tests the function get_aromatic_interactions, using 2VIU data.
	"""
	aromatic_resis = net.get_aromatic_interactions()
	assert len(aromatic_resis) == 34
	print(aromatic_resis)


