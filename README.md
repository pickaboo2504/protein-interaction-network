# protein-graph

Computes a molecular graph for protein structures.

## why?

Proteins fold into 3D structures, and have a natural graph representation: amino acids are nodes, and biochemical interactions are edges.

I wrote this package as part of a larger effort to do graph convolutional neural networks on protein structures (represented as graphs). However, that's not the only thing I can foresee doing with this.

One may be interested in the topology of proteins across species and over evolutionary time. This package can aid in answering this question.


## how do I use this package?

This package assumes that you have a standard protein structure file (e.g. a PDB file). This may be a file generated after solving the NMR or crystal structure of a protein, or it may be generated from homology modelling.

Once that has been generated, the molecular graph can be generated using Python code.

```python
from pin import ProteinInteractionNetwork

p = pin.ProteinInteractionNetwork('my_model.pdb')
```

Because the `ProteinInteractionNetwork` class inherits from NetworkX's `Graph` class, all methods that `Graph` has are inherited by `ProteinInteractionNetwork`, and it behaves just as a NetworkX graph does.

What this means is that all graph-theoretic metrics (e.g. degree centrality, betweenness centrality etc.) can be computed on the `ProteinInteractionNetwork` object.

See the HIV1 homology model example in the `examples/` directory for a minimal example.
