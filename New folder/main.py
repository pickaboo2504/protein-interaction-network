import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pin import (
    compute_chain_pos_aa_mapping,
    compute_distmat,
    get_interacting_atoms,
    get_ring_atoms,
    get_ring_centroids,
    pdb2df,
    read_pdb,
)
from resi_atoms import (
    AROMATIC_RESIS,
    BOND_TYPES,
    CATION_RESIS,
    HYDROPHOBIC_RESIS,
    NEG_AA,
    PI_RESIS,
    POS_AA,
    RESI_NAMES,
    SULPHUR_RESIS,
)

class ProteinInteractionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Protein Interaction Network")

        self.data_path = None  # Will hold the path to the selected PDB file
        self.net = None  # Will hold the protein interaction network
        self.pdb_df = None  # Will hold the DataFrame from PDB file
        self.distmat = None  # Will hold the distance matrix

        self.create_widgets()

    def create_widgets(self):
        # Frame to hold file selection and action buttons
        frame_file = tk.Frame(self.root)
        frame_file.pack(pady=10)

        # Label for file path display
        self.lbl_file = tk.Label(frame_file, text="No file selected")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        # Button to select PDB file
        btn_browse = tk.Button(frame_file, text="Browse", command=self.browse_file)
        btn_browse.pack(side=tk.LEFT, padx=10)

        # Button to start analysis
        btn_analyze = tk.Button(frame_file, text="Analyze", command=self.analyze_data)
        btn_analyze.pack(side=tk.LEFT, padx=10)

        # Frame to hold printed statements
        frame_output = tk.Frame(self.root)
        frame_output.pack(pady=10)

        # Text widget to display printed statements
        self.text_output = tk.Text(frame_output, height=10, width=80)
        self.text_output.pack()

        # Button to show graph
        btn_show_graph = tk.Button(self.root, text="Show Graph", command=self.show_graph)
        btn_show_graph.pack(pady=10)

        # Button to show 3D structure
        btn_show_3d = tk.Button(self.root, text="Show 3D Structure", command=self.show_3d_structure)
        btn_show_3d.pack(pady=10)

    def browse_file(self):
        self.data_path = filedialog.askopenfilename(filetypes=[("PDB files", "*.pdb")])
        self.lbl_file.config(text=self.data_path)

    def analyze_data(self):
        if not self.data_path:
            messagebox.showerror("Error", "Please select a PDB file first.")
            return

        try:
            # Read PDB file into a DataFrame
            self.pdb_df = pdb2df(self.data_path)

            # Construct protein interaction network
            self.net = read_pdb(self.data_path)

            # Compute distance matrix
            self.distmat = compute_distmat(self.pdb_df)

            # Example usage of functions
            interacting_atoms = get_interacting_atoms(6, self.distmat)
            ring_atoms_TYR = get_ring_atoms(self.pdb_df, "TYR")
            ring_centroids_TYR = get_ring_centroids(ring_atoms_TYR)

            # Display results in text widget
            self.text_output.delete(1.0, tk.END)  # Clear previous output
            self.text_output.insert(tk.END, f"Number of interacting atoms at 6A: {len(interacting_atoms[0])}\n")
            self.text_output.insert(tk.END, f"Number of ring atoms in TYR: {len(ring_atoms_TYR)}\n")
            self.text_output.insert(tk.END, f"Number of ring centroids in TYR: {len(ring_centroids_TYR)}\n")

        except Exception as e:
            messagebox.showerror("Error", f"Error processing PDB file: {str(e)}")

    def show_graph(self):
        if not self.net:
            messagebox.showerror("Error", "Please analyze data first.")
            return

        # Visualize the network graph
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.net)
        nx.draw(self.net, pos, with_labels=True, node_color='green', node_size=80, edge_color='blue', linewidths=5,
                font_size=6, font_weight='bold')
        plt.title("Protein Interaction Network")
        plt.show()

    def show_3d_structure(self):
        if self.pdb_df is None:
            messagebox.showerror("Error", "Please analyze data first.")
            return

        # Extract the coordinates
        x_coords = self.pdb_df["x_coord"]
        y_coords = self.pdb_df["y_coord"]
        z_coords = self.pdb_df["z_coord"]

        # Create 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x_coords, y_coords, z_coords)

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')

        plt.title("3D Structure of Protein")
        plt.show()

def main():
    root = tk.Tk()
    app = ProteinInteractionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
