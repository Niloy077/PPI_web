from Bio.PDB import PDBParser

def pdb_to_fasta(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    sequence = ""

    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
    }

    for chain in structure.get_chains():
        print(f"Processing Chain: {chain.id}")  # Debugging
        for residue in chain.get_residues():
            res_name = residue.get_resname()
            print(f"Residue: {res_name}")  # Debugging
            if res_name in three_to_one:
                sequence += three_to_one[res_name]

    return sequence

# Test
pdb_path = "1a3n.pdb"  # Replace with your actual file
fasta_seq = pdb_to_fasta(pdb_path)
print("Extracted FASTA Sequence:\n", fasta_seq)
