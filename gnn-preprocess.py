import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

# Load DrugBank data or SMILES strings
# Example: data = pd.read_csv("drugbank_data.csv")
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Example SMILES

# Convert SMILES to molecular graph
mol = Chem.MolFromSmiles(smiles)

# Generate molecular fingerprints
fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

# Extract molecular descriptors
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
tpsa = Descriptors.TPSA(mol)

# Create a dictionary of molecular features
mol_features = {
    "Molecular Weight": mw,
    "LogP": logp,
    "TPSA": tpsa,
    "Fingerprint": fingerprint
}

# Output features
print(mol_features)
