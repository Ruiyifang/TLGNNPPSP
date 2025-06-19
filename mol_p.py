from rdkit.Chem import Draw
from rdkit import Chem
mol = Chem.MolFromSmiles('C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)C(F)(F)C(F)')
Draw.MolToImage(mol, size=(150,150), kekulize=True)
Draw.ShowMol(mol, size=(150,150), kekulize=False)
Draw.MolToFile(mol, 'picture/output.png', size=(150, 150))