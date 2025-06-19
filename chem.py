from rdkit import Chem
from rdkit.Chem import AllChem

# 将SMILES表达式转换为分子对象
def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return mol

# 将分子图转换为基团贡献的切分
def fragment_molecule(mol):
    # 计算分子的基团贡献
    contributions = AllChem.FragmentOnSomeBonds(mol, list(mol.GetBonds()))

    # 输出分子基团贡献的结果
    fragments = []
    for mol_frag in contributions:
        fragments.append(Chem.MolToSmiles(mol_frag))

    return fragments

# 输入你的SMILES表达式
input_smiles = "在这里输入你的SMILES表达式"

# 将SMILES转换为分子对象
molecule = smiles_to_mol(input_smiles)

# 进行基团贡献的切分
resulting_fragments = fragment_molecule(molecule)

# 输出切分后的基团列表
print("基团列表:")
for idx, fragment in enumerate(resulting_fragments, start=1):
    print(f"基团 {idx}: {fragment}")
