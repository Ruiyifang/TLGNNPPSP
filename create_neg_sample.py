import copy
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import Recap
from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import BRICS



def is_valid_smiles(smi):
    try:
        Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
    except:
        print("not successfully processed smiles: ", smi)
        return False
    return True


df_target = pd.DataFrame(pd.read_csv('polyinfo_HSP_hz0827.csv'))

smiles_list=df_target['SMILES'].tolist()
labels_D=df_target['D'].tolist()
labels_P=df_target['P'].tolist()
labels_H=df_target['H'].tolist()

#D拟合的好？

labels=[]

smiles_list_v=[]
labels_D_v=[]
labels_P_v=[]
labels_H_v=[]

for D,P,H,smi in tqdm(zip(labels_D, labels_P,labels_H,smiles_list)):
    if not is_valid_smiles(smi): continue

    smiles_list_v.append(smi)
    labels_D_v.append(D)
    labels_P_v.append(P)
    labels_H_v.append(H)





df_target = pd.DataFrame(pd.read_csv('polymer_HSP_HSPiP_hz1010.csv'))

smiles_list=df_target['SMILES'].tolist()
labels_D=df_target['D'].tolist()
labels_P=df_target['P'].tolist()
labels_H=df_target['H'].tolist()


new_smiles_list=[]
for i in smiles_list:
    i_new=i.replace('X','')
    new_smiles_list.append(i_new)
#D拟合的好？



for D,P,H,smi in tqdm(zip(labels_D, labels_P,labels_H,new_smiles_list)):
    if not is_valid_smiles(smi): continue

    smiles_list_v.append(smi)
    labels_D_v.append(D)
    labels_P_v.append(P)
    labels_H_v.append(H)
alllabels_nag_smi=[]

all_frg_smi=[]
all_vail_smi=[]
print('smiles_list_v',len(smiles_list_v))
for i_smi in smiles_list_v:
    try:
        smi = i_smi
        mol_t = Chem.MolFromSmiles(smi)
        #mol_t




        for i in mol_t.GetAtoms():
            i.SetIntProp("atom_idx", i.GetIdx())
        for i in mol_t.GetBonds():
            i.SetIntProp("bond_idx", i.GetIdx())

        mol_s = MurckoScaffold.GetScaffoldForMol(mol_t)
        #mol_s


        d2d = rdMolDraw2D.MolDraw2DSVG(300,300)
        d2d.drawOptions().addBondIndices=True
        d2d.DrawMolecule(mol_t)
        d2d.FinishDrawing()
        #SVG(d2d.GetDrawingText())

        ringinfo = mol_t.GetRingInfo()
        bondrings = ringinfo.BondRings()
        #bondrings

        bondring_list = list(bondrings[0]+bondrings[1])
        #bondring_list

        all_bonds_idx = [bond.GetIdx() for bond in mol_t.GetBonds()]
        #print(all_bonds_idx)

        none_ring_bonds_list = []
        for i in all_bonds_idx:
            if i not in bondring_list:
                none_ring_bonds_list.append(i)
        none_ring_bonds_list

        mol_t.GetBondWithIdx(4).GetBondTypeAsDouble() 

        mol_t.GetBondWithIdx(3).GetBondTypeAsDouble() 

        d2d = rdMolDraw2D.MolDraw2DSVG(300,300)
        d2d.drawOptions().addAtomIndices=True
        d2d.DrawMolecule(mol_t)
        d2d.FinishDrawing()
        SVG(d2d.GetDrawingText())

        mol_t.GetAtomWithIdx(4).IsInRing()

        mol_t.GetBondWithIdx(1).IsInRing()

        cut_bonds = []
        for bond_idx in none_ring_bonds_list:
            bgn_atom_idx = mol_t.GetBondWithIdx(bond_idx).GetBeginAtomIdx()
            ebd_atom_idx = mol_t.GetBondWithIdx(bond_idx).GetEndAtomIdx()
            if mol_t.GetBondWithIdx(bond_idx).GetBondTypeAsDouble() == 1.0:
                if mol_t.GetAtomWithIdx(bgn_atom_idx).IsInRing()+mol_t.GetAtomWithIdx(ebd_atom_idx).IsInRing() == 1:
                    t_bond = mol_t.GetBondWithIdx(bond_idx)
                    t_bond_idx = t_bond.GetIntProp("bond_idx")
                    cut_bonds.append(t_bond_idx)


        res = Chem.FragmentOnBonds(mol_t, cut_bonds)

        #res

        frgs = Chem.GetMolFrags(res, asMols=True)
        #Draw.MolsToGridImage(frgs)
        frg_smi=[]
        for i in frgs:
            
            i=i.replace('*','')
            i=i.replace('[','')
            i=i.replace(']','')
            i=i.replace([0-9],'')
            if is_valid_smiles(i):
                frg_smi.append(Chem.MolToSmiles(i))
        all_frg_smi.append(frg_smi)
        all_vail_smi.append(i_smi)
    except:
        #print(i_smi)
        m = Chem.MolFromSmiles(i_smi)
        #m = Chem.MolFromSmiles('c1ccccc1OCCOC(=O)CC')
        hierarch = Recap.RecapDecompose(m)
        #print(list(hierarch.GetAllChildren().keys()))
        nww_all=[]
        for nww in list(hierarch.GetAllChildren().keys()):
            nww=nww.replace('*','')
            if is_valid_smiles(nww): 
                nww_all.append(nww)
        nww_all.append('C')
        all_frg_smi.append(nww_all)
        all_vail_smi.append(i_smi)


        

print(len(all_vail_smi))
print(len(all_frg_smi))
dic=dict(zip(all_vail_smi,all_frg_smi))

print(len(dic))
empty=0
for i in smiles_list_v:
    if dic[i]==[]:
        empty=empty+1
print('empty',empty)

#print(dic)
import pickle
f_save=open('neg_sample.pkl','wb')
pickle.dump(dic, f_save)
f_save.close()

f_read=open('neg_sample.pkl','rb')
dict_n=pickle.load(f_read)
print(dict_n)

