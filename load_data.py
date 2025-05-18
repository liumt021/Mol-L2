import os, re, glob, copy
from KG_process import *
from module import Molecular
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch_geometric.data import Dataset
import os.path as osp
import pandas as pd

import torch.utils.data
from torch_geometric.data import Data, Batch
import torch

allowable_features = {

    'atomic_num' : list(range(1, 122)),
    'formal_charge' : ['unk',-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'chirality' : ['unk',
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'hybridization' : ['unk',
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'numH' : ['unk',0, 1, 2, 3, 4, 5, 6, 7, 8],
    'implicit_valence' : ['unk',0, 1, 2, 3, 4, 5, 6],
    'degree' : ['unk',0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'isaromatic':[False,True],

    'bond_type' : ['unk',
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'bond_dirs' : [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'bond_isconjugated':[False,True],
    'bond_inring':[False,True],
    'bond_stereo': ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE","STEREOCIS", "STEREOTRANS"]
}

atom_dic = [len(allowable_features['atomic_num']),len(allowable_features['formal_charge']),len(allowable_features['chirality' ]),
            len(allowable_features['hybridization']),len(allowable_features['numH' ]),len(allowable_features['implicit_valence']),
            len(allowable_features['degree']),len(allowable_features['isaromatic'])]


bond_dic = [len(allowable_features['bond_type']),len(allowable_features['bond_dirs' ]),len(allowable_features['bond_isconjugated']),
            len(allowable_features['bond_inring']),len(allowable_features['bond_stereo'])]

atom_cumsum = np.cumsum(atom_dic)
bond_cumsum = np.cumsum(bond_dic)


def mol_to_graph_data_obj_complex(mol, s):
    assert mol!=None
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = \
        [allowable_features['atomic_num'].index(atom.GetAtomicNum())] +\
        [allowable_features['formal_charge'].index(atom.GetFormalCharge())+atom_cumsum[0]]+\
        [allowable_features['chirality'].index(atom.GetChiralTag()) + atom_cumsum[1]]+ \
        [allowable_features['hybridization'].index(atom.GetHybridization()) + atom_cumsum[2]]+ \
        [allowable_features['numH'].index(atom.GetTotalNumHs()) + atom_cumsum[3]] + \
        [allowable_features['implicit_valence'].index(atom.GetImplicitValence()) + atom_cumsum[4]] + \
        [allowable_features['degree'].index(atom.GetDegree()) + atom_cumsum[5]] + \
        [allowable_features['isaromatic'].index(atom.GetIsAromatic()) + atom_cumsum[6]]

        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)


    num_bond_features = 5
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature =\
            [allowable_features['bond_type'].index(bond.GetBondType())] + \
            [allowable_features['bond_dirs'].index(bond.GetBondDir())+bond_cumsum[0]]+ \
            [allowable_features['bond_isconjugated'].index(bond.GetIsConjugated())+bond_cumsum[1]] + \
            [allowable_features['bond_inring'].index(bond.IsInRing()) + bond_cumsum[2]] + \
            [allowable_features['bond_stereo'].index(str(bond.GetStereo()))+bond_cumsum[3]]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    m = Molecular(s)
    group_attr = m.mol.f_fgs  # group_embeddings

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, group_attr=group_attr)
    return data


class finetuneDataset(InMemoryDataset):
    def __init__(self, root='data/downstream', dataset='finetune', transform=None, pre_transform=None):
        super(finetuneDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    @property
    def processed_file_names(self):
        return [self.dataset+'.pt']
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            smiles_list = f.read().split('\n')
            smiles_list.pop()

        data_list = []
        for s in tqdm(smiles_list):
            rdkit_mol = AllChem.MolFromSmiles(s)
            if rdkit_mol != None:
              data = mol_to_graph_data_obj_complex(rdkit_mol, s)
              if data:
                  data.text = [s]
                  data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_data_loaders(self, batch_size=32, seed=42, split_ratio=(0.8, 0.1, 0.1)):
        data = self
        num_data = len(data)
        np.random.seed(seed)
        indices = np.random.permutation(num_data)

        train_end = int(split_ratio[0] * num_data)
        val_end = train_end + int(split_ratio[1] * num_data)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_dataset = data[train_idx.tolist()]
        val_dataset = data[val_idx.tolist()]
        test_dataset = data[test_idx.tolist()]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

class DDIDataset:
    def __init__(self, dataset_name, split_dir, batch_size=32):
        self.dataset_name = dataset_name.lower()
        self.split_dir = split_dir
        self.batch_size = batch_size

    def _load_one(self, file_path):
        df = pd.read_csv(file_path)
        data_list = []

        for idx, row in df.iterrows():
            if self.dataset_name == 'zhangddi':
                smiles1, smiles2, label = row['smiles_1'], row['smiles_2'], row['label']
            elif self.dataset_name == 'chchminer':
                smiles1, smiles2, label = row['smiles_1'], row['smiles_2'], row['label']
            elif self.dataset_name == 'deepddi':
                smiles1, smiles2, label = row['smiles_1'], row['smiles_2'], row['label']
            else:
                raise ValueError("Unsupported DDI dataset")

            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            if mol1 is None or mol2 is None:
                continue

            data1 = mol_to_graph_data_obj_complex(mol1, smiles1)
            data2 = mol_to_graph_data_obj_complex(mol2, smiles2)

            if data1 is None or data2 is None:
                continue

            data_list.append((data1, data2, torch.tensor(label, dtype=torch.float)))

        return data_list

    def _wrap_loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, follow_batch=[])

    def get_data_loaders(self):
        train_data = self._load_one(os.path.join(self.split_dir, f"{self.dataset_name}_train.csv"))
        valid_data = self._load_one(os.path.join(self.split_dir, f"{self.dataset_name}_valid.csv"))
        test_data = self._load_one(os.path.join(self.split_dir, f"{self.dataset_name}_test.csv"))

        return self._wrap_loader(train_data), self._wrap_loader(valid_data), self._wrap_loader(test_data)
