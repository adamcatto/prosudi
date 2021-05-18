import os

import numpy as np
import torch
from torch_geometric.data import Dataset, extract_zip, download_url

from preprocessing import reproduce_preprocessing_pdbbind, reproduce_preprocessing_scpdb


def list_all_files_in_dir(input_dir, file_types=[]):
	if file_types:
		result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_dir) for f in filenames if os.path.splitext(f)[1] in file_types]
	else:
		result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_dir) for f in filenames]

	return result


class PDBbindPointCloudDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PDBbindPointCloudDataset, self).__init__(root, transform, pre_transform)

        # -- set up data directories infrastructure

        if not os.path.exists(root):
            os.mkdir(root)
        
        input_dir = os.path.join(root, 'input')
        if not os.path.exists(input_dir):
            os.mkdir(input_dir)

        output_dir = os.path.join(root, 'output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        pdbbind_dir = os.path.join(input_dir, 'pdbbind')
        if not os.path.exists(pdbbind_dir):
            os.mkdir(pdbbind_dir)

        pdb_files_dir = os.path.join(pdbbind_dir, 'refined-set')
        if not os.path.exists(pdb_files_dir):
            os.mkdir(pdb_files_dir)

        raw_pdb_arrays_dir = os.path.join(pdbbind_dir, 'raw_pdb_arrays')
        if not os.path.exists(raw_pdb_arrays_dir):
            os.mkdir(raw_pdb_arrays_dir)

        train_files_dir = os.path.join(pdb_files_dir, 'train')
        if not os.path.exists(train_files_dir):
            os.mkdir(train_files_dir)

        test_files_dir = os.path.join(pdb_files_dir, 'test')
        if not os.path.exists(test_files_dir):
            os.mkdir(test_files_dir)

        train_raw_pdb_arrays_dir = os.path.join(raw_pdb_arrays_dir, 'train')
        if not os.path.exists(train_raw_pdb_arrays_dir):
            os.mkdir(train_raw_pdb_arrays_dir)

        test_raw_pdb_arrays_dir = os.path.join(raw_pdb_arrays_dir, 'test')
        if not os.path.exists(test_raw_pdb_arrays_dir):
            os.mkdir(test_raw_pdb_arrays_dir)

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'input', 'pdbbind')

    @property
    def processed_dir(self):
        return os.path.join(self.raw_dir, 'raw_pdb_arrays')

    @property
    def raw_file_names(self):
        return ['train.txt', 'test.txt']

    def download(self):
        # todo: upload train/test files for pdbbind/scpdb to github and link their URLs to `download_url()`
        pass
        # download_url('https://github.com/adamcatto/prosudi/blob/master/train.txt', self.raw_dir)
        # download_url('https://github.com/adamcatto/prosudi/blob/master/test.txt', self.raw_dir)

    # -- we won't need `processed_file_names`, since it seems easier to generate `processed_paths` straightaway
    @property
    def processed_file_names(self):
        pass

    @property
    def processed_paths(self):
        pdb_dir = os.path.join(self.root, 'input', 'pdbbind', 'refined-set')
        point_clouds_dir = os.path.join(self.root, 'input', 'pdbbind', 'raw_pdb_arrays')

        point_cloud_paths = list_all_files_in_dir(point_clouds_dir)

        # point_clouds_dir_train = os.path.join(self.processed_dir, 'train')
        # point_clouds_dir_test = os.path.join(self.processed_dir, 'test')

        # point_clouds_train_dir_names = os.listdir(point_clouds_dir_train)
        # point_clouds_test_dir_names = os.listdir(point_clouds_dir_test)

        # point_cloud_train_paths = [os.path.join(point_clouds_dir_train, f) for f in point_clouds_train_file_names]
        # point_cloud_test_paths = [os.path.join(point_clouds_dir_test, f) for f in point_clouds_test_file_names]

        return point_cloud_paths

    def process(self):
        reproduce_preprocessing_pdbbind()

    def len(self):
        return len(self.data)


class SCPDBPointCloudDataset(Dataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None):
        super(SCPDBPointCloudDataset, self).__init__(root, transform, pre_transform)
        
        self.train = train
        print(train)

        # -- set up data directories infrastructure

        if not os.path.exists(root):
            os.mkdir(root)
        
        input_dir = os.path.join(root, 'input')
        if not os.path.exists(input_dir):
            os.mkdir(input_dir)

        output_dir = os.path.join(root, 'output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        scpdb_dir = os.path.join(input_dir, 'scpdb')
        if not os.path.exists(scpdb_dir):
            os.mkdir(scpdb_dir)

        pdb_files_dir = os.path.join(scpdb_dir, 'scpdb_files')
        if not os.path.exists(pdb_files_dir):
            os.mkdir(pdb_files_dir)

        raw_pdb_arrays_dir = os.path.join(scpdb_dir, 'raw_pdb_arrays')
        if not os.path.exists(raw_pdb_arrays_dir):
            os.mkdir(raw_pdb_arrays_dir)

        train_files_dir = os.path.join(pdb_files_dir, 'train')
        if not os.path.exists(train_files_dir):
            os.mkdir(train_files_dir)

        test_files_dir = os.path.join(pdb_files_dir, 'test')
        if not os.path.exists(test_files_dir):
            os.mkdir(test_files_dir)

        train_raw_pdb_arrays_dir = os.path.join(raw_pdb_arrays_dir, 'train')
        if not os.path.exists(train_raw_pdb_arrays_dir):
            os.mkdir(train_raw_pdb_arrays_dir)

        test_raw_pdb_arrays_dir = os.path.join(raw_pdb_arrays_dir, 'test')
        if not os.path.exists(test_raw_pdb_arrays_dir):
            os.mkdir(test_raw_pdb_arrays_dir)

        # self.path = self.processed_paths[0] if train else self.processed_paths[1]
        # self.data, self.slices = torch.load(path)


    @property
    def raw_dir(self):
        return os.path.join(self.root, 'input', 'scpdb')

    @property
    def processed_dir(self):
        sub_dir = 'train' if self.train else 'test'
        return os.path.join(self.raw_dir, 'raw_pdb_arrays', sub_dir)

    @property
    def raw_file_names(self):
        return ['train.txt', 'test.txt']

    def download(self):
        # todo: upload train/test files for pdbbind/scpdb to github and link their URLs to `download_url()`
        pass
        # download_url('https://github.com/adamcatto/prosudi/blob/master/train.txt', self.raw_dir)
        # download_url('https://github.com/adamcatto/prosudi/blob/master/test.txt', self.raw_dir)

    # -- we won't need `processed_file_names`, since it seems easier to generate `processed_paths` straightaway
    @property
    def processed_file_names(self):
        pass

    @property
    def processed_paths(self):
        pdb_ids = [os.path.join(self.processed_dir, x) for x in os.listdir(self.processed_dir) if x[0] != '.']
        return sorted(pdb_ids)

    def process(self):
        reproduce_preprocessing_scpdb()

    def len(self):
        return len(self.processed_paths)

    def get(self, idx):
        pdb_id = self.processed_paths[idx]
        ligand, protein, site = tuple(sorted([os.path.join(self.path, pdb_id, x) for x in os.listdir(pdb_id) if x[0] != '.']))
        ligand = torch.from_numpy(np.load(ligand)).float()
        protein = torch.from_numpy(np.load(protein)).float()
        site = torch.from_numpy(np.load(site)).float()
        return ligand, protein, site



def reproduce_scpdb_dataset():
    protein_point_clouds = SCPDBPointCloudDataset(root='../data')
    return protein_point_clouds


# reproduce_pdbbind_dataset()
# reproduce_scpdb_dataset()