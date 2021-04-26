import os

import torch
from torch_geometric.data import InMemoryDataset, extract_zip, download_url

from preprocessing import reproduce_preprocessing


class ProteinPointCloudDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ProteinPointCloudDataset, self).__init__(root, transform, pre_transform)

        # -- set up data directories infrastructure

        if not os.path.exists(root):
            os.mkdir(root)
        
        input_dir = os.path.join(root, 'input')
        if not os.path.exists(input_dir):
            os.mkdir(input_dir)

        output_dir = os.path.join(root, 'output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        pdb_files_dir = os.path.join(input_dir, 'pdb_files')
        if not os.path.exists(pdb_files_dir):
            os.mkdir(pdb_files_dir)

        raw_pdb_arrays_dir = os.path.join(input_dir, 'raw_pdb_arrays')
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
        return os.path.join(self.root, 'input')

    @property
    def processed_dir(self):
        return os.path.join(self.raw_dir, 'raw_pdb_arrays')

    @property
    def raw_file_names(self):
        return ['train.txt', 'test.txt']

    def download(self):
        download_url('https://github.com/adamcatto/prosudi/blob/master/train.txt', self.raw_dir)
        download_url('https://github.com/adamcatto/prosudi/blob/master/test.txt', self.raw_dir)

    # -- we won't need `processed_file_names`, since it seems easier to generate `processed_paths` straightaway
    @property
    def processed_file_names(self):
        pass

    @property
    def processed_paths(self):
        pdb_dir = os.path.join(self.root, 'pdb_files')
        point_clouds_dir = os.path.join(self.root, 'raw_pdb_arrays')

        point_clouds_dir_train = os.path.join(self.processed_dir, 'train')
        point_clouds_dir_test = os.path.join(self.processed_dir, 'test')

        point_clouds_train_file_names = os.listdir(point_clouds_dir_train)
        point_clouds_test_file_names = os.listdir(point_clouds_dir_test)

        point_cloud_train_paths = [os.path.join(point_clouds_dir_train, f) for f in point_clouds_train_file_names]
        point_cloud_test_paths = [os.path.join(point_clouds_dir_test, f) for f in point_clouds_test_file_names]

        return point_cloud_train_paths + point_cloud_test_paths

    def process(self):
        reproduce_preprocessing()


def reproduce_dataset():
    protein_point_clouds = ProteinPointCloudDataset(root='../data')
    return protein_point_clouds


reproduce_dataset()