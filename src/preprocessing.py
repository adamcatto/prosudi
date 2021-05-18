from __future__ import annotations
from typing import Mapping, Union
import os
import subprocess

import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBList, PDBParser
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb


atom_label_to_num = {}

atom_label_to_num['C'] = 0
atom_label_to_num['H'] = 1
atom_label_to_num['O'] = 2
atom_label_to_num['N'] = 3
atom_label_to_num['S'] = 4
atom_label_to_num['P'] = 5
atom_label_to_num['I'] = 6
atom_label_to_num['F'] = 7
atom_label_to_num['Br'] = 8
atom_label_to_num['Cl'] = 9
atom_label_to_num['Mg'] = 10
atom_label_to_num['Fe'] = 11
atom_label_to_num['Ca'] = 12
atom_label_to_num['Zn'] = 13
atom_label_to_num['Du'] = 14
atom_label_to_num['Mn'] = 15
atom_label_to_num['Na'] = 16
atom_label_to_num['K'] = 17
atom_label_to_num['Cu'] = 18
atom_label_to_num['Se'] = 19

# num_atoms = 7
"""
def download_pdb_files(file_list_path: str, out_dir: str, server: str='http://ftp.wwpdb.org') -> None:
    pdbl = PDBList(server=server, verbose=False)
    with open(file_list_path, 'r') as molecule_id_list:
        molecule_id_list = molecule_id_list.readlines()
        for molecule_id in tqdm(molecule_id_list):
            pdb_id, chains = tuple(molecule_id.strip('\n').split('_'))
            #chains = chains.split('')
            pdbl.retrieve_pdb_file(pdb_id, pdir=out_dir, file_format='pdb')
"""


def one_hot_encode(arr):
    """
    adapted from https://stackoverflow.com/a/58676802/5338871
    """
    unique, inverse = np.unique(arr, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot


def get_element_symbols(atom_str):
    atom = atom_str.split('.')[0]
    return atom


def transform_pdb_to_numpy(pdb_file: str, experiment_type: str, center: bool=False) -> Mapping[str, np.array]:
    """
    adapted in part from dMaSIF – https://github.com/FreyrS/dMaSIF/blob/master/data_preprocessing/convert_pdb2npy.py
    read in a pdb
    `experiment_type` in ['pdbbind', 'scpdb']
    """
    # print(pdb_file)
    assert experiment_type in ['pdbbind', 'scpdb']

    # atom_label_to_num = {}
    num_atoms = 0
    if pdb_file[-4:] == 'mol2':
        try:
            df = PandasMol2().read_mol2(pdb_file).df
            coords = df[['x', 'y', 'z']].values
            # -- to get atom type, get first letter of string by converting to 1-byte array
            # thanks to https://stackoverflow.com/a/48320451/5338871 for this idea.
            atoms = df['atom_type'].values
            if atoms[0] == '':
                with open('../data/logs/' + experiment_type + '/problem_files.txt', 'a') as outfile:
                    outfile.write(pdb_file + '\n')
                return np.zeros((1,13))
            atoms = np.vectorize(get_element_symbols)(atoms)
        except:
            with open('../data/logs/' + experiment_type + '/problem_files.txt', 'a') as outfile:
                outfile.write(pdb_file + '\n')
            return np.zeros((1,13))
        
    else:
        try:
            df = PandasPdb().read_pdb(pdb_file).df['ATOM']
            coords = df[['x_coord', 'y_coord', 'z_coord']].values
            atoms = df['element_symbol'].values
            if atoms[0] == '':
                with open('../data/logs/problems_files.txt', 'a') as outfile:
                    outfile.write(pdb_file + '\n')
                return np.zeros((1,13))

            atoms = np.vectorize(get_element_symbols)(atoms)
        except:
            with open('../data/logs/problems_files.txt', 'a') as outfile:
                outfile.write(pdb_file + '\n')
            return np.zeros((1,13))

    types = np.vectorize(atom_label_to_num.__getitem__)(atoms)
    types_array = np.zeros((len(types), len(atom_label_to_num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)
    
    combined_array = np.concatenate((coords, types_array), axis=1)
    return combined_array


def batch_transform_pdb_to_numpy(pdb_file_list: Union[str, list], out_dir, center=False, verbose=True):
    if isinstance(pdb_file_list, str):
        files = os.listdir(pdb_file_list)
        filenames = [os.path.join(pdb_file_list, f) for f in files]
        pdb_file_list = filenames

    combined_dir = '../data/input/processed_files/combined_coords_atoms_arrays/'
    parser = PDBParser()
    if verbose:
        for i, pdb_file in tqdm(enumerate(pdb_file_list)):
            coords_and_types = transform_pdb_to_numpy(pdb_file, parser, center=center)
            # np.save(os.path.join(out_dir, files[i].rsplit('.', 1)[0] + '_atomxyz.npy'), coords_and_types['xyz'])
            # np.save(os.path.join(out_dir, files[i].rsplit('.', 1)[0] + '_atomtypes.npy'), coords_and_types['types'])
            np.save(os.path.join(out_dir, files[i].rsplit('.', 1)[0] + '_combined.npy'), coords_and_types['combined'])
    else:
        for i, pdb_file in enumerate(pdb_file_list):
            coords_and_types = transform_pdb_to_numpy(pdb_file, parser, center=center)
            # np.save(os.path.join(out_dir, files[i].rsplit('.', 1)[0] + '_atomxyz.npy'), coords_and_types['xyz'])
            # np.save(os.path.join(out_dir, files[i].rsplit('.', 1)[0] + '_atomtypes.npy'), coords_and_types['types'])
            np.save(os.path.join(out_dir, files[i].rsplit('.', 1)[0] + '_combined.npy'), coords_and_types['combined'])


def clean_files_pdbbind(pdb_files_dir='../data/input/pdbbind/refined-set/'):
    """
    remove ligand SDF files ==> we are left with only PDB files for protein, ligand, and pocket.
    """
    pdb_ids = [os.path.join(pdb_files_dir, x) for x in os.listdir(pdb_files_dir) if x[0] != '.']
    for pdb_id in tqdm(pdb_ids):
        files = [os.path.join(pdb_id, f) for f in os.listdir(pdb_id) if f[0] != '.']
        for f in files:
            if f[-4:] == '.sdf':
                subprocess.call(['rm', f])


def clean_files_scpdb(pdb_files_dir='../data/input/scpdb/scpdb_files/'):
    """
    remove all files that are not the mol2 file for protein, ligand, or site.
    """
    pdb_ids = [os.path.join(pdb_files_dir, x) for x in os.listdir(pdb_files_dir) if x[0] != '.']
    for pdb_id in tqdm(pdb_ids):
        files = [f for f in os.listdir(pdb_id) if f[0] != '.']
        filenames = [os.path.join(pdb_id, f) for f in os.listdir(pdb_id) if f[0] != '.']
        for i, f in enumerate(files):
            # if f[-4:] == '.sdf':
            if f not in ('protein.mol2', 'ligand.mol2', 'site.mol2'):
                subprocess.call(['rm', filenames[i]])


def train_test_split(pdb_id_dir, train_percentage=0.8):
    pdb_ids = os.listdir(pdb_id_dir)
    num_ids = len(pdb_ids)
    num_train = int(num_ids * train_percentage)
    np.random.shuffle(pdb_ids)
    train_ids = pdb_ids[0: num_train]
    test_ids = pdb_ids[num_train:]
    with open('../data/input/scpdb/train.txt', 'w') as train_ids_file:
        for train_id in train_ids:
            train_ids_file.write(train_id + '\n')

    with open('../data/input/scpdb/test.txt', 'w') as test_ids_file:
        for test_id in test_ids:
            test_ids_file.write(test_id + '\n')

    return train_ids, test_ids


def reproduce_preprocessing_pdbbind():
    data_dir = '../data/'

    
    input_dir = os.path.join(data_dir, 'input')
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    output_dir = os.path.join(data_dir, 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pdbbind_dir = os.path.join(input_dir, 'pdbbind')
    if not os.path.exists(pdbbind_dir):
        os.mkdir(pdbbind_dir)
    
    clean_files_pdbbind()

    train_ids = []
    test_ids = []


    train_ids_file = os.path.join(pdbbind_dir, 'train.txt')
    test_ids_file = os.path.join(pdbbind_dir, 'test.txt')
    
    

    with open(train_ids_file, 'r') as train_id_file:
        for train_id in train_id_file.readlines():
            train_ids.append(train_id.strip('\n'))

    # now we will have train.txt and test.txt if we didn't have them before, so no need to do further train test split
    with open(test_ids_file, 'r') as test_id_file:
        for test_id in test_id_file.readlines():
            test_ids.append(test_id.strip('\n'))

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
    
    dirs = [data_dir, input_dir, output_dir, pdb_files_dir, raw_pdb_arrays_dir, train_files_dir,
                test_files_dir, train_raw_pdb_arrays_dir, test_raw_pdb_arrays_dir]
    """
    for d in dirs:
        if not os.path.exists:
            os.mkdir(d)
    """

    # train_list_file = os.path.join(input_dir, 'train.txt')
    # test_list_file = os.path.join(input_dir, 'test.txt')
    # parser = PDBParser()
    # print(train_ids)

    # -- download train PDB files
    # download_pdb_files(file_list_path=train_list_file, out_dir=train_files_dir)

    # -- download test PDB files
    # download_pdb_files(file_list_path=test_list_file, out_dir=test_files_dir)

    for i, train_id_dir in tqdm(enumerate(train_ids)):
        # print(train_id_dir)
        full_train_id_dir = os.path.join(pdb_files_dir, train_id_dir)
        raw_pdb_arrays_train_id_dir = os.path.join(raw_pdb_arrays_dir, 'train', train_id_dir)
        if not os.path.exists(raw_pdb_arrays_train_id_dir):
            os.mkdir(raw_pdb_arrays_train_id_dir)
        else:
            continue
        # after each iteration we have a directory with protein, ligand, and pocket PDB files
        for x_file in os.listdir(full_train_id_dir):
            # print(x_file)
            if x_file[0] in ('.'):
                continue
            full_x_file_path = os.path.join(full_train_id_dir, x_file)
            combined_array = transform_pdb_to_numpy(full_x_file_path, 'pdbbind')
            np.save(os.path.join(raw_pdb_arrays_train_id_dir, x_file.rsplit('.', 1)[0] + '_combined.npy'), combined_array)
        
    for i, test_id_dir in tqdm(enumerate(test_ids)):
        # print(test_id_dir)
        full_test_id_dir = os.path.join(pdb_files_dir, test_id_dir)
        raw_pdb_arrays_test_id_dir = os.path.join(raw_pdb_arrays_dir, 'test', test_id_dir)
        if not os.path.exists(raw_pdb_arrays_test_id_dir):
            os.mkdir(raw_pdb_arrays_test_id_dir)
        # after each iteration we have a directory with protein, ligand, and pocket PDB files
        for x_file in os.listdir(full_test_id_dir):
            # print(x_file)
            if x_file[0] in ('.'):
                continue
            full_x_file_path = os.path.join(full_test_id_dir, x_file)
            combined_array = transform_pdb_to_numpy(full_x_file_path, 'pdbbind')
            np.save(os.path.join(raw_pdb_arrays_test_id_dir, x_file.rsplit('.', 1)[0] + '_combined.npy'), combined_array)
    # -- transform train PDB files to numpy arrays + save
    # batch_transform_pdb_to_numpy(pdb_file_list=train_files_dir, out_dir=train_raw_pdb_arrays_dir)

    # -- transform test PDB files to numpy arrays + save
    # batch_transform_pdb_to_numpy(pdb_file_list=test_files_dir, out_dir=test_raw_pdb_arrays_dir)
    with open('../data/logs/pdbbind/problem_files.txt', 'r') as to_remove:
        for f in to_remove.readlines():
            protein = f.split('/')[-1].split('_')[0]
            full_path_train = os.path.join('../data/input/pdbbind/raw_pdb_arrays/train', protein)
            full_path_test = os.path.join('../data/input/pdbbind/raw_pdb_arrays/test', protein)
            subprocess.call(['rm', '-rf',  full_path_train])
            subprocess.call(['rm', '-rf',  full_path_test])


def reproduce_preprocessing_scpdb():
    data_dir = '../data/'

    
    input_dir = os.path.join(data_dir, 'input')
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    output_dir = os.path.join(data_dir, 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    scpdb_dir = os.path.join(input_dir, 'scpdb')
    if not os.path.exists(scpdb_dir):
        os.mkdir(scpdb_dir)

    # train_test_split('../data/input/scpdb/scpdb_files/')
    
    # clean_files_scpdb()

    train_ids = []
    test_ids = []


    train_ids_file = os.path.join(scpdb_dir, 'train.txt')
    test_ids_file = os.path.join(scpdb_dir, 'test.txt')
    
    

    with open(train_ids_file, 'r') as train_id_file:
        for train_id in train_id_file.readlines():
            train_ids.append(train_id.strip('\n'))

    # now we will have train.txt and test.txt if we didn't have them before, so no need to do further train test split
    with open(test_ids_file, 'r') as test_id_file:
        for test_id in test_id_file.readlines():
            test_ids.append(test_id.strip('\n'))

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
    
    dirs = [data_dir, input_dir, output_dir, pdb_files_dir, raw_pdb_arrays_dir, train_files_dir,
                test_files_dir, train_raw_pdb_arrays_dir, test_raw_pdb_arrays_dir]
    """
    for d in dirs:
        if not os.path.exists:
            os.mkdir(d)
    """

    # train_list_file = os.path.join(input_dir, 'train.txt')
    # test_list_file = os.path.join(input_dir, 'test.txt')
    # parser = PDBParser()
    # print(train_ids)

    # -- download train PDB files
    # download_pdb_files(file_list_path=train_list_file, out_dir=train_files_dir)

    # -- download test PDB files
    # download_pdb_files(file_list_path=test_list_file, out_dir=test_files_dir)

    for i, train_id_dir in tqdm(enumerate(train_ids)):
        # print(train_id_dir)
        full_train_id_dir = os.path.join(pdb_files_dir, train_id_dir)
        raw_pdb_arrays_train_id_dir = os.path.join(raw_pdb_arrays_dir, 'train', train_id_dir)
        if not os.path.exists(raw_pdb_arrays_train_id_dir):
            os.mkdir(raw_pdb_arrays_train_id_dir)
        else:
            continue
        # after each iteration we have a directory with protein, ligand, and pocket PDB files
        for x_file in [f for f in os.listdir(full_train_id_dir) if f[0] != '.']:
            # print(x_file)
            # if x_file[0] in ('.'):
                # continue
            full_x_file_path = os.path.join(full_train_id_dir, x_file)
            combined_array = transform_pdb_to_numpy(full_x_file_path, 'scpdb')
            np.save(os.path.join(raw_pdb_arrays_train_id_dir, x_file.rsplit('.', 1)[0] + '_combined.npy'), combined_array)
        
    for i, test_id_dir in tqdm(enumerate(test_ids)):
        # print(test_id_dir)
        full_test_id_dir = os.path.join(pdb_files_dir, test_id_dir)
        raw_pdb_arrays_test_id_dir = os.path.join(raw_pdb_arrays_dir, 'test', test_id_dir)
        if not os.path.exists(raw_pdb_arrays_test_id_dir):
            os.mkdir(raw_pdb_arrays_test_id_dir)
        # after each iteration we have a directory with protein, ligand, and pocket PDB files
        for x_file in [f for f in os.listdir(full_test_id_dir) if f[0 != '.']]:
            # print(x_file)
            # if x_file[0] in ('.'):
                # continue
            full_x_file_path = os.path.join(full_test_id_dir, x_file)
            combined_array = transform_pdb_to_numpy(full_x_file_path, 'scpdb')
            np.save(os.path.join(raw_pdb_arrays_test_id_dir, x_file.rsplit('.', 1)[0] + '_combined.npy'), combined_array)
    # -- transform train PDB files to numpy arrays + save
    # batch_transform_pdb_to_numpy(pdb_file_list=train_files_dir, out_dir=train_raw_pdb_arrays_dir)

    # -- transform test PDB files to numpy arrays + save
    # batch_transform_pdb_to_numpy(pdb_file_list=test_files_dir, out_dir=test_raw_pdb_arrays_dir)
    with open('../data/logs/pdbbind/problem_files.txt', 'r') as to_remove:
        for f in to_remove.readlines():
            protein = f.split('/')[-1].split('_')[0]
            full_path_train = os.path.join('../data/input/scpdb/raw_pdb_arrays/train', protein)
            full_path_test = os.path.join('../data/input/scpdb/raw_pdb_arrays/test', protein)
            subprocess.call(['rm', '-rf',  full_path_train])
            subprocess.call(['rm', '-rf',  full_path_test])


def construct_ground_truth_segmentation(protein_array, binding_site_array):
    protein_coords = protein_array[:, 0:3]
    binding_site_coords = binding_site_array[:, 0:3]
    ground_truth = np.isin(protein_coords, binding_site_coords).all(axis=1).astype(int)
    # make into 2d array of 1 row by n cols
    return ground_truth[:, np.newaxis]


if __name__ == '__main__':
    # reproduce_preprocessing_pdbbind()
    reproduce_preprocessing_scpdb()