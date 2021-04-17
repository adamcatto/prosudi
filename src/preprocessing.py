from __future__ import annotations
from typing import Mapping, Union
import os
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBList, PDBParser

"""
atom_label_to_num = {}

atom_label_to_num['C'] = 0
atom_label_to_num['H'] = 1
atom_label_to_num['O'] = 2
atom_label_to_num['N'] = 3
atom_label_to_num['S'] = 4
atom_label_to_num['SE'] = 5
atom_label_to_num['ZN'] = 6

num_atoms = 7
"""


def download_pdb_files(file_list_path: str, out_dir: str, server: str='http://ftp.wwpdb.org') -> None:
    pdbl = PDBList(server=server, verbose=False)
    with open(file_list_path, 'r') as molecule_id_list:
        molecule_id_list = molecule_id_list.readlines()
        for molecule_id in tqdm(molecule_id_list):
            pdb_id, chains = tuple(molecule_id.strip('\n').split('_'))
            #chains = chains.split('')
            pdbl.retrieve_pdb_file(pdb_id, pdir=out_dir, file_format='pdb')


def transform_pdb_to_numpy(pdb_file: str, out_dir: str, center: bool=False) -> Mapping[str, np.array]:
    """
    adapted from dMaSIF – https://github.com/FreyrS/dMaSIF/blob/master/data_preprocessing/convert_pdb2npy.py
    read in a pdb
    """
    atom_label_to_num = {}
    num_atoms = 0
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file)
    atoms = structure.get_atoms()

    coords = []
    types = []

    for atom in atoms:
        coords.append(atom.get_coord())
        if atom.element not in atom_label_to_num.keys():
            atom_label_to_num[atom.element] = num_atoms
            num_atoms += 1
        types.append(atom_label_to_num[atom.element])
        #types.append(atom.element)

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(atom_label_to_num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)

    return {"xyz": coords, "types": types_array}


def batch_transform_pdb_to_numpy(pdb_file_list: Union[str, list], out_dir, center=False, verbose=True):
    if isinstance(pdb_file_list, str):
        files = os.listdir(pdb_file_list)
        filenames = [os.path.join(pdb_file_list, f) for f in files]
        pdb_file_list = filenames

    if verbose:
        for i, pdb_file in tqdm(enumerate(pdb_file_list)):
            coords_and_types = transform_pdb_to_numpy(pdb_file, out_dir, center=center)
            np.save(os.path.join(out_dir, files[i].rsplit('.', 1)[0] + '_atomxyz.npy'), coords_and_types['xyz'])
            np.save(os.path.join(out_dir, files[i].rsplit('.', 1)[0] + '_atomtypes.npy'), coords_and_types['types'])
    else:
        for i, pdb_file in enumerate(pdb_file_list):
            coords_and_types = transform_pdb_to_numpy(pdb_file, out_dir, center=center)
            np.save(os.path.join(out_dir, files[i].rsplit('.', 1)[0] + '_atomxyz.npy'), coords_and_types['xyz'])
            np.save(os.path.join(out_dir, files[i].rsplit('.', 1)[0] + '_atomtypes.npy'), coords_and_types['types'])


def reproduce_preprocessing():
    data_dir = '../data/'
    input_dir = os.path.join(data_dir, 'input')
    output_dir = os.path.join(data_dir, 'output')

    pdb_files_dir = os.path.join(input_dir, 'pdb_files')
    raw_pdb_arrays_dir = os.path.join(input_dir, 'raw_pdb_arrays')

    train_files_dir = os.path.join(pdb_files_dir, 'train')
    test_files_dir = os.path.join(pdb_files_dir, 'test')

    train_raw_pdb_arrays_dir = os.path.join(raw_pdb_arrays_dir, 'train')
    test_raw_pdb_arrays_dir = os.path.join(raw_pdb_arrays_dir, 'test')
    
    dirs = [data_dir, input_dir, output_dir, pdb_files_dir, raw_pdb_arrays_dir, train_files_dir,
                test_files_dir, train_raw_pdb_arrays_dir, test_raw_pdb_arrays_dir]

    for d in dirs:
        if not os.path.exists:
            os.mkdir(d)

    train_list_file = os.path.join(input_dir, 'train.txt')
    test_list_file = os.path.join(input_dir, 'test.txt')

    # -- download train PDB files
    download_pdb_files(file_list_path=train_list_file, out_dir=train_files_dir)

    # -- download test PDB files
    download_pdb_files(file_list_path=test_list_file, out_dir=test_files_dir)

    # -- transform train PDB files to numpy arrays + save
    batch_transform_pdb_to_numpy(pdb_file_list=train_files_dir, out_dir=train_raw_pdb_arrays_dir)

    # -- transform test PDB files to numpy arrays + save
    batch_transform_pdb_to_numpy(pdb_file_list=test_files_dir, out_dir=test_raw_pdb_arrays_dir)


if __name__ == '__main__':
    reproduce_preprocessing()