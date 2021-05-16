"""
This file is adapted from https://github.com/nmwsharp/diffusion-net/blob/master/src/geometry.py
in order to reproduce the DiffusionNet model
"""


import os.path
import sys
import random

import scipy
from scipy.sparse import linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)
import sklearn.neighbors
import numpy as np
from scipy import spatial
import torch
import igl
import robust_laplacian
from torch.distributions.categorical import Categorical

import utils
from utils import toNP


def norm(x, highdim=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return torch.norm(x, dim=len(x.shape) - 1)


def norm2(x, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return dot(x, x)


def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if(len(x.shape) == 1):
        raise ValueError("called normalize() on single vector of dim " +
                         str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called normalize() with large last dimension " +
                         str(x.shape) + " are you sure?")
    return x / (norm(x, highdim=highdim) + divide_eps).unsqueeze(-1)


def face_coords(verts, faces):
    coords = verts[faces]
    return coords


def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)


def dot(vec_A, vec_B):
    return torch.sum(vec_A * vec_B, dim=-1)


# Given (..., 3) vectors and normals, projects out any components of vecs
# which lies in the direction of normals. Normals are assumed to be unit.

def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)


def face_area(verts, faces):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)
    return 0.5 * norm(raw_normal)


def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal


def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:,2,:]
    return normal / np.linalg.norm(normal,axis=-1, keepdims=True)


def vertex_normals(verts, faces, n_neighbors_cloud=30):
    verts_np = toNP(verts)
    if isinstance(faces, list):
        is_cloud = faces == []
    else:
        is_cloud = faces.numel() == 0
    if is_cloud: # point cloud
    
        _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts_np[neigh_inds,:]
        neigh_points = neigh_points - verts_np[:,np.newaxis,:]
        normals = neighborhood_normal(neigh_points)

    else: # mesh

        normals = igl.per_vertex_normals(verts_np, toNP(faces))

        # if any are NaN, wiggle slightly and recompute
        bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_normals_mask.any():
            bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5) * scale
            wiggle_verts = verts_np + bad_normals_mask * wiggle
            normals = igl.per_vertex_normals(wiggle_verts, toNP(faces))

        # if still NaN assign random normals (probably means unreferenced verts in mesh)
        bad_normals_mask = np.isnan(normals).any(axis=1)
        if bad_normals_mask.any():
            normals[bad_normals_mask,:] = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5)[bad_normals_mask,:]
            normals = normals / np.linalg.norm(normals, axis=-1)[:,np.newaxis]
            

    normals = torch.from_numpy(normals).to(device=verts.device, dtype=verts.dtype)
        
    if torch.any(torch.isnan(normals)): raise ValueError("NaN normals :(")

    return normals


def build_tangent_frames(verts, faces, normals=None):

    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    if normals == None:
        vert_normals = vertex_normals(verts, faces)  # (V,3)
    else:
        vert_normals = normals 

    # = find an orthogonal basis

    basis_cand1 = torch.tensor([1, 0, 0]).to(device=device, dtype=dtype).expand(V, -1)
    basis_cand2 = torch.tensor([0, 1, 0]).to(device=device, dtype=dtype).expand(V, -1)
    
    basisX = torch.where((torch.abs(dot(vert_normals, basis_cand1))
                          < 0.9).unsqueeze(-1), basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = torch.stack((basisX, basisY, vert_normals), dim=-2)
    
    if torch.any(torch.isnan(frames)):

        if torch.any(torch.isnan(vert_normals)):
            print("NaN Z")
        if torch.any(torch.isnan(basisX)):
            print("NaN X")
        if torch.any(torch.isnan(basisY)):
            print("NaN Y")

    return frames


def build_grad_point_cloud(verts, frames, n_neighbors_cloud=30):

    verts_np = toNP(verts)
    frames_np = toNP(frames)

    _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
    neigh_points = verts_np[neigh_inds,:]
    neigh_vecs = neigh_points - verts_np[:,np.newaxis,:]

    # TODO this could easily be way faster. For instance we could avoid the weird edges format and the corresponding pure-python loop via some numpy broadcasting of the same logic. The way it works right now is just to share code with the mesh version. But its low priority since its preprocessing code.

    edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors_cloud)
    edges = np.stack((edge_inds_from, neigh_inds.flatten()))
    edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)
   
    return build_grad(verts_np, torch.tensor(edges), edge_tangent_vecs)


def edge_tangent_vectors(verts, frames, edges):
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = torch.stack((compX, compY), dim=-1)

    return edge_tangent


def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient. All values pointwise.
    - edges: (2, E)
    """

    edges_np = toNP(edges)
    edge_tangent_vectors_np = toNP(edge_tangent_vectors)

    # TODO find a way to do this in pure numpy?

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges_np.shape[1]):
        tail_ind = edges_np[0, iE]
        tip_ind = edges_np[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]
        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges_np[1, iE]
            ind_lookup.append(jV)
            w_e = 1.

            lhs_mat[i_neigh][:] = w_e * edge_tangent_vectors[iE][:]
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * (1)

        sol_mat = np.linalg.pinv(lhs_mat) @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(
            N, N)).tocsc()

    return mat


def compute_operators(verts, faces, k_eig, normals=None):
    """
    Builds spectral operators for a mesh/point cloud. Constructs mass matrix, eigenvalues/vectors for Laplacian, along with gradient from spcetral domain.
    Torch in / torch out.
    Arguments:
      - verts: (V,3) vertex positions
      - faces: (F,3) list of triangular faces. If empty, assumed to be a point cloud.
      - k_eig: number of eigenvectors to use
    Returns:
      - frames: (V,3,3) X/Y/Z coordinate frame at each vertex. Z coordinate is normal (e.g. [:,2,:] for normals)
      - massvec: (V) real diagonal of lumped mass matrix
      - evals: (k) list of eigenvalues of the Laplacian
      - evecs: (V,k) list of eigenvectors of the Laplacian 
      - grad_from_spectral: a (2,V,k) matrix, which maps a scalar field in the spectral space to gradients in the X/Y basis at each vertex
    Note: this is a generalized eigenvalue problem, so the mass matrix matters! The eigenvectors are only othrthonormal with respect tothe mass matrix, like v^H M v, so the mass (given as the diagonal vector massvec) needs to be used in projections, etc.
    """

    device = verts.device
    dtype = verts.dtype
    V = verts.shape[0]
    print_debug = False
    if isinstance(faces, list):
        is_cloud = faces == []
    else:
        is_cloud = faces.numel() == 0

    eps = 1e-6
    
    verts_np = toNP(verts).astype(np.float64)
    if not is_cloud:
        faces_np = toNP(faces)
    frames = build_tangent_frames(verts, faces, normals=normals)
    frames_np = toNP(frames)

    # if vertices contain non-coordinate information, extract only the coordinate information – first three columns
    if verts.shape[1] > 3:
        verts = verts[:, 0:3]

    # Build the scalar Laplacian
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts_np)
    else:
        L, M = robust_laplacian.mesh_laplacian(verts_np, faces_np)
    eps_L = scipy.sparse.identity(V) * eps
    eps_M = scipy.sparse.identity(V) * eps * np.sum(M)
    massvec_np = M.diagonal()

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col
    evals_np, evecs_np = sla.eigsh(L + eps_L, k_eig, M + eps_M, sigma=1e-8)

    # == Build spectral grad & divcurl while we're at it

    # For meshes, we use the same edges as were used to build the Laplacian. For point clouds, use a whole local neighborhood
    if is_cloud:
        grad_mat_np = build_grad_point_cloud(verts, frames)
    else:
        edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype)
        edge_vecs = edge_tangent_vectors(verts, frames, edges)
        grad_mat_np = build_grad(verts, edges, edge_vecs)

    grad_from_spectral_np = grad_mat_np @ evecs_np
    grad_from_spectral_np = np.stack((grad_from_spectral_np.real, grad_from_spectral_np.imag),axis=-1) # split to 2 x real matrix instead of complex
    
    # === Convert back to torch
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)
    grad_from_spectral = torch.from_numpy(grad_from_spectral_np).to(device=device, dtype=dtype)

    return frames, massvec, evals, evecs, grad_from_spectral


def get_all_operators(opts, verts_list, faces_list):
    N = len(verts_list)
            
    frames = [None] * N
    massvec = [None] * N
    evals = [None] * N
    evecs = [None] * N
    grad_from_spectral = [None] * N

    # process in random order
    inds = [i for i in range(N)]
    random.shuffle(inds)
   
    for num, i in enumerate(inds):
        print("get_all_operators() processing {} / {} {:.3f}%".format(num, N, num / N * 100))
        outputs = get_operators(opts, verts_list[i], faces_list[i], opts.k_eig)
        frames[i] = outputs[0]
        massvec[i] = outputs[1]
        evals[i] = outputs[2]
        evecs[i] = outputs[3]
        grad_from_spectral[i] = outputs[4]
        
    return frames, massvec, evals, evecs, grad_from_spectral


def get_operators(opts, verts, faces, k_eig, normals=None, overwrite_cache=False, truncate_cache=False):
    """
    See documentation for compute_operators(). This essentailly just wraps a call to compute_operators, using a cache if possible.
    All arrays are always computed using double precision for stability, then truncated to single precision floats to store on disk, and finally returned as a tensor with dtype/device matching the `verts` input.
    """

    device = verts.device
    dtype = verts.dtype
    verts_np = toNP(verts)
    if isinstance(faces, list):
        is_cloud = faces == []
    else:
        is_cloud = faces.numel() == 0
    if not is_cloud:
        faces_np = toNP(faces)

    # if vertices contain non-coordinate information, extract only the coordinate information – first three columns
    if verts.shape[1] > 3:
        verts = verts[:, 0:3]

    if(np.isnan(verts_np).any()):
        raise RuntimeError("tried to construct operators from NaN verts")

    # Check the cache directory
    # Note 1: Collisions here are exceptionally unlikely, so we could probably just use the hash...
    #         but for good measure we check values nonetheless.
    # Note 2: There is a small possibility for race conditions to lead to bucket gaps or duplicate
    #         entries in this cache. The good news is that that is totally fine, and at most slightly
    #         slows performance with rare extra cache misses.
    found = False
    if opts.eigensystem_cache_dir is not None:
        utils.ensure_dir_exists(opts.eigensystem_cache_dir)
        hash_key_str = str(utils.hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                opts.eigensystem_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")
            
            try:
                # print('loading path: ' + str(search_path))
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]
                cache_k_eig = npzfile["k_eig"].item()

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts, cache_verts)) or (not np.array_equal(faces, cache_faces)):
                    i_cache_search += 1
                    print("hash collision! searching next.")
                    continue

                # If we're overwriting, or there aren't enough eigenvalues, just delete it; we'll create a new
                # entry below more eigenvalues
                if overwrite_cache or cache_k_eig < k_eig:
                    print("  overwiting / not enough eigenvalues --- recomputing")
                    os.remove(search_path)
                    break

                # This entry matches! Return it.
                found = True
                frames = npzfile["frames"]
                mass = npzfile["mass"]
                evals = npzfile["evals"][:k_eig]
                evecs = npzfile["evecs"][:, :k_eig]
                grad_from_spectral = npzfile["grad_from_spectral"][:,:k_eig,:]

                if truncate_cache and cache_k_eig > k_eig:
                    print("TRUNCATING CACHE {} --> {}".format(cache_k_eig, k_eig))
                    np.savez(search_path,
                             verts=verts_np,
                             frames=frames,
                             faces=faces_np,
                             k_eig=k_eig,
                             mass=mass,
                             evals=evals,
                             evecs=evecs,
                             grad_from_spectral=grad_from_spectral,
                             )


                frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
                grad_from_spectral = torch.from_numpy(grad_from_spectral).to(device=device, dtype=dtype)
                
                break

            except FileNotFoundError:
                print("  cache miss -- constructing operators")
                break
            
            except Exception as E:
                print("unexpected error loading file: " + str(E))
                print("-- constructing operators")
                break

    if not found:

        # No matching entry found; recompute.
        frames, mass, evals, evecs, grad_from_spectral = compute_operators(verts, faces, k_eig, normals=normals)

        dtype_np = np.float32

        # Store it in the cache
        if opts.eigensystem_cache_dir is not None:
            np.savez(search_path,
                     verts=verts_np,
                     frames=toNP(frames).astype(dtype_np),
                     faces=faces_np,
                     k_eig=k_eig,
                     mass=toNP(mass).astype(dtype_np),
                     evals=toNP(evals).astype(dtype_np),
                     evecs=toNP(evecs).astype(dtype_np),
                     grad_from_spectral=toNP(grad_from_spectral).astype(dtype_np),
                     )


    return frames, mass, evals, evecs, grad_from_spectral


def to_basis(values, basis, massvec):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (B,V,D)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,K,D) transformed values
    """
    basisT = basis.transpose(-2, -1)
    return torch.matmul(basisT, values * massvec.unsqueeze(-1))


def from_basis(values, basis):
    """
    Transform data out of an orthonormal basis
    Inputs:
      - values: (K,D)
      - basis: (V,K)
    Outputs:
      - (V,D) reconstructed values
    """
    if values.is_complex() or basis.is_complex():
        return utils.cmatmul(utils.ensure_complex(basis), utils.ensure_complex(values))
    else:
        return torch.matmul(basis, values)


def normalize_positions(pos):
    # center and unit-scale positions
    pos = (pos - torch.mean(pos, dim=-2, keepdim=True))
    scale = torch.max(norm(pos), dim=-1, keepdim=True).values.unsqueeze(-1)
    pos = pos / scale
    return pos


# Finds the k nearest neighbors of source on target.
# Return is two tensors (distances, indices). Returned points will be sorted in increasing order of distance.
def find_knn(points_source, points_target, k, largest=False, omit_diagonal=False, method='brute'):

    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    if method != 'cpu_kd' and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = 'cpu_kd'
        print("switching to cpu_kd knn")

    if method == 'brute':

        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(-1, points_target.shape[0], -1)
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(points_source.shape[0], -1, -1)

        diff_mat = points_source_expand - points_target_expand
        dist_mat = norm(diff_mat)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float('inf')

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result
    
    elif method == 'cpu_kd':

        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = toNP(points_source)
        points_target_np = toNP(points_target)

        # Build the tree
        kd_tree = sklearn.neighbors.KDTree(points_target_np)

        k_search = k+1 if omit_diagonal else k 
        _, neighbors = kd_tree.query(points_source_np, k=k_search)
        
        if omit_diagonal: 
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape((neighbors.shape[0], neighbors.shape[1]-1))

        inds = torch.tensor(neighbors, device=points_source.device, dtype=torch.int64)
        dists = norm(points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds])

        return dists, inds
    
    else:
        raise ValueError("unrecognized method")
