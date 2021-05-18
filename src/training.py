import os

import torch
from torch._C import device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logit_loss
import progressbar

from dataset import SCPDBPointCloudDataset
from models import DiffusionNetBindingSite
from preprocessing import construct_ground_truth_segmentation


# == utils / systems config
print('setting up systems stuff...')
binding_site_writer = SummaryWriter('output/logs/binding_site')
ppi_writer = SummaryWriter('output/logs/ppi')
torch.backends.cudnn.deterministic = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# == hyperparameters

lr = 1e-3
num_protein_eigenvecs = 128
num_ligand_eigenvecs = 32
batch_size = 1
num_epochs = 200
decay_rate = 0.5
decay_every = 100

# == paths
root_dir = '../data/'
trained_models_dir = os.path.join(root_dir, 'output', 'trained_models')
model_save_path = os.path.join(trained_models_dir, 'binding_site_seg.pth')

# == datasets
print('setting training datasets...')
scpdb_dataset_train = SCPDBPointCloudDataset(root=root_dir, train=True)
scpdb_dataloader_train = DataLoader(scpdb_dataset_train, batch_size=None)

print('setting testing datasets...')
scpdb_dataset_test = SCPDBPointCloudDataset(root=root_dir, train=False)
scpdb_dataloader_test = DataLoader(scpdb_dataset_test, batch_size=None)

# == model
print('building model')
net = DiffusionNetBindingSite(input_size=23)
net = net.to(device)

# == training
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

def train(epoch):
    if epoch > 0 and epoch % decay_every == 0:
        lr = lr * decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

    net.train()
    optimizer.zero_grad()
    
    jaccard_indices = []

    for ligand, protein, site in progressbar.progressbar(scpdb_dataloader_train):
        ligand = ligand.to(device)
        protein = protein.to(device)
        site = site.to(device)

        segmentation = net(protein, ligand)
        ground_truth_segmentation = construct_ground_truth_segmentation(site)
        ground_truth_segmentation = torch.from_numpy(ground_truth_segmentation)

        loss = bce_logit_loss(segmentation, ground_truth_segmentation)
        loss.backward()

        # track accuracy
        predictions = (segmentation > 0.5).int()

        true_positives = torch.logical_and(predictions, ground_truth_segmentation)
        false_positives = torch.greater(predictions, ground_truth_segmentation)
        false_negatives = torch.less(predictions, ground_truth_segmentation)

        num_correct = torch.sum(true_positives)
        iou = num_correct / (num_correct + torch.sum(false_positives) + torch.sum(false_negatives))
        jaccard_indices.append(iou)

        # optimize

        optimizer.step()
        optimizer.zero_grad()
    
    mean_iou = sum(jaccard_indices / len(jaccard_indices))
    print("Epoch {} - Train: {:06.4f}".format(epoch, mean_iou))


for epoch in range(num_epochs):
    train(epoch)