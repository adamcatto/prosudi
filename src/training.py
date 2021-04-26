from torch.utils.tensorboard import SummaryWriter



# == utils / systems config

binding_site_writer = SummaryWriter('logs/binding_site')
ppi_writer = SummaryWriter('logs/ppi')
torch.backends.cudnn.deterministic = True

# == hyperparameters

lr = 1e-3
num_eigenvecs = 128
batch_size = 1

