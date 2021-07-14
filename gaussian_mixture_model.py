'''
Implementation of Gaussian Mixture Model by maximizing the log likelihood of the observations.
The number of latent variables is countable and we can thus build and optimize the model.

We have the following challenges when we want to add covariance matrices to the mix : 
- The diagonal values of the covariance matrix need be > 0 > use exp function
- The covariance matrix has to be symmetric
- The non-diagonal values of the covariance matrix need be between 0 and 1 => use sigmoid function 
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from torch.utils.data import DataLoader, Dataset
# Other imports of use
import numpy as np
import tqdm

#############################################################################################################################
# Dataset object
class MyDataset(Dataset):   
    def __init__(self, samples, classes):
        super(MyDataset, self).__init__()
        self.samples = samples    
        self.classes = classes
        
    def __getitem__(self,index):
        return self.samples[index, ::]
    
    def __len__(self):
        return self.samples.size()[0]

#############################################################################################################################
# Model
class PytorchBasedGMM(nn.Module):
    def __init__(self, number_of_gaussian_mixture_components, n_dimension = 2, prior_logits = None, mus = None, covariance_matrices = None, training_epochs = 1000, batch_size = 64, learning_rate = 1e-3):
        super(PytorchBasedGMM, self).__init__()
        self.num_mixture_comps = self.n_mix = number_of_gaussian_mixture_components
        self.n_dimension = n_dimension 
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.parameters_provided = False
        if mus:
            assert self.n_dimension == len(mus[0])
        # In the beginning it makes sense to assign equal probability to all of the gaussian mixture components
        if prior_logits:
            self.prior_logits = torch.FloatTensor(prior_logits)
        else:
            self.prior_logits = nn.Parameter(torch.zeros(self.num_mixture_comps, dtype=torch.float32), requires_grad=True)
        # Get the values for the means of the gaussian distributions
        if mus:
            self.mus = torch.FloatTensor(mus)
        else:
            self.mus = nn.Parameter(torch.randn(self.num_mixture_comps, n_dimension, dtype=torch.float32), requires_grad=True)
        # Get the values for the covariance matrices of the gaussian distributions
        if covariance_matrices:
            self.parameters_provided = True
            self.covariance_matrices_ = torch.FloatTensor(covariance_matrices)
        else:
            self.covariance_matrices_ = nn.Parameter(torch.ones(self.num_mixture_comps, self.n_dimension, self.n_dimension))
            # covariance_matrices_ = torch.zeros(self.num_mixture_comps, n_dimension, n_dimension, dtype = torch.float32)
            # for i in range(self.num_mixture_comps):
            #     temp = torch.FloatTensor(self.n_dimension, self.n_dimension).uniform_(0,1).tril(diagonal=-1) + torch.eye(n_dimension)
            #     param = nn.Parameter(temp)
            #     covariance_matrices_[i, ::] = param + param.transpose(0, 1) # Enforcing the covariance matrix to be symmetric
            # self.covariance_matrices = nn.Parameter(covariance_matrices_)

    def log_likelihood(self, X):
        prior_probs = F.softmax(self.prior_logits, dim = 0)
        all_probs = torch.zeros(X.shape[0], self.num_mixture_comps)
        for i in range(self.num_mixture_comps):
            mean = self.mus[i]
            covariance_matrix = self.covariance_matrices_[i]
            non_diagonals_mask = torch.ones(self.n_dimension, self.n_dimension) - torch.eye(self.n_dimension)
            diagonal_mask = torch.eye(self.n_dimension)
            covariance_matrix = torch.exp(covariance_matrix) * diagonal_mask + torch.tanh(((covariance_matrix + covariance_matrix.transpose(0,1)) / 2)) * non_diagonals_mask
            probs = np.power(2* np.pi, - self.n_dimension/2.) * torch.pow(covariance_matrix.det(), -1./2) * torch.exp(-0.5 * torch.sum((X - mean) * (torch.matmul(covariance_matrix.inverse(), (X - mean).transpose(0, 1))).transpose(0,1), dim = 1))
            all_probs[:, i] = probs
        return torch.sum(torch.log(torch.sum(all_probs * prior_probs, dim = 1)))

    def fit(self, samples, classes = None):
        # Create a dataloader
        optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        dataset_custom = MyDataset(samples, classes)
        # num_workers should be set to 0 for windows
        dataloader_custom = DataLoader(dataset_custom, batch_size = self.batch_size, shuffle = True, num_workers = 0)
        for training_epoch in tqdm.tqdm(range(self.training_epochs)):
            for i, batch in enumerate(dataloader_custom):
                nll = negative_log_likelihood = -self.log_likelihood(batch)
                optimizer.zero_grad()
                nll.backward(retain_graph = True)
                optimizer.step()

    @property
    def covariance_matrices(self):
        if self.parameters_provided:
            return self.covariance_matrices_
        covariance_matrices = torch.zeros(self.num_mixture_comps, self.n_dimension, self.n_dimension)
        for i in range(self.num_mixture_comps):
            covariance_matrix = self.covariance_matrices_[i]
            non_diagonals_mask = torch.ones(self.n_dimension, self.n_dimension) - torch.eye(self.n_dimension)
            diagonal_mask = torch.eye(self.n_dimension)
            covariance_matrix = torch.exp(covariance_matrix) * diagonal_mask + torch.tanh(((covariance_matrix + covariance_matrix.transpose(0,1)) / 2)) * non_diagonals_mask  
            covariance_matrices[i, ::] = covariance_matrix  
        return covariance_matrices        

    def sample(self, num_samples):
        with torch.no_grad():
            # How would we sample from the distribution ?
            # First we sample the class
            prior_probs = F.softmax(self.prior_logits, dim = 0) # Get prior probabilities
            sampling_distribution = distributions.Categorical(prior_probs)
            classes = z_samples = sampling_distribution.sample((num_samples,))

            mus = self.mus[classes]
            # covariance_matrices = self.covariance_matrices[classes]            
            cholesky_matrices = torch.zeros(self.num_mixture_comps, self.n_dimension, self.n_dimension)
            for i in range(self.num_mixture_comps):
                cholesky_matrices[i,::] = torch.cholesky(self.covariance_matrices[i])
            cholesky_matrices = cholesky_matrices[classes]
            
            # Getting any multivariate gaussian distribution from a standard normal distribution
            x = mus +  torch.matmul(cholesky_matrices, torch.randn(num_samples, self.n_dimension).unsqueeze(-1)).squeeze()

        return x, classes



    