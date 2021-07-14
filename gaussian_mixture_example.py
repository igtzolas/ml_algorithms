from gaussian_mixture_model import PytorchBasedGMM

# Some hyperparameters
number_of_gaussian_mixture_components = 3
n_dimension = 2
n_samples = 3000

#############################################################################################################################
prior_logits = [2., 3., 4.]
mus = [[-5, 5], [5, 5], [0, -5]]
covariance_matrices = [ [[1., 0.],[0., 1.]], [[2., 0.],[0., 2.]], [[5., 0],[0, 1.]] ]
# print(F.softmax(torch.tensor(prior_logits)))
source = PytorchBasedGMM(number_of_gaussian_mixture_components, n_dimension = n_dimension, prior_logits = prior_logits, mus = mus, covariance_matrices = covariance_matrices)
# Generating samples from the prior
samples, classes = source.sample(n_samples)

#############################################################################################################################

tracker = PytorchBasedGMM(number_of_gaussian_mixture_components, training_epochs=300, learning_rate=0.01)
tracker.fit(samples)
