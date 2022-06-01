from typing import List, Optional
from overrides import overrides
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from tqdm import trange

from probekit.utils.dataset import FastTensorDataLoader, ClassificationDataset
from probekit.trainers.neural_probe_trainer import NeuralProbeTrainer
from probekit.utils.types import PyTorchDevice, Specification
from probekit.models.discriminative.neural_probe import NeuralProbeModel
from laplace.laplace import DiagLaplace, KronLaplace


class MargLikTrainer(NeuralProbeTrainer):
    """
    For posterior_structure = 'diag', we can efficiently handle one hyperparameter per NN parameter.
    For posterior_structure = 'kron', we only deal with per-parameter regularization since otherwise
    the efficient structure would break (and hence it is not supported by the Laplace package)
    """

    def __init__(
        self, model: NeuralProbeModel, dataset: ClassificationDataset, device: PyTorchDevice,
        lr: float = 1e-1, num_epochs: int = 2000, batch_size: Optional[int] = None,
        report_progress: bool = True, lr_hyp: float = 1e-1, num_hypersteps: int = 100,
        hyper_frequency: int = 1, posterior_structure: str = 'diag', early_stopping: bool = True):

        if posterior_structure not in ['diag', 'kron']:
            raise ValueError('Can only work with diagonal or Kronecker posterior approx.')
        self._posterior_struct = posterior_structure
        self._prior_precision = None
        self._network_parameters = None
        self._marglik = None
        self._effective_dimensions = None
        self._lr_hyp = lr_hyp
        self._num_hypersteps = num_hypersteps
        self._hyper_frequency = hyper_frequency
        self._early_stopping = early_stopping

        super().__init__(
            model=model, dataset=dataset, device=device, decomposable=True, lr=lr, num_epochs=num_epochs,
            batch_size=batch_size, report_progress=report_progress)

    def _expand_prior_precision(self, prior_prec):
        theta = parameters_to_vector(self._model.parameters())
        device, P = theta.device, len(theta)
        assert prior_prec.ndim == 1
        if len(prior_prec) == 1:  # scalar
            return torch.ones(P, device=self._device) * prior_prec
        elif len(prior_prec) == P:  # full diagonal
            return prior_prec
        else:
            return torch.cat([delta * torch.ones_like(m).flatten()
                              for delta, m in zip(prior_prec, self._model.parameters())])

    @staticmethod
    def _get_kron_eff_params(lap: KronLaplace):
        eff_params = list()
        posterior_precision = lap.posterior_precision
        for lambdas, delta in zip(posterior_precision.eigenvalues, posterior_precision.deltas):
            if len(lambdas) == 1:
                lams = lambdas[0]
            elif len(lambdas) == 2:
                lams = torch.ger(lambdas[0], lambdas[1]).flatten()
            else:
                raise ValueError('Invalid eigendecomposition of Kron.')
            eff_params.append(((lams + delta) / delta).detach().cpu())
        return torch.cat(eff_params)

    @overrides
    def _train_for_dimensions(self, select_dimensions: Optional[List[int]] = None):
        """
        This function should train the probe model (self._model).

        `select_dimensions` can be safely ignored in our use case (we use it for another project where we are
        doing variable selection on the dimensions of BERT embeddings).
        """
        # NOTE: notation
        # use N = number of data points, M = number of samples in batch
        # K = number of outputs (classes), X is a minibatch, y are the labels
        inputs_tensor, values_tensor = self._dataset.get_inputs_values_tensor(select_dimensions)
        N, dim = inputs_tensor.shape
        model = self._model
        # decay learning rate to lr * 1e-3 for standard parameters due to stochastic grads
        min_lr_factor = 1e-3
        gamma = np.exp(np.log(min_lr_factor) / self._num_epochs)

        # Set up parent objective and optimizer
        if self._posterior_struct == 'diag':
            # full diagonal prior precision
            Laplace = DiagLaplace
            theta = parameters_to_vector(model.parameters())
            log_prior_prec = torch.log(torch.ones_like(theta))
        elif self._posterior_struct == 'kron':
            # prior per neural network parameter group
            Laplace = KronLaplace
            n_param_groups = len(list(model.parameters()))
            log_prior_prec = torch.log(torch.ones(n_param_groups, device=self._device))
        else:
            raise ValueError('Invalid posterior structure.')
        log_prior_prec.requires_grad = True
        hyper_optimizer = optim.Adam([log_prior_prec], lr=self._lr_hyp)

        # Set up standard loss and optimizer
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=self._lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        data_loader = FastTensorDataLoader(
            inputs_tensor, values_tensor, batch_size=self._batch_size, shuffle=True)
        setattr(data_loader, 'dataset', N * [0])  # I know, this is nice

        t = trange(self._num_epochs, desc="Training neural probe",
                   disable=not self._report_progress, leave=False)

        best_marglik = np.inf
        best_model = None
        best_precision = None
        best_eff_dim = None
        best_eff_params = None
        losses = list()
        margliks = list()
        for epoch in t:
            epoch_loss = 0
            for X, y in data_loader:
                M = len(y)
                optimizer.zero_grad()
                theta = parameters_to_vector(model.parameters())
                prior_prec = self._expand_prior_precision(torch.exp(log_prior_prec.detach()))
                loss = N / M * criterion(model(X), y) + 0.5 * (prior_prec * theta) @ theta
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu().item() / len(data_loader)
            losses.append(epoch_loss/N)
            scheduler.step()

            # only update hyperparameters every "Frequency" steps
            if (epoch % self._hyper_frequency) != 0:
                continue

            lap = Laplace(model, 'classification')
            lap.fit(data_loader)
            for _ in range(self._num_hypersteps):
                hyper_optimizer.zero_grad()
                prior_prec = torch.exp(log_prior_prec)
                marglik = -lap.log_marginal_likelihood(prior_prec)
                marglik.backward()
                hyper_optimizer.step()
                margliks.append(marglik.item()/N)

            # save best model if improved and always save if no early stopping used.
            if (margliks[-1] < best_marglik) or (not self._early_stopping):
                best_model = deepcopy(model)
                best_precision = deepcopy(prior_prec.detach().cpu())
                best_marglik = margliks[-1]
                best_eff_dim = lap.log_det_ratio.item()
                if self._posterior_struct == 'diag':
                    best_eff_params = (lap.posterior_precision / lap.prior_precision_diag).detach().cpu()
                elif self._posterior_struct == 'kron':
                    best_eff_params = self._get_kron_eff_params(lap)

            t.set_postfix(loss=losses[-1], marglik=margliks[-1],
                          lr=optimizer.param_groups[0]["lr"],
                          lr_hyp=hyper_optimizer.param_groups[0]["lr"])

        self._model.load_state_dict(best_model.state_dict())
        self._effective_dimensions = best_eff_dim
        self._prior_precision = best_precision.numpy().tolist()
        self._marglik = best_marglik
        self._effective_parameters = best_eff_params.numpy().tolist()
        self._network_parameters = [p.detach().cpu().numpy().tolist() for p in model.parameters()]

        return losses, margliks

    @overrides
    def _get_specification(self, select_dimensions: List[int]) -> Specification:
        return {
            "model": self._model
        }
