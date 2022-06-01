from typing import Optional, List, Tuple
from overrides import overrides
import torch
from torch.autograd import grad
from torch.nn.utils import parameters_to_vector
from torch.distributions import Categorical
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
from backpack import backpack, extend, memory_cleanup
from backpack.extensions import DiagHessian

from probekit.utils.dataset import ClassificationDataset
from probekit.metrics.metric import Metric
from probekit.models.probe import Probe

from metrics.laplace_ggn import diag_ggn


class Laplace(Metric):
    """
    Computes the Laplace-GGN approximation to the marginal likelihood.
    The Hessian is approximated by a Jacobian product with the second
    derivative of the loss.
    """
    def __init__(self, dataset: ClassificationDataset, prior_precision: float, cov_type='diag',
                 batch_size=100):
        super().__init__()
        self._cov_type = cov_type
        self._dataset = dataset
        self._prior_prec = prior_precision
        self._batch_size = batch_size

    def _compute(self, probe: Probe, select_dimensions: Optional[List[int]] = None) -> float:
        delta = self._prior_prec
        X, y = self._dataset.get_inputs_values_tensor(select_dimensions)
        loader = DataLoader([(Xi, yi) for Xi, yi in zip(X, y)], batch_size=self._batch_size,
                            shuffle=False)
        model = probe.get_underlying_model()
        theta = parameters_to_vector(model.parameters()).detach()
        if self._cov_type == 'diag':
            nn_loss = torch.nn.CrossEntropyLoss(reduction='sum')
            model = extend(model)
            lossfunc = extend(nn_loss)
            precision = torch.ones_like(theta, device=theta.device)
            log_prob = 0.
            for X, y in loader:
                f = model(X)
                loss = lossfunc(f, y)
                with backpack(DiagHessian()):
                    loss.backward()
                precision += diag_ggn(model, True) / delta
                likelihood = Categorical(logits=f.detach())
                log_prob += likelihood.log_prob(y).sum().cpu().item()
            log_det_ratio = torch.sum(torch.log(precision)).cpu().item()
        else:
            raise ValueError('invalid cov type, only diag supported')

        # scatter
        scatter = theta.square().sum().cpu().item() * delta

        # putting it all together
        return log_prob - 0.5 * (scatter + log_det_ratio)

