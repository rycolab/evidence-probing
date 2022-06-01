from typing import Optional, List, Tuple
from overrides import overrides
import torch
from torch.autograd import grad
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import math
from tqdm import tqdm
from backpack import backpack, extend, memory_cleanup
from backpack.extensions import BatchGrad
from backpack.context import CTX
from backpack.extensions import DiagGGNExact

from probekit.utils.dataset import ClassificationDataset
from probekit.metrics.metric import Metric
from probekit.models.probe import Probe


EPS = 1e-9


def cleanup(module):
    for child in module.children():
        cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)


def diag_ggn(model, true_hessian):
    if true_hessian:
        dh = torch.cat([p.diag_h.data.flatten() for p in model.parameters()])
        return dh.detach()
    else:
        dggn = torch.cat([p.diag_ggn_exact.data.flatten() for p in model.parameters()])
        return dggn.detach()


def jacobians(model, data, K):
    # K is model-output size or N classes
    # Jacobians are batch x params x output
    model = extend(model)
    to_stack = []
    for i in range(K):
        model.zero_grad()
        out = model(data)
        with backpack(BatchGrad()):
            if K > 1:
                out[:, i].sum().backward()
            else:
                out.sum().backward()
            to_cat = []
            for param in model.parameters():
                to_cat.append(param.grad_batch.detach().reshape(data.shape[0], -1))
            Jk = torch.cat(to_cat, dim=1)
        to_stack.append(Jk)
        if i == 0:
            f = out.detach()
    # cleanup
    model.zero_grad()
    CTX.remove_hooks()
    cleanup(model)
    return torch.stack(to_stack, dim=2), f


class LaplaceGGN(Metric):
    """
    Computes the Laplace-GGN approximation to the marginal likelihood.
    The Hessian is approximated by a Jacobian product with the second
    derivative of the loss.
    """
    def __init__(self, dataset: ClassificationDataset, prior_precision: float, cov_type='full',
                 batch_size=100, compute_effective_parameters=False):
        super().__init__()
        self._cov_type = cov_type
        self._dataset = dataset
        self._prior_prec = prior_precision
        self._batch_size = batch_size
        self._effective_params = compute_effective_parameters

    def _compute(self, probe: Probe, select_dimensions: Optional[List[int]] = None) -> float:
        delta = self._prior_prec
        Xs, ys = self._dataset.get_inputs_values_tensor(select_dimensions)
        K = ys.max() + 1
        # loader = DataLoader([(Xi, yi) for Xi, yi in zip(X, y)], batch_size=self._batch_size,
        #                     shuffle=False)
        model = probe.get_underlying_model()
        theta = parameters_to_vector(model.parameters()).detach()
        # p(D) \approx log p(data | theta) - 1/2 d^T P_o d - 1/2 log det P_0^{-1}/Sigma
        # where Sigma = [sum_{i=1}^N J_i^T H(f_i) J_i + P_0]^{-1}
        # and P_0 = delta * I_P is the prior precision, theta are the parameters

        if self._cov_type == 'full':
            # H(f_i) = second derivative of log likelihood wrt. f, i.e., p - pp^T = Var(y) (exp family)
            # log det ratio: log det Sigma_0/Sigma =...= log det [sum_i delta^{-1} * J_i^T Lam_i J_i + I_P]
            # = log det W
            # clamp ensures positive definite matrix. EPS chosen very conservatively
            log_prob = 0.
            W = torch.eye(len(theta), device=theta.device)
            for i in range(0, len(Xs), self._batch_size):
                X, y = Xs[i:i+self._batch_size], ys[i:i+self._batch_size]
                Js, f = jacobians(model, X, K)
                p = torch.softmax(f, dim=-1)
                p_safe = torch.clamp(p, EPS, 1 - EPS)
                Hs = torch.diag_embed(p_safe) - torch.einsum('ij,ik->ijk', p_safe, p_safe)
                likelihood = Categorical(logits=f)
                log_prob += likelihood.log_prob(y).sum().cpu().item()
                W += torch.einsum('npk,nkc,nqc->pq', Js, Hs, Js) / delta
            # theoretically, this log-determinant is finite since the determinant is non-zero.
            # practically, we don't have this so there is a cascade of ever increasing runtime below
            log_det_ratio = W.logdet().cpu().item()
            if math.isnan(log_det_ratio):
                # use double precision
                W = W.double()
                log_det_ratio = W.logdet().cpu().item()
                if math.isnan(log_det_ratio):  # still..
                    print('last resort: clamped eigendecomposition for logdet')
                    # take only eigvals and the real part of them
                    # we know that \forall i : eig_val_i >= 1.
                    # hence, if theyre not, we can clamp them
                    eig_vals = torch.eig(W)[0][:, 0]
                    eig_vals = torch.clamp(eig_vals, min=1.)
                    log_det_ratio = torch.sum(torch.log(eig_vals)).cpu().item()
        elif self._cov_type == 'diag':
            nn_loss = torch.nn.CrossEntropyLoss(reduction='sum')
            model = extend(model)
            lossfunc = extend(nn_loss)
            log_prob = 0.
            precision = torch.ones_like(theta, device=theta.device)
            for i in range(0, len(Xs), self._batch_size):
                X, y = Xs[i:i+self._batch_size], ys[i:i+self._batch_size]
                f = model(X)
                loss = lossfunc(f, y)
                with backpack(DiagGGNExact()):
                    loss.backward()
                precision += diag_ggn(model, False) / delta
                likelihood = Categorical(logits=f.detach())
                log_prob += likelihood.log_prob(y).sum().cpu().item()
            log_det_ratio = torch.sum(torch.log(precision)).cpu().item()
        else:
            raise ValueError('invalid cov type')

        # scatter
        scatter = theta.square().sum().cpu().item() * delta

        # putting it all together
        if self._effective_params:
            return log_det_ratio
        return log_prob - 0.5 * (scatter + log_det_ratio)

