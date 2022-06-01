from typing import List, Optional
from overrides import overrides
import torch
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from tqdm import trange

from probekit.utils.dataset import FastTensorDataLoader, ClassificationDataset
from probekit.trainers.neural_probe_trainer import NeuralProbeTrainer
from probekit.utils.types import PyTorchDevice, Specification
from probekit.models.discriminative.neural_probe import NeuralProbeModel


class ConvergenceTrainer(NeuralProbeTrainer):
    """
    NOTE: No early stopping heuristic is currently in place.
    """
    def __init__(
            self, model: NeuralProbeModel, dataset: ClassificationDataset, device: PyTorchDevice,
            lr: float = 1e-1, num_epochs: int = 2000, batch_size: Optional[int] = None,
            report_progress: bool = True, l2_regularization: float = 0.1):
        self._regularization = l2_regularization

        super().__init__(
            model=model, dataset=dataset, device=device, decomposable=True, lr=lr, num_epochs=num_epochs,
            batch_size=batch_size, report_progress=report_progress)

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
        # model.reset_parameters()

        # Train model
        # weight-decay parameter is equivalent to the precision of a spherical Gaussian prior.
        optimizer = optim.Adam(model.parameters(), lr=self._lr, weight_decay=0.0)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

        data_loader = FastTensorDataLoader(
            inputs_tensor, values_tensor, batch_size=self._batch_size, shuffle=True)

        t = trange(self._num_epochs, desc="Training neural probe", disable=not self._report_progress,
                   leave=False)
        # prev_loss = None
        losses = list()
        for epoch in t:
            epoch_loss = 0.
            for X, y in data_loader:
                def closure():
                    optimizer.zero_grad()
                    M = X.shape[0]
                    f = model(X)
                    p = parameters_to_vector(model.parameters())
                    loss = N / M * loss_fn(f, y) + 0.5 * self._regularization * p.square().sum()

                    loss.backward()
                    return loss.cpu().item()

                epoch_loss += optimizer.step(closure)

            # WIP:
            # deterministic:
            # if self._batch_size >= N:
            #     if prev_loss is not None:
            #         if prev_loss <= epoch_loss:
            #             # halve learning rate
            #             optimizer.param_groups[0]['lr'] /= 2.
            # # stochastic
            # else:
            #     # decrease the step size every 100 epochs
            #     optimizer.param_groups[0]['lr'] = self._lr * (epoch // 100 + 1)
            # prev_loss = epoch_loss

            losses.append(epoch_loss/N)

            t.set_postfix(loss=epoch_loss/N, lr=optimizer.param_groups[0]["lr"],
                          grad=torch.cat([p.grad.view(-1) for p in model.parameters()]).norm().item()/N)

        return losses

    @overrides
    def _get_specification(self, select_dimensions: List[int]) -> Specification:
        # NOTE: Again, we can ignore `select_dimensions` in our case
        return {
            "model": self._model
        }

