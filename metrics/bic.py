from typing import Optional, List
from overrides import overrides
import math

from probekit.utils.dataset import ClassificationDataset
from probekit.metrics.metric import Metric
from probekit.models.probe import Probe


class BIC(Metric):
    """
    Computes the Bayesian Information Criterion (BIC) approximation to the marginal likelihood.
    """
    def __init__(self, dataset: ClassificationDataset):
        super().__init__()

        self._dataset = dataset

    def _compute(self, probe: Probe, select_dimensions: Optional[List[int]] = None) -> float:
        inputs, true = self._dataset.get_inputs_values_tensor(select_dimensions)
        num_samples = inputs.shape[0]
        log_prob = probe.log_prob_class_given_input(inputs, true)

        probe_model = probe.get_underlying_model()
        num_parameters = sum([param.numel() for param in probe_model.parameters()])

        return - (num_parameters / 2) * math.log(num_samples) + log_prob.sum().item()
