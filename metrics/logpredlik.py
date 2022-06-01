from typing import Optional, List
from overrides import overrides
import math

from probekit.utils.dataset import ClassificationDataset
from probekit.metrics.metric import Metric
from probekit.models.probe import Probe


class LogPredictiveLikelihood(Metric):
    """
    Computes the log predictive likelihood, i.e., on non-training data.
    Typically, this is done by computing the predictive integral
    $p(y_*|theta, x_*) p(theta|data) dtheta$ which amounts simply to
    p(y_*|theta_map) when using MAP inference.
    """
    def __init__(self, dataset: ClassificationDataset, normalize=True):
        super().__init__()
        self._dataset = dataset
        self._normalize = normalize

    def _compute(self, probe: Probe, select_dimensions: Optional[List[int]] = None) -> float:
        inputs, true = self._dataset.get_inputs_values_tensor(select_dimensions)
        log_prob = probe.log_prob_class_given_input(inputs, true).detach().sum().cpu().item()
        if self._normalize:
            log_prob /= len(inputs)
        return log_prob

