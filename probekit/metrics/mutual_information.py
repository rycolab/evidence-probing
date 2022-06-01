from overrides import overrides
from typing import List, Optional
import torch
import math

from probekit.utils.dataset import ClassificationDataset
from probekit.metrics.metric import Metric
from probekit.models.probe import Probe


class MutualInformation(Metric):
    def __init__(self, dataset: ClassificationDataset, normalize: bool = False, bits: bool = False):
        self._normalize = normalize
        self._bits = bits
        self._dataset = dataset

        self._prob_empirical_class = self._compute_prob_empirical_class()
        self._entropy = self._estimate_entropy()

        super().__init__()

    def _compute(self, probe: Probe, select_dimensions: Optional[List[int]] = None) -> float:
        entropy = self._entropy
        conditional_entropy = self._estimate_conditional_entropy(probe, select_dimensions)

        if self._bits:
            entropy /= math.log(2)
            conditional_entropy /= math.log(2)

        mi = entropy - conditional_entropy
        if self._normalize:
            mi /= entropy

        return mi

    def _compute_prob_empirical_class(self) -> torch.Tensor:
        empirical_counts = torch.tensor(
            [len(words) for property_value, words in self._dataset.items()]).float().to(
                self._dataset.get_device())
        total_count = empirical_counts.sum()
        return empirical_counts / total_count

    def _estimate_entropy(self) -> float:
        empirical_probs = self._prob_empirical_class
        return -empirical_probs.dot(empirical_probs.log()).cpu().item()

    def _estimate_conditional_entropy(self, probe: Probe,
                                      select_dimensions: Optional[List[int]] = None) -> float:
        class_probs = self._prob_empirical_class
        dataset_embeddings_tensors = self._dataset.get_embeddings_tensors(select_dimensions)

        # Estimate class-specific integrals
        # shape: (num_classes,)
        estimated_integrals = torch.stack([
            self._estimate_integral_for_property_value(probe, property_id, embeddings_tensor)
            for property_id, (_, embeddings_tensor) in enumerate(dataset_embeddings_tensors.items())], dim=0)

        # Compute final value
        return class_probs.dot(estimated_integrals).item()

    def _estimate_integral_for_property_value(self, probe: Probe, property_id: int,
                                              embeddings_tensor: torch.Tensor) -> torch.Tensor:
        # TODO: Can probably optimize this
        num_samples = embeddings_tensor.shape[0]

        class_tensor = property_id * torch.ones(num_samples,).to(self._dataset.get_device())

        # Compute probabilities
        # shape: (num_classes,)
        log_prob = probe.log_prob_class_given_input(embeddings_tensor, class_tensor)

        return - (1 / num_samples) * log_prob.sum()
