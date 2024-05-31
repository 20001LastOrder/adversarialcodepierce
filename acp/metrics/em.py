from typing import Any

from acp.metrics.base import BaseMetric


class EM(BaseMetric):
    def evaluate(self, ground_truth: Any, predicted: Any) -> float:
        if ground_truth == predicted:
            return 1.0
        return 0.0
