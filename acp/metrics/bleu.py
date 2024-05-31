import nltk
from nltk.translate.bleu_score import SmoothingFunction

from acp.metrics.base import BaseMetric


class BLEU(BaseMetric):
    def evaluate(self, reference: str, hypothesis: str, ignore_case=True) -> float:
        cc = SmoothingFunction()

        if ignore_case:
            reference = reference.lower()
            hypothesis = hypothesis.lower()

        if len(hypothesis) == 1 and len(reference) == 1:
            if hypothesis == reference:
                return 1.0
            else:
                return 0.0
        return nltk.translate.bleu(
            [reference], hypothesis, smoothing_function=cc.method4
        )
