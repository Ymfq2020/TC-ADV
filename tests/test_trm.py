from types import SimpleNamespace

import torch

from tc_adv.discriminators.trm import TemporalRationalityModule
from tc_adv.config.schemas import TRMConfig


def _sample(subject: str, obj: str, timestamp: int):
    return SimpleNamespace(quadruple=SimpleNamespace(subject=subject, object=obj, timestamp=timestamp))


def test_trm_scores_decay_outside_active_window():
    module = TemporalRationalityModule(TRMConfig())
    module.build_index(
        [
            _sample("A", "B", 1),
            _sample("A", "C", 2),
            _sample("A", "D", 3),
        ]
    )
    near = module.normalized_activity_score("A", 2)
    far = module.normalized_activity_score("A", 20)
    assert near > far


def test_trm_normalization_is_bounded():
    module = TemporalRationalityModule(TRMConfig())
    module.build_index([_sample("A", "B", 5), _sample("A", "C", 6)])
    score = module.normalized_activity_score("A", 5)
    assert 0.0 <= score <= 1.0 + 1e-6


def test_trm_probability_keeps_monotonicity_after_weight_updates():
    module = TemporalRationalityModule(TRMConfig())
    with torch.no_grad():
        module.linear.weight.fill_(-10.0)
        module.linear.bias.zero_()
    high_support = module.probability_from_scores(torch.tensor([0.9]), torch.tensor([0.9]))
    low_support = module.probability_from_scores(torch.tensor([0.1]), torch.tensor([0.1]))
    assert float(high_support[0].detach()) < float(low_support[0].detach())
