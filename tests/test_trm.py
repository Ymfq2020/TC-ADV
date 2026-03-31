from types import SimpleNamespace

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
