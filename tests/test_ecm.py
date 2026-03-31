from tc_adv.config.schemas import ECMConfig
from tc_adv.discriminators.ecm import EvolutionaryConsistencyModule, merge_neighbor_histories, sinusoidal_time_encoding


def test_merge_neighbor_histories_filters_future_neighbors():
    neighbors, deltas = merge_neighbor_histories(
        subject_neighbors=["B", "C"],
        object_neighbors=["D"],
        subject_deltas=[0.0, -1.0],
        object_deltas=[2.0],
        history_window=4,
    )
    assert "C" not in neighbors
    assert all(delta >= 0.0 for delta in deltas)


def test_time_encoding_dimension_matches_config():
    encoding = sinusoidal_time_encoding([0.0, 1.0, 2.0], dim=8)
    assert len(encoding) == 3
    assert len(encoding[0]) == 8


def test_ecm_python_path_returns_probability():
    module = EvolutionaryConsistencyModule(embedding_dim=4, config=ECMConfig(hidden_dim=4, time_encoding_dim=4, num_heads=1))
    prob = module.probability(
        subject_embed=[[0.1, 0.2, 0.3, 0.4]],
        relation_embed=[[0.0, 0.1, 0.0, 0.1]],
        object_embed=[[0.2, 0.1, 0.0, -0.1]],
        history_entity_embed=[[[0.1, 0.1, 0.1, 0.1]]],
        history_deltas=[[1.0]],
        history_mask=[[True]],
    )
    assert 0.0 <= prob[0] <= 1.0
