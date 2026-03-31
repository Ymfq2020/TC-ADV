from tc_adv.training.objectives import StepRatioScheduler, anneal_temperature, dynamic_margin


def test_dynamic_margin_monotonic():
    low = dynamic_margin(0.1, alpha=1.0, beta=2.5)
    high = dynamic_margin(0.9, alpha=1.0, beta=2.5)
    assert high > low > 0.0


def test_dynamic_margin_keeps_floor_when_probability_zero():
    margin = dynamic_margin(0.0, alpha=1.0, beta=2.5)
    assert margin >= 1.0


def test_temperature_annealing_respects_floor():
    assert anneal_temperature(0.1, anneal_rate=0.5, min_temp=0.05) == 0.05


def test_scheduler_maintains_3_to_1_ratio():
    scheduler = StepRatioScheduler(g_steps=3, d_steps=1)
    assert scheduler.cycle() == ["G", "G", "G", "D"]
