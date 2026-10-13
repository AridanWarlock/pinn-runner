"""Learning-rate boundary and resume semantics use optimizer steps."""

from copy import deepcopy

import pytest
import torch

from pinnstorch.runner.settings import Reader, ScheduleSettings, Training
from pinnstorch.training.optimization import ConstantRate, ExponentialRate


def test_exponential_endpoints_and_resume_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    scheduler = ExponentialRate(10, 0.1).build(optimizer)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    for _ in range(4):
        optimizer.zero_grad()
        parameter.square().backward()
        optimizer.step()
        scheduler.step()
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed_optimizer = torch.optim.Adam([resumed_parameter], lr=0.001)
    resumed_scheduler = ExponentialRate(10, 0.1).build(resumed_optimizer)
    resumed_scheduler.load_state_dict(deepcopy(scheduler.state_dict()))
    resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    for _ in range(6):
        for param, opt, sched in (
            (parameter, optimizer, scheduler),
            (resumed_parameter, resumed_optimizer, resumed_scheduler),
        ):
            opt.zero_grad()
            param.square().backward()
            opt.step()
            sched.step()
        assert optimizer.param_groups[0]["lr"] == resumed_optimizer.param_groups[0]["lr"]
        torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)
    assert scheduler.last_epoch == 10
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0001)
    assert ConstantRate().build(optimizer) is None


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1, float("nan"), True])
def test_invalid_decay_rejected(ratio: object) -> None:
    with pytest.raises(ValueError):
        ScheduleSettings.read(Reader({"kind": "exponential", "final_ratio": ratio}, "schedule"))


def test_decay_needs_explicit_step_budget() -> None:
    with pytest.raises(ValueError, match="explicit"):
        Training.read(
            Reader({"schedule": {"kind": "exponential", "final_ratio": 0.1}}, "training")
        )
    assert Training.read(Reader({}, "training")).schedule.kind == "constant"
