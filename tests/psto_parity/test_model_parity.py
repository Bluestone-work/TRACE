import importlib.util
from pathlib import Path

import torch

from trace.pursuit.models import FullModel as TraceFullModel


def _official_model_module():
    path = Path(__file__).parents[2] / "upstream/psto-aav-pursuit-official/models/model.py"
    spec = importlib.util.spec_from_file_location("psto_official_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dual_stream_model_structure_and_output_parity():
    official = _official_model_module()
    torch.manual_seed(123)
    reference = official.FullModel()
    trace = TraceFullModel()
    trace.load_state_dict(reference.state_dict())
    image = torch.randn(3, 2, 6, 120)
    state = torch.randn(3, 13)
    with torch.no_grad():
        expected = reference(image, state)
        actual = trace(image, state)
    assert expected.shape == actual.shape == (3, 48)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
