import torch
import torch.nn.functional as F

from trace.pursuit.lidar import PSTOLidarAdapter


def test_lidar_downsample_matches_official_max_pool():
    scan = torch.arange(2 * 24 * 480, dtype=torch.float32).reshape(2, 24, 480)
    expected = F.max_pool2d(scan.unsqueeze(1), kernel_size=(4, 4), stride=(4, 4)).squeeze(1)
    actual = PSTOLidarAdapter().downsample(scan)
    assert actual.shape == (2, 6, 120)
    assert torch.equal(actual, expected)


def test_lidar_proximity_conversion_is_explicit():
    proximity = torch.tensor([0.0, 2.0, 9.5, 10.0])
    hit_distance = 10.0 - proximity
    assert torch.equal(hit_distance, torch.tensor([10.0, 8.0, 0.5, 0.0]))
