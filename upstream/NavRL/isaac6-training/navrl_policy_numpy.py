"""NumPy inference for the released NavRL policy.

The Isaac Sim 6 standalone Python is Python 3.12 while the original NavRL
environment is Python 3.10. Exporting the checkpoint once avoids importing two
incompatible simulator runtimes in one process.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.clip(x, -30.0, 30.0)))


def _leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, x, 0.01 * x)


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5) * weight + bias


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def _conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: tuple[int, int]) -> np.ndarray:
    # x: C,H,W; weight: Cout,Cin,Kh,Kw. NavRL uses small 36x4 maps.
    sh, sw = stride
    ph, pw = weight.shape[2] // 2, weight.shape[3] // 2
    x = np.pad(x, ((0, 0), (ph, ph), (pw, pw)), mode="constant")
    cout, _, kh, kw = weight.shape
    oh = (x.shape[1] - kh) // sh + 1
    ow = (x.shape[2] - kw) // sw + 1
    y = np.empty((cout, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            patch = x[:, i * sh : i * sh + kh, j * sw : j * sw + kw]
            y[:, i, j] = np.sum(weight * patch[None, ...], axis=(1, 2, 3)) + bias
    return y


class NavRLPolicy:
    """Deterministic mean-action evaluator for the released checkpoint."""

    def __init__(self, bundle: str | Path | None = None) -> None:
        if bundle is None:
            bundle = Path(__file__).resolve().parent / "navrl_checkpoint_numpy.npz"
        z = np.load(bundle)
        self.w = {k: z[k] for k in z.files}

    def _get(self, name: str) -> np.ndarray:
        return self.w[name.replace('.', '__')]

    def __call__(self, state: np.ndarray, lidar: np.ndarray, direction: np.ndarray,
                 dynamic_obstacle: np.ndarray | None = None) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(1, 8)
        lidar = np.asarray(lidar, dtype=np.float32).reshape(1, 36, 4)
        direction = np.asarray(direction, dtype=np.float32).reshape(1, 3)
        if dynamic_obstacle is None:
            dynamic_obstacle = np.zeros((1, 5, 10), dtype=np.float32)
        dynamic_obstacle = np.asarray(dynamic_obstacle, dtype=np.float32).reshape(1, 5, 10)

        x = _elu(_conv2d(lidar, self._get('feature_extractor.module.0.module.0.weight'),
                         self._get('feature_extractor.module.0.module.0.bias'), (1, 1)))
        x = _elu(_conv2d(x, self._get('feature_extractor.module.0.module.2.weight'),
                         self._get('feature_extractor.module.0.module.2.bias'), (2, 1)))
        x = _elu(_conv2d(x, self._get('feature_extractor.module.0.module.4.weight'),
                         self._get('feature_extractor.module.0.module.4.bias'), (2, 2)))
        x = x.reshape(1, -1)
        x = _linear(x, self._get('feature_extractor.module.0.module.7.weight'),
                    self._get('feature_extractor.module.0.module.7.bias'))
        x = _layer_norm(x, self._get('feature_extractor.module.0.module.8.weight'),
                        self._get('feature_extractor.module.0.module.8.bias'))

        d = dynamic_obstacle.reshape(1, -1)
        d = _linear(d, self._get('feature_extractor.module.1.module.1.0.weight'),
                    self._get('feature_extractor.module.1.module.1.0.bias'))
        d = _layer_norm(_leaky_relu(d), self._get('feature_extractor.module.1.module.1.2.weight'),
                        self._get('feature_extractor.module.1.module.1.2.bias'))
        d = _linear(d, self._get('feature_extractor.module.1.module.1.3.weight'),
                    self._get('feature_extractor.module.1.module.1.3.bias'))
        d = _layer_norm(_leaky_relu(d), self._get('feature_extractor.module.1.module.1.5.weight'),
                        self._get('feature_extractor.module.1.module.1.5.bias'))

        f = np.concatenate([x, state, d], axis=-1)
        f = _linear(f, self._get('feature_extractor.module.3.module.0.weight'),
                    self._get('feature_extractor.module.3.module.0.bias'))
        f = _layer_norm(_leaky_relu(f), self._get('feature_extractor.module.3.module.2.weight'),
                        self._get('feature_extractor.module.3.module.2.bias'))
        f = _linear(f, self._get('feature_extractor.module.3.module.3.weight'),
                    self._get('feature_extractor.module.3.module.3.bias'))
        f = _layer_norm(_leaky_relu(f), self._get('feature_extractor.module.3.module.5.weight'),
                        self._get('feature_extractor.module.3.module.5.bias'))

        alpha = 1.0 + np.log1p(np.exp(np.clip(_linear(f, self._get('actor.module.0.module.alpha_layer.weight'),
                                                   self._get('actor.module.0.module.alpha_layer.bias')), -30, 30)))
        beta = 1.0 + np.log1p(np.exp(np.clip(_linear(f, self._get('actor.module.0.module.beta_layer.weight'),
                                                  self._get('actor.module.0.module.beta_layer.bias')), -30, 30)))
        local = (alpha / (alpha + beta) * 2.0 - 1.0) * 2.0
        # Reproduce quick-demo's local target frame conversion for a Z-up world.
        gx, gy, gz = direction[0]
        norm = max(float(np.hypot(gx, gy)), 1e-6)
        ex = np.array([gx / norm, gy / norm, 0.0], dtype=np.float32)
        ey = np.array([-ex[1], ex[0], 0.0], dtype=np.float32)
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        world = local[0, 0] * ex + local[0, 1] * ey + local[0, 2] * ez
        return world.astype(np.float32)
