"""Compare the port against the actual upstream controller and RotorGroup.

Run in Isaac's Python with torch_site on PYTHONPATH, without starting the app. AST
loading avoids importing the old simulator API while executing the unmodified
upstream controller classes and quaternion utilities.
"""

import ast
import importlib.util
import json
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import yaml

from quadrotor_dynamics import QuadrotorActuator


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(__file__).resolve().parent
    package = root.parent / "isaac-training/third_party/OmniDrones/omni_drones"
    math = load_file("upstream_drone_math", package / "utils/torch.py")
    source = package / "controllers/lee_position_controller.py"
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ("compute_parameters", "LeePositionController")]
    namespace = dict(vars(math), torch=torch, nn=torch.nn, yaml=yaml, osp=osp,
                     __file__=str(source))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    port = QuadrotorActuator(0.016)
    upstream = namespace["LeePositionController"](9.81, port.params)
    rng = np.random.default_rng(7)
    errors = []
    for _ in range(100):
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        state = np.r_[rng.normal(size=3), quaternion, rng.normal(size=6)].astype(np.float32)
        target = rng.uniform(-2, 2, 3).astype(np.float32)
        upstream_cmd = upstream(torch.from_numpy(state[None]),
                                target_vel=torch.from_numpy(target[None])).numpy()[0]
        upstream_forces = (upstream_cmd + 1) * 0.5 * port.max_forces
        port.command(state[3:7], state[7:10], state[10:13], target)
        errors.append(float(np.max(np.abs(upstream_forces - port.requested_forces))))
    assert max(errors) < 5e-5, errors

    rotor_class = load_file("upstream_rotors", package / "actuators/rotor_group.py").RotorGroup
    rotors = rotor_class(port.params["rotor_configuration"], dt=0.016)
    port.reset()
    rotors.throttle.copy_(torch.tensor(port.throttle, dtype=torch.float32))
    motor_error = 0.0
    for _ in range(100):
        commands = rng.uniform(-1, 1, 4).astype(np.float32)
        forces, moments = rotors(torch.from_numpy(commands))
        port_forces, port_moments = port.motor_step((commands + 1) * 0.5 * port.max_forces)
        motor_error = max(motor_error, float(np.max(np.abs(forces.numpy() - port_forces))),
                          float(np.max(np.abs(moments.numpy() - port_moments))))
    assert motor_error < 5e-6, motor_error
    report = {"status": "passed", "controller_samples": 100,
              "max_controller_force_error_N": max(errors), "motor_samples": 100,
              "max_motor_force_or_moment_error": motor_error}
    output = root / "artifacts/controller_parity.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("UPSTREAM_CONTROLLER_PARITY_OK", json.dumps(report))


if __name__ == "__main__":
    main()
