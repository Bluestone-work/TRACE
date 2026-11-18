# PSTO reproduction notes

PSTO uses decentralized actors with a centralized critic. Each actor receives a 13-D
state (quaternion, body velocity, angular velocity, previous 3-D action), a downsampled
24x480 LiDAR image, and a heatmap encoding the evader/team intent. The heatmap uses
positive evader values and negative teammate values with temporal decay. Evader motion is
scripted using pursuer repulsion, obstacle repulsion, boundary bias and altitude control.

The actor samples two independent Beta variables. The first controls planar direction and
the second controls speed. The resulting body velocity is rotated by yaw and sent to the
Lee velocity controller. It is not a four-motor command.

The auxiliary LSTM consumes observed historical positions plus `agent_dt` and predicts
future trajectory points. Future ground truth is used only as a supervised target during
training; it is not exposed to the actor.

This repository separates the official snapshot from the TRACE adapter so dependency
changes for Isaac Sim 6 cannot alter the original implementation.
