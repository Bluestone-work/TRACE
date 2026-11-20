# PSTO follow-up training recommendations

The current evidence does not justify another paper-scale MAPPO run. The next
work should isolate environment and control errors before changing the reward
or network.

1. **Instrument one episode.** Log PhysX contact pairs, contact impulses,
   obstacle center and scale, raw/downsampled LiDAR minima, pose/velocity,
   reward terms, termination flags, and the action in body and world frames at
   every 30 Hz policy step.
2. **Calibrate the flight path.** Run hover and fixed-velocity tests for one,
   two, and three Hummingbirds in an empty scene. Compare position response,
   altitude drift, and yaw response against the official Lee controller in its
   own Isaac Sim 4.1 environment once that environment is installed.
3. **Calibrate perception.** Put a single known cuboid at controlled distances
   and compare native ray hits, no-hit representation, 4x4 downsampling, and
   visibility masks. Do not tune collision thresholds until this table agrees.
4. **Use a collision-free open-space curriculum only as a diagnostic.** Keep
   the reward and network fixed, run a short fixed-seed MAPPO update in an
   empty scene, and verify that the policy can produce stable trajectories.
5. **Reintroduce obstacles gradually.** Increase obstacle density after the
   controller and LiDAR tests pass. Report each scene distribution separately;
   do not call this official 1024-environment reproduction.
6. **Then train with multiple seeds.** Use the bounded native configuration,
   independent evaluation scenes, and report capture, collision, escape,
   truncation, visibility, and trajectory-prediction metrics together.

The immediate blocker is evidence, not a missing long-run budget: the current
14/20 collision episodes cannot yet be separated into true PhysX contact and
conservative geometric classification from the saved rollout alone.
