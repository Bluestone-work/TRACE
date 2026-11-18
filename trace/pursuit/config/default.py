from dataclasses import dataclass
@dataclass
class PSTOConfig:
    num_pursuers:int=2; num_evaders:int=1; lidar_vertical:int=24; lidar_horizontal:int=480
    lidar_downrate:int=4; lidar_range:float=10.; physics_dt:float=1/240; policy_hz:float=30.
    gamma:float=.995; gae_lambda:float=.95; clip_param:float=.2; capture_threshold:float=1.
