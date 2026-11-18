import torch
import torch.nn.functional as F
class PSTOLidarAdapter:
    def __init__(self, vertical=24,horizontal=480,downrate=4,range_m=10.): self.vertical=vertical; self.horizontal=horizontal; self.downrate=downrate; self.range=range_m
    @property
    def down_shape(self): return (self.vertical//self.downrate,self.horizontal//self.downrate)
    def downsample(self, scan):
        if scan.shape[-2:]!=(self.vertical,self.horizontal): raise ValueError(f'expected {(self.vertical,self.horizontal)}, got {tuple(scan.shape[-2:])}')
        return F.max_pool2d(scan.reshape(-1,1,self.vertical,self.horizontal),self.downrate,self.downrate).reshape(*scan.shape[:-2],*self.down_shape)
    def normalize(self,scan): return torch.clamp(scan/self.range,0.,1.)
