import torch
import numpy as np
import os
from typing import List, Tuple

class TurboQuantizer_V2:
    def __init__(self, dim: int, bits: int = 3):
        self.dim = dim
        self.bits = bits
        self.levels_count = 2**bits
        
        # 1. Random Orthogonal Rotation Matrix (Fixed)
        # Using a fixed seed for reproducibility of the rotation across sessions
        torch.manual_seed(42)
        mat = torch.randn(dim, dim)
        q, _ = torch.linalg.qr(mat)
        self.rotation = q.float()
        
        # 2. Fixed Gaussian Quantization Levels (No learning needed)
        # Pre-calculate levels based on standard normal distribution N(0, 1)
        with torch.no_grad():
            sample = torch.randn(1000000)
            self.levels = torch.quantile(sample, torch.linspace(0, 1, self.levels_count)).float()

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, dim] or [dim]
        if x.dim() == 1:
            return torch.matmul(x, self.rotation)
        return torch.matmul(x, self.rotation)

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_single = (x.dim() == 1)
        if is_single:
            x = x.unsqueeze(0)
        
        y = self.rotate(x)
        
        # find closest level
        # y: [N, dim], levels: [L] -> diff: [N, dim, L]
        diff = torch.abs(y.unsqueeze(-1) - self.levels.unsqueeze(0).unsqueeze(0))
        indices = torch.argmin(diff, dim=-1) # [N, dim]
        
        # Calculate residuals
        # y_hat: [N, dim]
        y_hat = self.levels[indices]
        residual = y - y_hat
        
        # 1-bit sign and scalar scale
        signs = torch.sign(residual).to(torch.int8)
        scales = torch.std(residual, dim=1).to(torch.float16) # [N]
        
        if is_single:
            return indices.squeeze(0), signs.squeeze(0), scales.squeeze(0)
        return indices, signs, scales

class TurboDiskStore:
    def __init__(self, dim: int, bits: int = 3, storage_dir: str = "turbo_storage",
                 quantizer=None):
        self.dim = dim
        self.bits = bits
        
        if quantizer is not None:
            self.quantizer = quantizer
            self.use_eden = hasattr(quantizer, '_scale_bias')
        else:
            self.quantizer = TurboQuantizer_V2(dim, bits)
            self.use_eden = False
            
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.indices_path = os.path.join(storage_dir, "indices.bin")
        self.signs_path = os.path.join(storage_dir, "signs.bin")
        self.scales_path = os.path.join(storage_dir, "scales.bin")
        self.scale_s_path = os.path.join(storage_dir, "scale_s.bin")
        
        self.count = 0
        self._setup_mmaps()

    def _setup_mmaps(self):
        self.max_capacity = 1_000_000 
        
        self.indices_mmap = np.memmap(self.indices_path, dtype='uint8', mode='w+',
                                      shape=(self.max_capacity, self.dim))
        self.signs_mmap = np.memmap(self.signs_path, dtype='int8', mode='w+',
                                    shape=(self.max_capacity, self.dim))
        self.scales_mmap = np.memmap(self.scales_path, dtype='float16', mode='w+',
                                     shape=(self.max_capacity,))
        self.scale_s_mmap = np.memmap(self.scale_s_path, dtype='float32', mode='w+',
                                      shape=(self.max_capacity,))

    def add(self, x: torch.Tensor):
        if self.count >= self.max_capacity:
            raise MemoryError("Disk store reached max capacity.")
        
        if self.use_eden:
            indices, r_signs, r_scale, S = self.quantizer.quantize(x)
            signs = r_signs if r_signs is not None else torch.zeros_like(indices, dtype=torch.int8)
            scale = r_scale if r_scale is not None else torch.tensor(0.0)
            s_val = S.item() if S.dim() == 0 else S.squeeze(0).item()
        else:
            indices, signs, scale = self.quantizer.quantize(x)
            s_val = 1.0
        
        # Store unpacked directly (eliminates unpack_bits/unpack_signs at search time)
        self.indices_mmap[self.count] = indices.cpu().numpy().astype('uint8')
        self.signs_mmap[self.count] = signs.cpu().numpy().astype('int8')
        self.scales_mmap[self.count] = scale.item() if hasattr(scale, 'item') else float(scale)
        self.scale_s_mmap[self.count] = s_val
        
        self.count += 1
        self.indices_mmap.flush()
        self.signs_mmap.flush()
        self.scales_mmap.flush()
        self.scale_s_mmap.flush()

    def search(self, query: torch.Tensor, top_k: int = 1):
        y_query = self.quantizer.rotate(query)
        
        levels = self.quantizer.codebook if self.use_eden else self.quantizer.levels
        lut = torch.outer(y_query, levels)
        
        # Load pre-unpacked data directly
        indices = torch.from_numpy(self.indices_mmap[:self.count].copy()).long()
        signs = torch.from_numpy(self.signs_mmap[:self.count].copy()).float()
        scales = torch.from_numpy(self.scales_mmap[:self.count].copy()).float()
        scale_s = torch.from_numpy(self.scale_s_mmap[:self.count].copy()).float()
        
        dim_indices = torch.arange(self.dim).unsqueeze(0).expand(self.count, -1)
        scores = torch.sum(lut[dim_indices, indices], dim=1)
        
        if self.use_eden:
            scores = scores * scale_s
            res_sum = torch.matmul(signs.float(), y_query)
            residual_scores = res_sum * scales
        else:
            residual_scores = torch.zeros(self.count, device=scores.device)
        
        total_scores = scores + residual_scores
        
        top_scores, top_indices = torch.topk(total_scores, min(top_k, self.count))
        
        return list(zip(top_indices.tolist(), total_scores.tolist()))
