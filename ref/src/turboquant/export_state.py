import json
import numpy as np
import torch
from src.turboquant.quantizers import DriveV3Quantizer

def export_drive_v3_state(dim=1024, bits=3, filename="drive_v3_state.json"):
    print(f"Exporting DriveV3 state (dim={dim}, bits={bits})...")
    
    quantizer = DriveV3Quantizer(dim=dim, bits=bits, seed=42)
    
    rotation = quantizer._internal.rotation.numpy().flatten().tolist()
    
    codebook = quantizer._internal.codebook.numpy().tolist()
    
    state = {
        "dim": dim,
        "bits": bits,
        "rotation": rotation,
        "codebook": codebook
    }
    
    with open(filename, "w") as f:
        json.dump(state, f)
    
    print(f"Successfully exported state to {filename}")

if __name__ == "__main__":
    export_drive_v3_state()
