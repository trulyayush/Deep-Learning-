"""Optional external FastAPI GPU inference worker for ECHO-MIND."""
import os
import platform
import torch
from fastapi import FastAPI

app = FastAPI(title="ECHO-MIND External PyTorch Worker", version="1.0.0")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "os": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.get("/api/models")
def list_adapters():
    return [
        {
            "id": "xceptionnet-deepfake",
            "status": "adapter_ready" if os.path.exists("checkpoints/xception.pt") else "checkpoint_missing",
        },
        {
            "id": "wavlm-audio-forensics",
            "status": "adapter_ready" if os.path.exists("checkpoints/wavlm.pt") else "checkpoint_missing",
        },
    ]
