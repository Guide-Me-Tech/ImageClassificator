
import torch
import clip
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
clip.load(os.getenv("MODEL_NAME"), device)