
import torch
import clip
import os
import open_clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip.load(os.getenv("MODEL_NAME"), device)
open_clip.get_tokenizer(os.getenv("MODEL_NAME_OPEN_CLIP") or "ViT-B-32-quickgelu")