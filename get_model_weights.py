import os
from pathlib import Path

import gdown

OUR_URL = ""
# MOUTH_MODEL = "https://drive.google.com/uc?id=1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm"
MOUTH_MODEL = "https://drive.google.com/uc?id=179NgMsHo9TeZCLLtNWFVgRehDvzteMZE"

root_dir = Path(__file__).absolute().resolve().parent
model_dir = root_dir / "models"
model_dir.mkdir(exist_ok=True, parents=True)

# output_model = "models/final_model_weights.pth"
mouth_model_path = "src/utils/mouth_model.pth"

# gdown.download(OUR_URL, output_model, quiet=False)
gdown.download(MOUTH_MODEL, mouth_model_path, quiet=False)
