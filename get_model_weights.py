import os
from pathlib import Path

import gdown

TASNET_URL = "https://drive.google.com/uc?id=1iS8Xshry2IMrPOuO5iQTE_ju8-CbRhTS"
MOUTH_MODEL = "https://drive.google.com/uc?id=179NgMsHo9TeZCLLtNWFVgRehDvzteMZE"

mouth_model_path = "src/utils/mouth_model.pth"
conv_tasnet_model_path = "models/conv_tasnet.pth"

gdown.download(TASNET_URL, conv_tasnet_model_path, quiet=False)
gdown.download(MOUTH_MODEL, mouth_model_path, quiet=False)
