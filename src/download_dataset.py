# download dataset from google drive

import gdown


link = "https://drive.google.com/drive/folders/1_3JckYWL6bLGh8cu_JtG2LzMEAdoCGat"
output = "data.zip"

gdown.download(link, output, quiet=False)

