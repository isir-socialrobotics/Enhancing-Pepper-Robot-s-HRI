import os
import gdown

google_drive_link = 'https://drive.google.com/drive/folders/1_3JckYWL6bLGh8cu_JtG2LzMEAdoCGat?usp=sharing'


download_path = '~/Downloads/dataset/'


os.makedirs(download_path, exist_ok=True)

# Download dataset
gdown.download(google_drive_link, os.path.join(download_path, 'dataset.zip'), quiet=False)
