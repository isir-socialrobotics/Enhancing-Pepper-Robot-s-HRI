import os
import gdown

# Inserisci qui il link di Google Drive
google_drive_link = 'https://drive.google.com/drive/folders/1_3JckYWL6bLGh8cu_JtG2LzMEAdoCGat?usp=sharing'

# Specifica il percorso della cartella dove vuoi salvare il dataset
download_path = '/dataset/'

# Crea la cartella se non esiste già
os.makedirs(download_path, exist_ok=True)

# Scarica il dataset
gdown.download(google_drive_link, os.path.join(download_path, 'nome_del_file.ext'), quiet=False)
