"""List files (id, path) in the authors' public CTRL Google Drive folder without downloading."""
import gdown

URL = 'https://drive.google.com/drive/folders/19-pvKCTLgJ_x6j1C3AvKgHM3GYMNxf6I'
for f in gdown.download_folder(URL, skip_download=True, quiet=True):
    print(f.id, f.path)
