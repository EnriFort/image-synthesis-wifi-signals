''' Classe custom dataset  personalizzata per creare i dataset delle immagini e dei segnali'''

import pathlib
import os
from torch.utils.data import Dataset
import torch
# per leggere i file .mat e le immagini
from scipy.io import loadmat 
from PIL import Image

class CustomDataset(Dataset): # sottoclasse di torch.utils.data.Dataset
  def __init__(self, img_dir: str, sgn_dir: str, 
               transform_img=None): # transform è opzionale
    assert os.path.exists(img_dir)
    assert os.path.exists(sgn_dir)
    # Prende tutti i path delle immagini/segnali dentro le cartelle di root
    self.img_paths = sorted(list(pathlib.Path(img_dir).glob("*/*.jpg")))
    self.sgn_paths = sorted(list(pathlib.Path(sgn_dir).glob("*/*.mat")))
    # Prepara la transform
    self.transform_img = transform_img

  def load_data(self, index: int):
    "Apre un'immagine e la restituisce"
    image_paths = self.img_paths[index]
    signal_paths = self.sgn_paths[index]
    return Image.open(image_paths), loadmat(signal_paths)
    #return Image.open(image_paths).convert("RGB"), loadmat(signal_paths)

  def __len__(self) -> int:
    "Restituisce il numero totale di sample"
    return len(self.img_paths)

  def __getitem__(self, index: int):
    "Restituisce un sample di dati, data-image e data-signal (img, sgn)"
    img, sgn = self.load_data(index)
    sgn = sgn['filteredMedianTimeAmp'] # prende solo il valore 'filteredMedianTimeAmp', ovvero, un np.ndarray
    sgn = torch.as_tensor(sgn, dtype=torch.float32)

    # Applica la trasformazione alle immagini, se necessario
    if self.transform_img:
      return self.transform_img(img), sgn  
    else:
      return img, sgn