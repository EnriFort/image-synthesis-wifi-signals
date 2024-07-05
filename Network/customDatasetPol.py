''' Questa è una versione modificata di "CustomDataset", per 
    considerare due tipi di polarizzazioni '''

import pathlib
import os
from torch.utils.data import Dataset
import torch
import numpy as np
# per leggere i file .mat e le immagini
from scipy.io import loadmat 
from PIL import Image


class CustomDatasetPol(Dataset):
  def __init__(self, img_dir: str, sgn_dir_H: str, sgn_dir_V,
               transform_img=None):
    
    assert os.path.exists(img_dir)
    assert os.path.exists(sgn_dir_H)
    assert os.path.exists(sgn_dir_V)

    self.img_paths = sorted(list(pathlib.Path(img_dir).glob("*/*.jpg")))
    self.sgn_paths_H = sorted(list(pathlib.Path(sgn_dir_H).glob("*/*.mat")))
    self.sgn_paths_V = sorted(list(pathlib.Path(sgn_dir_V).glob("*/*.mat")))

    self.transform_img = transform_img

  def load_data(self, index: int):
    "Prende due tipi di segnali, quelli polarizzati H e quelli V"
    image_paths = self.img_paths[index]
    signal_paths_H = self.sgn_paths_H[index]
    signal_paths_V = self.sgn_paths_V[index]
    return Image.open(image_paths), loadmat(signal_paths_H), loadmat(signal_paths_V) 
    
  def __len__(self) -> int:
    return len(self.img_paths)

  def __getitem__(self, index: int):
    "Restituisce un sample di dati, data-image e data-signal (img, sgn)"
    img, sgn_H, sgn_V = self.load_data(index)
    sgn_H, sgn_V = sgn_H['filteredMedianTimeAmp'], sgn_V['filteredMedianTimeAmp'] 
    
    sgn = np.add(sgn_H, sgn_V) # calcola la somma tra le polarizzazioni  
    #sgn = np.divide(sgn_H, sgn_V) # calcola il rapporto tra le polarizzazioni  
    sgn = torch.as_tensor(sgn, dtype=torch.float32) 
    
    # Trasforma se necessario
    if self.transform_img:
      return self.transform_img(img), sgn  
    else:
      return img, sgn