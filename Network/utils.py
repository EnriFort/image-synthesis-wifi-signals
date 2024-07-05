# Python file containing usefull functions
import torch
from pathlib import Path
from pytorch_msssim import ssim # pip install pytorch-msssim
import cv2 # pip install opencv-python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# function to compute mean and standard deviation of the datset
def get_mean_std(loader):
  # VAR[X] = E[X**2] - E[x]**2
  channels_sum, channels_squared_sum, num_batches = 0, 0, 0

  for data, _ in loader:
    channels_sum += torch.mean(data, dim=[0,2,3])
    channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
    num_batches += 1
  
  mean = channels_sum/num_batches
  std = (channels_squared_sum/num_batches - mean**2)**0.5

  return mean, std 


# If you need to calculate MS-SSIM/SSIM on normalized images
def SSIM(X, Y):
  X = (X + 1)/2 # [-1, 1] => [0, 1]
  Y = (Y + 1)/2
  ssim_val = ssim(X, Y, data_range=1, size_average=True)
  return ssim_val


# De-Normalize function
def deNormalize(img, min, max):
  norm_img = cv2.normalize(img, None, alpha = min, beta = max, 
                           norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
  norm_img = norm_img.astype(np.uint8)
  return norm_img


# to plot and save resulting images
def plot_images(result_path, outputs):
    
    for k in range(len(outputs)):
        #plt.figure(figsize=(9, 2))

        imgs = outputs[k][0].cpu().detach().numpy() 
        recon = outputs[k][1].cpu().detach().numpy()

        # de-normalize
        imgs = deNormalize(imgs, 0, 255) # si può usare img = ((img * std) + mean)
        recon = deNormalize(recon, 0, 255) # recon = ((recon * std) + mean)

        for i, item in enumerate(imgs):
            if i >= 1: break
            #plt.subplot(2, 9, i+1)   
            item = item.transpose(1, 2, 0)
            save_images(item, "img", result_path, k+1)
            #plt.imshow(item)
                
        for i, item in enumerate(recon):
            if i >= 1: break
            #plt.subplot(2, 9, 9+i+1) # row_length + i + 1  
            item = item.transpose(1, 2, 0) 
            save_images(item, "recon", result_path, k+1)
            #plt.imshow(item)


# to plot curves of train and test loss/accuracy
def plot_loss_curves(results: dict[str, list[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}
    """
    # Get the loss values of the results dictionary (training and validation)
    loss = results['train_loss']
    val_loss = results['val_loss']

    # Get the accuracy values of the results dictionary (training and validation)
    accuracy = results['train_acc']
    val_accuracy = results['val_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


# To save original and reconstructed images
def save_images(img: np.ndarray, img_type: str, path: str, index: int):
  
  im = Image.fromarray(img)
  img_name = Path(img_type + str(index) + ".jpeg")
  im.save(path / img_name)