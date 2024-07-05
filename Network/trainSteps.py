'''
Contiene le funzioni usare nel train loop principale
'''

import torch
from utils import SSIM
import os.path

def train_step(models: dict[str, torch.nn.Module],
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizers: dict[str, torch.optim.Optimizer],
               device: str):
  
  ssim_value, train_loss, train_acc = 0, 0, 0

  # Put models in train mode
  for name, model in models.items():
    models[name] = model.train()
  
  for (img, sgnl) in dataloader:
    img, sgnl = img.to(device), sgnl.to(device)

    # 1. Optimizer zero grad
    optimizers["enc"].zero_grad()
    optimizers["dec"].zero_grad()
    optimizers["lstm"].zero_grad()

    # 2. Forward pass
    z = models["enc"](img) # CNN2D encoder
    img2 = models["dec"](z) # CNN2D decoder
    z_i = models["lstm"](sgnl) # LSTM encoder
    sgnl_img = models["dec"](z_i)

    # 3. Calculate and accumulate loss
    loss1 = loss_fn(z, z_i) # MSEz(Z, Z')
    loss2 = loss_fn(img, img2) # MSEy(img1, img2)
    loss3 = loss_fn(img2, sgnl_img) # MSEs(img2, sgnlimg)
    
    loss_tot = loss1 + loss2 + loss3
    train_loss += loss_tot.item()
  
    # 4. Loss backward
    loss_tot.backward() # calculate the gradients

    # 5. Optimizer step
    optimizers["enc"].step() # update the weights
    optimizers["dec"].step()    
    optimizers["lstm"].step() 

    # Calculate and accumulate "accuracy"
    ssim_value = SSIM(img, sgnl_img)
    train_acc += ssim_value
  
  # Compute # average loss and "accuracy" per epoch
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc


def val_step(models: dict[str, torch.nn.Module],
            models_path: str,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            device: str,
            best_accuracy: float):

  ssim_value, val_loss, val_acc = 0, 0, 0

  # Put models in evaluation mode
  for name, model in models.items():
    models[name] = model.eval()
  
  # Zero Grad
  with torch.no_grad():
    
    for (img, sgnl) in dataloader:
      img, sgnl = img.to(device), sgnl.to(device)

      # 1. Forward pass
      z = models["enc"](img)
      img2 = models["dec"](z) 
      z_i = models["lstm"](sgnl) 
      sgnl_img = models["dec"](z_i)
      
      # 2. Calculate and accumulate loss
      loss1 = loss_fn(z, z_i)
      loss2 = loss_fn(img, img2) 
      loss3 = loss_fn(img2, sgnl_img)
      
      loss_tot = loss1 + loss2 + loss3
      val_loss += loss_tot.item()

      # 3. Calculate and accumulate "accuracy"
      ssim_value = SSIM(img, sgnl_img)
      val_acc += ssim_value

    # 4. Save the model every time SSIM increments
    if ssim_value > best_accuracy:
      best_accuracy = ssim_value
      for (name, model) in models.items():
          model_name = str(name) + ".pth" # create complete path name where to save model
          FILE = os.path.join(models_path, model_name)
          torch.save(models[name].state_dict(), FILE) # save the current models

  # average loss and "accuracy"
  val_loss /= len(dataloader)
  val_acc /= len(dataloader)

  return val_loss, val_acc, best_accuracy


def test_step(models: dict[str, torch.nn.Module],
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str):
  
  ssim_value, test_loss, test_acc = 0, 0, 0
  outputs = []
  
  # Put models in evaluation mode
  for name, model in models.items():
    models[name] = model.eval()

  for (img, sgnl) in dataloader:
    img, sgnl = img.to(device), sgnl.to(device)

    # Forward
    z_i = models["lstm"](sgnl) 
    sgnl_img = models["dec"](z_i)

    # Loss
    loss = loss_fn(sgnl_img, img)
    ssim_value = SSIM(img, sgnl_img)

    test_loss += loss.item() 
    test_acc += ssim_value

    # output images
    outputs.append((img, sgnl_img)) 

  # average loss and "accuracy"
  test_loss /= len(dataloader)
  test_acc /= len(dataloader)
  return test_loss, test_acc, outputs