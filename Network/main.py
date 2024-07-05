'''
READ THIS
 -) Per eseguire il file con i parametri a scelta si può lanciare il seguente comando: xEs.  python .\Network\main.py  --img_size=128 --lr=0.0002 
    --wd=...altri parametri a scelta, altrimenti saranno utilizzati i parametri di default.
 
 -) Per il dataset: data/img contiene le immagini, data/sgnl contiene i segnali; N_pkts contiene i segnali con N pacchetti; obj_amp_X contiene 
    le ampiezze polarizzate con X = H, V o HV(entrambe). Per S e R vanno combinati H e V.
'''

# import statements
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm.auto import tqdm #pip install tqdm
from timeit import default_timer as timer 
import os.path

from utils import get_mean_std, plot_loss_curves, plot_images
from customDataset import CustomDataset
from customDatasetPol import CustomDatasetPol
from networks import Encoder, Decoder, LSTM
from trainSteps import train_step, val_step, test_step
import argparse

# CAMBIA LE COSTANTI PER I VARI TIPI DI TEST
COMB_POL = False # per combinare le polarizzazioni passare usare COMB_POL = True
POL = "V" # tipo di polarizzazione da usare: H, V, S o R
PKTS = "50" # numero di pacchetti da usare: 50, 100, 200 o 400

# Inizia con il main del codice
if __name__ == '__main__':
        
    # Per leggere i parametri passati dalla linea di comando si usa il 'parser'
    parser = argparse.ArgumentParser(
        description="Train a network for image synthesis") 
    
    # Hyperparameters (da settare dalla linea di comando)
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0002,
        help='Spcifies learing rate for optimizer. (default: 1e-3)'
    )
    parser.add_argument(
        '--wd',
        type=float,
        default=0,
        help='Spcifies weight decay for optimizer. (default: 1e-5)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs. (default: 5)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for data loaders. (default: 16)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of workers for data loader. (default: 1)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=128,
        help='Image weight and heigth for resizing it. (default: 128)'
    )
    
    # Richiamando 'opt' nel codice si possono utilizzare gli argomenti passati dalla linea di comando
    opt = parser.parse_args() 

    # Imposta il device (cpu o gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Il device disponibile è la " + device)

    # Prepara i vari percorsi
    data_path = os.path.join('data', 'OBJECTS') # root directory
    models_path = os.path.join(data_path, 'models') # percorso dove salvare il miglior modello durante la validazione
    results_path = os.path.join(data_path, 'results', POL,  PKTS, '200epoch') # percorso dove salvare le immagini sintetizzate
    
    # train e test dataset contenenti le immagini
    train_img_dir, test_img_dir = os.path.join(data_path, 'img', 'obj_img', 'train'), os.path.join(data_path, 'img', 'obj_img', 'test')
    
    # train e test dataset contenenti i segnali
    if (COMB_POL == True): # per combinare le polarizzazioni mi servono entrambi i tipi di polarizzazione
        train_sgn_dir_H, test_sgn_dir_H  = os.path.join(data_path, 'sgnl', PKTS+'_pkts', 'obj_amp_H', 'train'), os.path.join(data_path, 'sgnl', PKTS + '_pkts', 'obj_amp_H', 'test')
        train_sgn_dir_V, test_sgn_dir_V  = os.path.join(data_path, 'sgnl', PKTS+'_pkts', 'obj_amp_V', 'train'), os.path.join(data_path, 'sgnl', PKTS + '_pkts', 'obj_amp_V', 'test')
    else: # altrimenti uso l'insieme POL scelto
        train_sgn_dir, test_sgn_dir = os.path.join(data_path, 'sgnl', PKTS+'_pkts', 'obj_amp_'+POL, 'train'), os.path.join(data_path, 'sgnl', PKTS+'_pkts', 'obj_amp_'+POL, 'test') 

    # Si usa la media e la deviazione standard di 0.5 per normalizzare le immagini
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # trasform per le immagini
    img_transform = transforms.Compose([
        transforms.Resize(size=(opt.img_size, opt.img_size)),  # ridimensiono le immagini a HxW
        transforms.ToTensor(), # converto a tensore
        transforms.Normalize(mean, std) # normalizzo
    ])


    if (COMB_POL == True):
    # Carico i dataset
        train_dataset = CustomDatasetPol(img_dir=train_img_dir, sgn_dir_H=train_sgn_dir_H, sgn_dir_V=train_sgn_dir_V,
                                        transform_img=img_transform)
        test_dataset = CustomDatasetPol(img_dir=test_img_dir, sgn_dir_H=test_sgn_dir_V, sgn_dir_V=test_sgn_dir_H,
                                transform_img=img_transform)
    else: 
        train_dataset = CustomDataset(img_dir=train_img_dir, sgn_dir=train_sgn_dir,
                                        transform_img=img_transform)
        test_dataset = CustomDataset(img_dir=test_img_dir, sgn_dir=test_sgn_dir,
                                        transform_img=img_transform)

    # Trasformo i dataset in dataloader per renderli iterabili
    train_dataloader = DataLoader(dataset=train_dataset, 
                                batch_size=opt.batch_size, # quanti sample per ogni batch?
                                num_workers=opt.num_workers, # quanti sottoprocessi usare per il caricamento dei dati? (higher = more)
                                shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, 
                                batch_size=opt.batch_size, 
                                num_workers=opt.num_workers, 
                                shuffle=False)

    # Instanzio la rete e la carico sul device
    enc = Encoder().to(device)
    dec = Decoder().to(device)
    ''' input size = numero di feature = numero di pacchetti 
    hidden_size = h_p size = dimensione vettore di feature radio
    latent_size = Z' size = vettore transposto per il decoder di immagini
    cambia kernel size se modifichi la risoluzione delle immagini: 256pixel --> 62 kernel size, 128 --> 30, 64 --> 14
    '''
    lstm = LSTM(input_size=50, hidden_size=100, latent_size=64, num_layers=1, kernel_size=30).to(device)


    # Creo la funzione di loss
    criterion = nn.MSELoss()           

    # Creo gli optimizers
    enc_optim = torch.optim.Adam(enc.parameters(),
                                lr=opt.lr, weight_decay=opt.wd,
                                betas=(0.5, 0.999))
    dec_optim = torch.optim.Adam(dec.parameters(),
                                lr=opt.lr, weight_decay=opt.wd,
                                betas=(0.5, 0.999)) 
    lstm_optim = torch.optim.Adam(lstm.parameters(),
                                lr=opt.lr, weight_decay=opt.wd,
                                betas=(0.5, 0.999))

    # Creo i dizionari da dare in input alla funzione di train:
    # serve un dizionario perche ho 3 modelli, non solo uno.
    models = { 
        "enc": enc,
        "dec": dec,
        "lstm": lstm
    }
    optimizers = {
        "enc": enc_optim,
        "dec": dec_optim,
        "lstm": lstm_optim
    }

    # Creo un dizionario vuoto per i risultati (per plottarli)
    results = {"train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    max_ssim = 0 # valore SSIM massimo corrente

    start_time = timer()
    for epoch in tqdm(range(opt.epochs)):
        
        # train loop
        train_loss, train_acc = train_step(models, train_dataloader, criterion, optimizers, device)
        # validation loop
        val_loss, val_acc, max = val_step(models, models_path, test_dataloader, criterion, device, max_ssim)
        # aggiorno l'SSIM massimo
        max_ssim = max

        print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f} "
                )
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc.cpu().detach()) # detach perchè è un tensore
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc.cpu().detach())

    end_time = timer()
    print(f"Tempo totale di addestramento: {end_time-start_time:.3f} secondi")

    #plot_loss_curves(results)

    # carico e TESTO i modelli salvati
    print("Testing: ")
    for name, model in models.items():
        model_name = name + ".pth"
        FILE = os.path.join(models_path, model_name)
        models[name].load_state_dict(torch.load(FILE))
    
    # test loop
    test_loss, test_acc, outputs  = test_step(models, test_dataloader, criterion, device)
    print(  f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}")

    # plotto e salvo le immagini sintetizzate
    print("Plot and save output images...")
    plot_images(results_path, outputs)