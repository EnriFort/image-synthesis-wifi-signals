'''
Definisce l'encoder e il decoder come sottoclassi dei nn.module
L'encoder consiste in una serie di strati convoluzionali
Il decoder consiste in una serie di strati convoluzionali transposti
'''
# Input image: Batch number, color_channels, Height, Width (B, C, H, W)
# Output_Shape of the image after i'th layer = [(W-K+2P)/S]+1
# W is the input volume
# K is the kernel size
# P is the padding
# S is the stride

import torch
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        #nn.Dropout(0.3),
        nn.Conv2d(16, 32, 3, 2, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        #nn.Dropout(0.3),
        nn.Conv2d(32, 64, 3)
    )

  def forward(self, x):
    return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
      super(Decoder, self).__init__()

      self.decoder = nn.Sequential(
          nn.ConvTranspose2d(64, 32, 3),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          #nn.Dropout(0.3),
          nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
          nn.BatchNorm2d(16),
          nn.ReLU(),
          #nn.Dropout(0.3),
          nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
          nn.Tanh() 
          #nn.Sigmoid()
      )

    def forward(self, x):
      return self.decoder(x)


# rete LSTM con P unità, ovvero, una per pacchetto, e una convoluzione 2D transposta
# è applicata sul risultato dell'ultima unità per abilitare la conversione di dominio radio-visione

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers, kernel_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # ==> h_p
        self.conv = nn.ConvTranspose2d(hidden_size, latent_size, kernel_size, stride=2, padding=0) # ==> Z'. Input atteso: (N, Cin, Hin, Win) 
      
    def forward(self, x):
        out, (_, _) = self.lstm(x) 
        
        # Get the last LSTM unit hidden state vector (h_p)
        out = out[:, -1, :] # si può usare anche: out = hidden[-1]
        #print(out.shape)
        
        # Ridimensiona l'output della LSTM per darlo in input al ConvTranspose2D
        reshaped_out = out.view(out.shape[0], out.shape[1], 1, 1) # aggiunge due dimensioni per lo strato ConvTranspose2D 
        latent = self.conv(reshaped_out) 
        return latent