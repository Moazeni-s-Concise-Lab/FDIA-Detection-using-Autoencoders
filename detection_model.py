import torch
import torch.nn as nn

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: Compressing the input measurements
        # to obtain latent representation

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(46,40),
            torch.nn.Tanh(),
            torch.nn.Linear(40,35),
            torch.nn.Tanh(),
            torch.nn.Linear(35,25),
            torch.nn.Tanh(),
            torch.nn.Linear(25,20),
            torch.nn.Tanh(),
            torch.nn.Linear(20,18)
        )
        
        # Decoder: Reconstructing the input measurements
        # from latent space to input space

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(18,20),
            torch.nn.Tanh(),
            torch.nn.Linear(20,25),
            torch.nn.Tanh(),
            torch.nn.Linear(25,35),
            torch.nn.Tanh(),
            torch.nn.Linear(35,40),
            torch.nn.Tanh(),
            torch.nn.Linear(40,46)  
            # torch.nn.Tanh()
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded