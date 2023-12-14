import torch

import torch.nn as nn
import torch.nn.functional as F


"""
NOTE: you can add as many functions as you need in this file, and for all the classes you can define extra methods if you need
"""
class VAE(nn.Module):
  
  def __init__(self, latent_dim, class_emb_dim, num_classes=10):
    super().__init__()

    self.encoder = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
    )

    # defining the network to estimate the mean
    self.mu_net = nn.Linear(64*7*7, latent_dim)

    # defining the network to estimate the log-variance
    self.logvar_net = nn.Linear(64*7*7, latent_dim)

    # defining the class embedding module
    self.class_embedding = nn.Embedding(num_classes, class_emb_dim)

    self.latent_dims = latent_dim

    self.device=torch.device('cuda')

    # defining the decoder here
    self.decoder_linear = nn.Linear(self.latent_dims + class_emb_dim, 64*7*7)
    self.decoder_ConvT1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
    self.decoder_ConvT2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

  def encode(self, x):
    x = self.encoder(x)
    x = x.view(x.size(0), -1)
    mu = self.mu_net(x)
    logvar = self.logvar_net(x)
    return mu, logvar

  def decode(self, z, y):
    z = torch.cat((z, y), dim =1)
    z = self.decoder_linear(z)
    z = z.view(z.size(0), 64, 7, 7)
    z = F.relu(self.decoder_ConvT1(z))
    z = F.sigmoid(self.decoder_ConvT2(z))
    z = z.view(z.size(0), 1, 28, 28)
    return z

  def forward(self, x: torch.Tensor, y: torch.Tensor):
    """
    Args:
        x (torch.Tensor): image [B, 1, 28, 28]
        y (torch.Tensor): labels [B]

    Returns:
        reconstructed: image [B, 1, 28, 28]
        mu: [B, latent_dim]
        logvar: [B, latent_dim]
    """

    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    emb_labels = self.class_embedding(y)
    reconstructed = self.decode(z, emb_labels)
    return reconstructed, mu, logvar

  def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
    """
    applies the reparameterization trick
    """

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    new_sample = mu + eps * std  # using the mu and logvar generate a sample
    return new_sample

  def kl_loss(self, mu, logvar):
    """
    calculates the KL divergence between a normal distribution with mean "mu" and
    log-variance "logvar" and the standard normal distribution (mean=0, var=1)
    """

    kl_div = 0.5*torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar)
    return kl_div

  def get_loss(self, x: torch.Tensor, y: torch.Tensor):
    """
    given the image x, and the label y calculates the prior loss and reconstruction loss
    """

    reconstructed, mu, logvar = self.forward(x, y)
    loss = nn.BCELoss()   
    recons_loss = loss(reconstructed, x)    #reconstruction loss
    prior_loss = self.kl_loss(mu, logvar)   # prior matching loss
    return recons_loss, prior_loss

  @torch.no_grad()
  def generate_sample(self, num_images: int, y, device):
    """
    generates num_images samples by passing noise to the model's decoder
    if y is not None (e.g., y = torch.tensor([1, 2, 3]).to(device)) the model
    generates samples according to the specified labels

    Returns:
        samples: [num_images, 1, 28, 28]
    """

    # sample from noise, find the class embedding and use both in the decoder to generate new samples
    z = torch.randn(num_images, self.latent_dims).to(self.device)
    labels = self.class_embedding(y).to(self.device)
    samples = self.decode(z, labels).to(self.device)
    return samples

def load_vae_and_generate():
    device = torch.device('cuda')
    vae = VAE(latent_dim=10, class_emb_dim=5)
    # loading the weights of VAE
    vae.load_state_dict(torch.load('vae.pt'))
    vae = vae.to(device)

    desired_labels = []
    for i in range(10):
        for _ in range(5):
            desired_labels.append(i)

    desired_labels = torch.tensor(desired_labels).to(device)
    generated_samples = vae.generate_sample(50, desired_labels, device)

    return generated_samples

