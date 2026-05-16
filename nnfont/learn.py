import torch
from torch import nn, tensor

DEVICE = "cuda"

class Generator(nn.Module):
    def __init__(self, output_dim) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(100, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, output_dim)
        )
        self.fns = nn.ModuleList(
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            nn.Tanh() # Swappable with whatever we choose for data normalization
        )

    def forward(self, X):
        y = None
        for layer, fn in zip(self.layers, self.fns):
            y = fn(layer(X))

        return y

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.LazyLinear(1024), # Lazy Linear so we can specify the incoming feature size later
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 2) # one neuron for bold/normal, one neuron for serif/sans serif
        )
        self.fns = nn.ModuleList(
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
        )

    def forward(self, X):
        y = None
        for layer, fn in zip(self.layers, self.fns):
            y = fn(layer(X))

        return y


def train(generator: Generator, discriminator: Discriminator, X, y, num_epochs=100, batch_size=64, latent_size=64):
    BCELoss = nn.BCELoss()
    generator.train()
    discriminator.train()
    gen_optim = torch.optim.Adam(generator.parameters())
    dis_optim = torch.optim.Adam(discriminator.parameters())

    for epoch in range(num_epochs):
        # real images with discriminator
        outputs = discriminator(X)
        dis_loss_real = BCELoss(outputs, y)
        
        # fake images with generator
        z = torch.randn(batch_size, latent_size).to(DEVICE)
        fakes = generator(z)
        outputs = discriminator(fakes)
        dis_loss_fake = BCELoss(outputs, torch.zeros(batch_size, 1).to(DEVICE))

        dis_loss = dis_loss_real + dis_loss_fake
        gen_optim.zero_grad()
        dis_optim.zero_grad()
        dis_loss.backwards()
        dis_optim.step()

        # training of generator
        z = torch.randn(batch_size, latent_size).to(DEVICE)
        fakes = generator(z)
        outputs = discriminator(fakes)

        gen_loss = BCELoss(outputs, torch.ones(batch_size, 1).to(DEVICE))
        gen_optim.zero_grad()
        dis_optim.zero_grad()
        gen_loss.backward()
        gen_optim.step()

        print(f"epoch: {epoch}, gen_loss: {gen_loss}, dis_loss: {dis_loss}")



def main():
    dataset = None
    generator = Generator(2048) # a small starting size
    discriminator = Discriminator()

    train(generator, discriminator, dataset)

    # can evaluate here

if __name__ == "__main__":
    main()
