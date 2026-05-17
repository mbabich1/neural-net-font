import torch
from torch import nn, tensor
from rasterize_font import load_words_as_tensor, flatten_words, unflatten_word
from visualize import plot_word
import random

DEVICE = "cuda"


class Generator(nn.Module):
    def __init__(self, output_dim) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(100, 256),
                nn.Linear(256, 512),
                nn.Linear(512, 1024),
                nn.Linear(1024, output_dim),
            ]
        )
        self.fns = nn.ModuleList(
            [
                nn.LeakyReLU(0.2),
                nn.LeakyReLU(0.2),
                nn.LeakyReLU(0.2),
                nn.Tanh(),  # Swappable with whatever we choose for data normalization
            ]
        )

    def forward(self, X):
        z = X
        for layer, fn in zip(self.layers, self.fns):
            z = fn(layer(z))

        return z


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(4096, 1024),
                nn.Linear(1024, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 1),
            ]
        )
        self.fns = nn.ModuleList(
            [nn.LeakyReLU(0.2), nn.LeakyReLU(0.2), nn.LeakyReLU(0.2), nn.Identity()]
        )
        self.dropout = nn.ModuleList(
            [nn.Dropout(0.3), nn.Dropout(0.3), nn.Dropout(0.3), nn.Identity()]
        )

    def forward(self, X):
        z = X
        for layer, fn, dropout in zip(self.layers, self.fns, self.dropout):
            z = dropout(fn(layer(z)))

        return z
    

def prep_smp(smp_tensor: torch.Tensor) -> torch.Tensor:
    unnormalized = (smp_tensor + 1) / 2 # brings out of tanh to sigmoid range
    bytes = torch.clamp(unnormalized * 255, 0, 255).to(torch.uint8)
    unflattened = unflatten_word(bytes, 256)
    return unflattened


def train(
    generator: Generator,
    discriminator: Discriminator,
    data_loader,
    num_epochs=100,
    batch_size=128,
    latent_size=100,
):
    BCELoss = nn.BCELoss()
    BCELogitsLoss = nn.BCEWithLogitsLoss()
    generator.train()
    discriminator.train()
    gen_optim = torch.optim.Adam(generator.parameters())
    dis_optim = torch.optim.Adam(discriminator.parameters())

    for epoch in range(num_epochs):
        tl_gen_loss = 0
        tl_dis_loss = 0
        fakes = None
        for batch_idx, (batch_images,) in enumerate(data_loader):
            # real images with discriminator
            real_images = batch_images.to(DEVICE)
            outputs = discriminator(real_images)
            dis_loss_real = BCELogitsLoss(outputs, (torch.ones(batch_size, 1) * 0.9).to(DEVICE))

            # fake images with generator
            z = torch.randn(batch_size, latent_size).to(DEVICE)
            fakes = generator(z)
            outputs = discriminator(fakes)
            dis_loss_fake = BCELogitsLoss(outputs, torch.zeros(batch_size, 1).to(DEVICE))

            dis_loss = dis_loss_real + dis_loss_fake
            gen_optim.zero_grad()
            dis_optim.zero_grad()
            dis_loss.backward()
            dis_optim.step()

            # training of generator
            z = torch.randn(batch_size, latent_size).to(DEVICE)
            fakes = generator(z)
            outputs = discriminator(fakes)

            gen_loss = BCELogitsLoss(outputs, torch.ones(batch_size, 1).to(DEVICE))
            gen_optim.zero_grad()
            dis_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            tl_gen_loss += gen_loss.item()
            tl_dis_loss += dis_loss.item()

        print(f"epoch: {epoch}, gen_loss: {tl_gen_loss}, dis_loss: {tl_dis_loss}")

        curr_batch_size = batch_images.size(0)
        random_idx = random.randint(0, curr_batch_size - 1)

        raw_real_smp = batch_images[random_idx].detach().cpu()
        raw_fake_smp = fakes[random_idx].detach().cpu()

        real_processed = prep_smp(raw_real_smp)
        fake_processed = prep_smp(raw_fake_smp)

        # call to visualize
        plot_word(real_processed, img_id=f"out/real_epoch_{epoch}.png") # real
        plot_word(fake_processed, img_id=f"out/fake_epoch_{epoch}.png") # generated


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.min()
    max_val = tensor.max()

    normalized_tensor = 2 * ((tensor - min_val) / (max_val - min_val)) - 1

    return normalized_tensor


def main():
    words_tensor = load_words_as_tensor(size=12)
    print(type(words_tensor), words_tensor.shape)
    words_tensor, len = flatten_words(words_tensor)
    print(type(words_tensor), words_tensor.shape)
    print(words_tensor[0])

    # normalize data to be tanh compatible
    words_tensor = normalize_tensor(words_tensor)

    dataset = torch.utils.data.TensorDataset(
        words_tensor
    )  # TODO: concat labels when doing label-based

    generator = Generator(4096).to(DEVICE)  # a small starting size
    discriminator = Discriminator().to(DEVICE)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, drop_last=True
    )

    train(generator, discriminator, data_loader)

    # can evaluate here


if __name__ == "__main__":
    main()
