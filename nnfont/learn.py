import torch
from torch import nn, tensor
from rasterize_font import load_all_fonts, flatten_words, unflatten_word
from cache_file import cache_after_first_run
from visualize import plot_word
import random

DEVICE = "cuda"


class Generator(nn.Module):
    def __init__(self, latent_size=100, num_axes=3) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size + num_axes, 512, kernel_size=(1, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 1x4 -> 2x8
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 2x8 -> 4x16
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4x16 -> 8x32
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 8x32 -> 16x256
            nn.ConvTranspose2d(64, 1, kernel_size=(4, 16), stride=(2, 4), padding=(1, 1), bias=False),
            nn.Tanh()
        )

    def forward(self, z, attributes):
        z = z.view(z.size(0), z.size(1), 1, 1)
        attributes = attributes.view(attributes.size(0), attributes.size(1), 1, 1)

        input = torch.cat([z, attributes], dim=1)
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, num_axes=3) -> None:
        super().__init__()
        self.main = nn.Sequential(
            # 1 x 16 x 256
            nn.Conv2d(1 + num_axes, 64, kernel_size=(4, 6), stride=(2, 4), padding=(1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 8 x 32
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 4 x 16
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 2 x 8
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 1 x 4 -> 1 x 1 x 1
            nn.Conv2d(512, 1, kernel_size=(1, 4), stride=1, padding=0, bias=False)
        )

    def forward(self, img: torch.Tensor, attributes: torch.Tensor):
        if len(img.shape) == 3:
            img = img.unsqueeze(1)

        channels = attributes.view(attributes.size(0), attributes.size(1), 1, 1)
        channels = channels.expand(-1, -1, 16, 128)

        img_with_attributes = torch.cat([img, channels], dim=1)
        return self.main(img_with_attributes).view(-1, 1)
    

def prep_smp(smp_tensor: torch.Tensor) -> torch.Tensor:
    unnormalized = (smp_tensor + 1) / 2 # brings out of tanh to sigmoid range
    bytes = torch.clamp(unnormalized * 255, 0, 255).to(torch.uint8)
    # unflattened = unflatten_word(bytes, 256)
    bytes = bytes.squeeze().cpu()
    return bytes


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
        for batch_idx, (batch_images, batch_fonts) in enumerate(data_loader):
            current_batch_size = batch_images.size(0)
            real_targets = torch.full((current_batch_size, 1), 0.9, device=DEVICE)
            fake_targets = torch.zeros((current_batch_size, 1), device=DEVICE)

            # real images with discriminator
            real_images = batch_images.to(DEVICE)
            outputs = discriminator(real_images)
            dis_loss_real = BCELogitsLoss(outputs, real_targets)

            # fake images with generator
            z = torch.randn(batch_size, latent_size).to(DEVICE)
            fakes = generator(z)
            outputs = discriminator(fakes)
            dis_loss_fake = BCELogitsLoss(outputs, fake_targets)

            dis_loss = dis_loss_real + dis_loss_fake
            gen_optim.zero_grad()
            dis_optim.zero_grad()
            dis_loss.backward()
            dis_optim.step()

            # training of generator
            z = torch.randn(batch_size, latent_size).to(DEVICE)
            fakes = generator(z)
            outputs = discriminator(fakes)

            gen_loss = BCELogitsLoss(outputs, torch.ones_like(real_targets)) # ones_like, because we want a distinct tensor
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
    words_tensor, one_hot_labels = cache_after_first_run(lambda : load_all_fonts(size=12), 'words12')
    if len(words_tensor.shape) == 3:
        words_tensor = words_tensor.unsqueeze(1)
    print(type(words_tensor), words_tensor.shape)
    print(one_hot_labels.shape)


    # words_tensor, len = flatten_words(words_tensor) # only for non-convolutional
    # print(type(words_tensor), words_tensor.shape)
    print(words_tensor[0])

    # normalize data to be tanh compatible
    words_tensor = normalize_tensor(words_tensor)

    dataset = torch.utils.data.TensorDataset(
        words_tensor,
        one_hot_labels
    )

    generator = Generator(100).to(DEVICE)  # a small starting size
    discriminator = Discriminator().to(DEVICE)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
    )

    train(generator, discriminator, data_loader)

    # can evaluate here


if __name__ == "__main__":
    main()
