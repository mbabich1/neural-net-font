import torch
from torch import nn, tensor
from rasterize_font import load_all_fonts, flatten_words, unflatten_word
from cache_file import cache_after_first_run
from visualize import plot_word, plot_words_grid
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
            nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 2x8 -> 4x16
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 4x16 -> 8x32
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 8x32 -> 16x128
            nn.ConvTranspose2d(128, 1, kernel_size=(4, 6), stride=(2, 4), padding=(1, 1), bias=False),
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
            # 1 x 16 x 128
            nn.Conv2d(1 + num_axes, 128, kernel_size=(4, 6), stride=(2, 4), padding=(1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 8 x 32
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 4 x 16
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 2 x 8
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
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

    # stuff for per-epoch evaluation
    style_permutations = [
            [0, 0, 0],  # Sans, Regular, Upright
            [0, 0, 1],  # Sans, Regular, Italic
            [0, 1, 0],  # Sans, Bold,    Upright
            [0, 1, 1],  # Sans, Bold,    Italic
            [1, 0, 0],  # Serif, Regular, Upright
            [1, 0, 1],  # Serif, Regular, Italic
            [1, 1, 0],  # Serif, Bold,    Upright
            [1, 1, 1],  # Serif, Bold,    Italic
        ]

    g = torch.Generator().manual_seed(42)
    base_z = torch.randn(8, latent_size, generator=g)
    eval_z = base_z.repeat(8, 1).to(DEVICE)

    eval_attrs = torch.tensor([style for style in style_permutations for _ in range(8)], dtype=torch.float32).to(DEVICE)

    for epoch in range(num_epochs):
        tl_gen_loss = 0
        tl_dis_loss = 0
        fakes = None
        for batch_idx, (batch_images, batch_fonts) in enumerate(data_loader):
            current_batch_size = batch_images.size(0)

            # real images with discriminator
            real_images = batch_images.to(DEVICE)
            font_labels = batch_fonts.to(DEVICE)

            real_targets = torch.full((current_batch_size, 1), 0.9, device=DEVICE)
            fake_targets = torch.zeros((current_batch_size, 1), device=DEVICE)

            dis_optim.zero_grad()
            outputs = discriminator(real_images, font_labels)
            dis_loss_real = BCELogitsLoss(outputs, real_targets)

            # fake images with generator
            z = torch.randn(batch_size, latent_size, device=DEVICE)
            fakes = generator(z, font_labels)
            outputs = discriminator(fakes.detach(), font_labels)
            dis_loss_fake = BCELogitsLoss(outputs, fake_targets)

            dis_loss = dis_loss_real + dis_loss_fake
            dis_loss.backward()
            dis_optim.step()

            # training of generator
            gen_optim.zero_grad()
            outputs = discriminator(fakes, font_labels)

            gen_loss = BCELogitsLoss(outputs, torch.ones_like(real_targets)) # ones_like, because we want a distinct tensor
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


        # call the generator to plot a word of each type
        generator.eval()

        with torch.no_grad():
            eval_fakes = generator(eval_z, eval_attrs)

        print(eval_fakes.shape)
        unnormalized = (eval_fakes + 1) / 2.0
        bytes = torch.clamp(unnormalized * 255, 0, 255).to(torch.uint8)

        eval_words_np = bytes.squeeze(1).cpu().numpy()

        plot_words_grid(
            words=eval_words_np[0:8],
            img_id=f"out/sans_normal_normal_{epoch}.png"
        )
        plot_words_grid(
            words=eval_words_np[8:16],
            img_id=f"out/sans_normal_italic_{epoch}.png"
        )
        plot_words_grid(
            words=eval_words_np[16:24],
            img_id=f"out/sans_bold_normal_{epoch}.png"
        )
        plot_words_grid(
            words=eval_words_np[24:32],
            img_id=f"out/sans_bold_italic_{epoch}.png"
        )
        plot_words_grid(
            words=eval_words_np[32:40],
            img_id=f"out/serif_normal_normal_{epoch}.png"
        )
        plot_words_grid(
            words=eval_words_np[40:48],
            img_id=f"out/serif_normal_italic_{epoch}.png"
        )
        plot_words_grid(
            words=eval_words_np[48:56],
            img_id=f"out/serif_bold_normal_{epoch}.png"
        )
        plot_words_grid(
            words=eval_words_np[56:64],
            img_id=f"out/serif_bold_italic_{epoch}.png"
        )

        generator.train()

        torch.save({
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
        }, f"model_state_dicts_{epoch}.pt")



def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.min()
    max_val = tensor.max()

    normalized_tensor = 2 * ((tensor - min_val) / (max_val - min_val)) - 1

    return normalized_tensor


def main():
    words_tensor, one_hot_labels = cache_after_first_run(lambda : load_all_fonts(size=12), 'words12-128-1200k')
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
