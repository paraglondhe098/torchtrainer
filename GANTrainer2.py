import torch
from tqdm import tqdm
import os
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler


class GANTrainer:
    def __init__(self, generator, discriminator,
                 opt_g, opt_d, latent_dims, batch_size, device, save_path,
                 denorm_func
                 ):
        self.Generator = generator.to(device)
        self.Discriminator = discriminator.to(device)
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.latent_dims = latent_dims
        self.device = device
        self.constant = self.generate_latent(64)
        self.batch_size = batch_size
        self.save_path = save_path
        self.denorm = denorm_func
        self.scaler_g = GradScaler()
        self.scaler_d = GradScaler()
        self.current_epoch = 0
        os.makedirs(save_path, exist_ok=True)
        self.save_samples(save_path, -1, self.constant)

    def generate_targets(self, batch_size, label):
        return torch.ones(batch_size, 1, device=self.device) * label

    def generate_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_dims, 1, 1, device=self.device)

    @staticmethod
    @torch.no_grad()
    def get_score(preds):
        return torch.mean(preds).item()

    def train_discriminator(self, real_images):
        self.Discriminator.train()
        self.Generator.eval()
        optimizer = self.opt_d
        scaler = self.scaler_d
        batch_size = real_images.size(0)

        real_images = real_images.to(self.device)  # X
        real_targets = self.generate_targets(batch_size, 1.0)  # 1
        latent_vector = self.generate_latent(batch_size)  # Z
        fake_targets = self.generate_targets(batch_size, 0.0)  # 0

        optimizer.zero_grad()
        with autocast():
            real_preds = self.Discriminator(real_images)  # D(X)
            real_loss = F.binary_cross_entropy_with_logits(real_preds, real_targets)  # -E[log(D(X))]
            fake_images = self.Generator(latent_vector)  # G(Z)
            fake_preds = self.Discriminator(fake_images)  # D(G(Z))
            fake_loss = F.binary_cross_entropy_with_logits(fake_preds, fake_targets)  # -E[log(1-D(G(Z)))]
            total_loss = real_loss + fake_loss  # -E[log(D(X))]-E[log(1-D(G(Z)))]

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        real_score = self.get_score(F.sigmoid(real_preds))  # % real predicted as real
        fake_score = self.get_score(F.sigmoid(fake_preds))  # % fake predicted as fake

        return total_loss.item(), real_score, fake_score

    def train_generator(self):
        self.Generator.train()
        self.Discriminator.train()
        batch_size = self.batch_size
        optimizer = self.opt_g
        scaler = self.scaler_g

        latent_vector = self.generate_latent(batch_size)  # Z
        targets = self.generate_targets(batch_size, 1.0)  # 1

        optimizer.zero_grad()

        with autocast():
            fake_images = self.Generator(latent_vector)  # G(Z)
            preds = self.Discriminator(fake_images)  # D(G(Z))
            loss = F.binary_cross_entropy_with_logits(preds, targets)  # -E[D(G(Z))]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return loss.item()

    @torch.no_grad()
    def save_samples(self, path, index, latent_vector, show=False):
        if self.current_epoch > 5:
            self.Generator.eval()
        fake_images = self.Generator(latent_vector)
        folder_path = path
        filename = "generated-images{0:0=4d}.png".format(index + 1)
        save_image(self.denorm(fake_images, (0.5, 0.5)), os.path.join(folder_path, filename), nrow=8)
        print("Saving", filename)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(self.denorm(fake_images, (0.5, 0.5)).cpu().detach(), nrow=8).permute(1, 2, 0))

    def fit(self, dataloader, epochs):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []
        loss_g = 0
        loss_d = 0
        real_score = 0
        fake_score = 0

        for epoch in range(epochs):
            self.current_epoch = epoch
            for real_images, _ in tqdm(dataloader):
                loss_d, real_score, fake_score = self.train_discriminator(real_images)
                loss_g = self.train_generator()

            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)

            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch + 1, epochs, loss_g, loss_d, real_score, fake_score))

            self.save_samples(self.save_path, epoch, self.constant)

        History = {"losses_g": losses_g,
                   "losses_d": losses_d,
                   "real_scores": real_scores,
                   "fake_scores": fake_scores,
                   "epochs": list(range(epochs))}

        return History
