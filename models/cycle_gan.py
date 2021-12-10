import itertools

import torch
from torch.optim import Adam
from torch.nn import L1Loss, BCELoss
from models.utils import weights_init, ImagePool
from networks import Generator, Discriminator


class CycleGAN:
    def __init__(self, dataset_a, dataset_b, learning_rate, pool_size, device='cpu'):
        self.device = device
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b

        self.bce_loss = BCELoss()
        self.l1_loss = L1Loss()

        # init 4 networks
        self.g_a = Generator()
        self.g_b = Generator()
        self.d_a = Discriminator()
        self.d_b = Discriminator()

        self.fake_a_pool = ImagePool(pool_size)  # create image buffer to store previously generated images
        self.fake_b_pool = ImagePool(pool_size)  # create image buffer to store previously generated images

        self.g_a.apply(weights_init)
        self.g_b.apply(weights_init)
        self.d_a.apply(weights_init)
        self.d_b.apply(weights_init)

        # Init optimizers
        self.optimizer_g = Adam(itertools.chain(self.g_a.parameters(), self.g_b.parameters()),
                                lr=learning_rate, betas=(0.5, 0.999))
        self.optimizer_d_a = Adam(self.d_a.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizer_d_b = Adam(self.d_b.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    def train(self, num_epochs):

        for _ in range(num_epochs):
            for i, data_a in enumerate(self.dataset_a):
                data_b = self.dataset_b[i]
                self.step(data_a, data_b)

    def forward(self, real_a, real_b):
        fake_b = self.g_a(real_a)
        rec_a = self.g_b(fake_b)
        fake_a = self.g_b(real_b)
        rec_b = self.g_a(fake_a)

        return fake_a, fake_b, rec_a, rec_b

    def backward_g(self, real_a, real_b, fake_a, fake_b, rec_a, rec_b):
        self.optimizer_g.zero_grad()

        loss_identity_a, loss_identity_b = self.compute_identity_loss(real_a, real_b)
        loss_gan_ab, loss_gan_ba = self.compute_gan_loss(fake_a, fake_b)
        loss_cycle_bab, loss_cycle_aba = self.compute_cycle_loss(real_a, real_b, rec_a, rec_b)
        loss_total = loss_identity_a + loss_identity_b + loss_gan_ab + loss_gan_ba + loss_cycle_bab + loss_cycle_aba

        loss_total.backward()
        self.optimizer_g.step()

    def backward_d_a(self, real_a, fake_a):
        err = self.backward_d_common(real_a, fake_a, self.fake_a_pool)
        err.backward()
        self.optimizer_d_a.step()

    def backward_d_b(self, real_b, fake_b):
        err = self.backward_d_common(real_b, fake_b, self.fake_a_pool)
        err.backward()
        self.optimizer_d_b.step()

    def backward_d_common(self, real, fake, pool):
        batch_size = real.size(0)
        ones = torch.ones((batch_size, 1), device=self.device)
        zeros = torch.ones((batch_size, 1), device=self.device)
        self.d_a.zero_grad()
        real_output_a = self.d_a(real)
        err_real = self.bce_loss(real_output_a, ones)

        fake = pool.query(fake)
        fake_output = self.d_a(fake.detach())
        err_fake = self.bce_loss(fake_output, zeros)

        return (err_real + err_fake) / 2

    def compute_identity_loss(self, real_a, real_b):
        id_a = self.g_b(real_a)
        id_b = self.g_a(real_b)

        return self.l1_loss(id_a, real_a), self.l1_loss(id_b, real_b)

    def compute_gan_loss(self, fake_a, fake_b):
        batch_size = fake_a.size(0)
        ones = torch.ones((batch_size, 1), device=self.device)

        loss_gan_ba = self.bce_loss(fake_a, ones)
        loss_gan_ab = self.bce_loss(fake_b, ones)
        return loss_gan_ba, loss_gan_ab

    def compute_cycle_loss(self, real_a, real_b, rec_a, rec_b):
        loss_cycle_aba = self.l1_loss(rec_a, real_a)
        loss_cycle_bab = self.l1_loss(rec_b, real_b)

        return loss_cycle_aba, loss_cycle_bab

    def step(self, data_a, data_b):
        real_a = data_a.to(self.device)
        real_b = data_b.to(self.device)
        fake_a, fake_b, rec_a, rec_b = self.forward(real_a, real_b)

        self.backward_g(real_a, real_b, fake_a, fake_b, rec_a, rec_b)
        self.backward_d_a(real_a, fake_a)
        self.backward_d_b(real_b, fake_b)
