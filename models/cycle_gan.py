import itertools

import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import L1Loss, MSELoss
from models.utils import weights_init, ImagePool
from .networks import Generator, Discriminator
import torchvision.utils as vutils


class CycleGAN:
    def __init__(self, dataloader, learning_rate, pool_size, output_folder, device='cpu'):
    #def __init__(self, dataset_a, dataset_b, learning_rate, pool_size, output_folder, device='cpu'):
        self.device = device
        self.output_folder = output_folder
        self.dataloader = dataloader
        #self.dataset_a = dataset_a
        #self.dataset_b = dataset_b

        self.adversarial_loss = MSELoss()
        self.l1_loss = L1Loss()

        # init 4 networks
        self.g_a = Generator().to(device) # netG_A2B
        self.g_b = Generator().to(device) # netG_B2A
        self.d_a = Discriminator().to(device) # netD_A
        self.d_b = Discriminator().to(device) # netD_B

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

        for e in range(num_epochs):
            print('=====Epoch %d=====' % e)
            #progress_bar = tqdm(enumerate(self.dataset_a), total=len(self.dataset_a))
            progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
            #for i, data_a in progress_bar:
            for i, data in progress_bar:
                data_a = data['A']
                data_b = data['B']
                #self.dataset_b[i]
                self.combined_step(data_a, data_b, i, e)

            if e % 10 == 0:
                # do check pointing
                torch.save(self.g_a.state_dict(), f"weights/netG_A_epoch_{e}.pth")
                torch.save(self.g_b.state_dict(), f"weights/netG_B_epoch_{e}.pth")
                torch.save(self.d_a.state_dict(), f"weights/netD_A_epoch_{e}.pth")
                torch.save(self.d_b.state_dict(), f"weights/netD_B_epoch_{e}.pth")

    def forward(self, real_a, real_b):
        # print(real_a.shape)
        fake_b = self.g_a(real_a)
        rec_a = self.g_b(fake_b)
        fake_a = self.g_b(real_b)
        rec_b = self.g_a(fake_a)

        return fake_a, fake_b, rec_a, rec_b

    def backward_g(self, real_a, real_b, fake_a, fake_b, rec_a, rec_b):
        self.optimizer_g.zero_grad()

        loss_identity_a, loss_identity_b = self.compute_identity_loss(real_a, real_b)
        loss_gan_ba, loss_gan_ab = self.compute_gan_loss(fake_a, fake_b)
        loss_cycle_aba, loss_cycle_bab = self.compute_cycle_loss(real_a, real_b, rec_a, rec_b)
        loss_total = loss_identity_a + loss_identity_b + loss_gan_ab + loss_gan_ba + loss_cycle_bab + loss_cycle_aba
        print(loss_total)
        loss_total.backward()
        self.optimizer_g.step()

    def backward_d_a(self, real_a, fake_a):
        err = self.backward_d_common(self.d_a, real_a, fake_a, self.fake_a_pool)
        err.backward()
        self.optimizer_d_a.step()

    def backward_d_b(self, real_b, fake_b):
        err = self.backward_d_common(self.d_b, real_b, fake_b, self.fake_a_pool)
        err.backward()
        self.optimizer_d_b.step()

    def backward_d_common(self, netD, real, fake, pool):
        batch_size = real.size(0)
        ones = torch.ones((batch_size, 1), device=self.device)
        zeros = torch.ones((batch_size, 1), device=self.device)
        netD.zero_grad()
        real_output_a = netD(real)
        err_real = self.adversarial_loss(real_output_a, ones)

        fake = pool.query(fake)
        fake_output = netD(fake.detach())
        err_fake = self.adversarial_loss(fake_output, zeros)

        return (err_real + err_fake) / 2

    def compute_identity_loss(self, real_a, real_b):
        id_a = self.g_b(real_a)
        id_b = self.g_a(real_b)

        return self.l1_loss(id_a, real_a), self.l1_loss(id_b, real_b)

    def compute_gan_loss(self, fake_a, fake_b):
        batch_size = fake_a.size(0)
        ones = torch.ones((batch_size, 1), device=self.device)

        fake_output_a = self.d_a(fake_a)
        fake_output_b = self.d_b(fake_b)

        loss_gan_ba = self.adversarial_loss(fake_output_a, ones)
        loss_gan_ab = self.adversarial_loss(fake_output_b, ones)
        return loss_gan_ba, loss_gan_ab

    def compute_cycle_loss(self, real_a, real_b, rec_a, rec_b):
        loss_cycle_aba = self.l1_loss(rec_a, real_a)
        loss_cycle_bab = self.l1_loss(rec_b, real_b)

        return loss_cycle_aba, loss_cycle_bab

    def step(self, data_a, data_b, i, epoch):
        real_a = data_a.to(self.device)
        real_b = data_b.to(self.device)
        fake_a, fake_b, rec_a, rec_b = self.forward(real_a, real_b)

        self.backward_g(real_a, real_b, fake_a, fake_b, rec_a, rec_b)
        self.backward_d_a(real_a, fake_a)
        self.backward_d_b(real_b, fake_b)

        if i % 100 == 0:
            vutils.save_image(real_a,
                              f"{self.output_folder}/B/real_samples_epoch_{epoch}_{i}.png",
                              normalize=True)
            vutils.save_image(real_b, f"{self.output_folder}/A/real_samples_epoch_{epoch}_{i}.png", normalize=True)

            fake_image_A = 0.5 * (self.g_b(real_b).data + 1.0)
            fake_image_B = 0.5 * (self.g_a(real_a).data + 1.0)

            vutils.save_image(fake_image_A.detach(),
                              f"{self.output_folder}/A/fake_samples_epoch_{epoch}_{i}.png", normalize=True)
            vutils.save_image(fake_image_B.detach(),
                              f"{self.output_folder}/B/fake_samples_epoch_{epoch}_{i}.png", normalize=True)
        
    def combined_step(self, real_A, real_B, i, epoch):
        ones = torch.ones((real_A.size(0), 1), device=self.device) # real labels
        zeros = torch.ones((real_A.size(0), 1), device=self.device)# fake labels

        self.optimizer_g.zero_grad()

        # Identity loss
        # G_B2A(real_A) should look like real_A (use g_b)
        # G_A2B(real_B) should look like real_B (use g_a)
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        identity_A = self.g_b(real_A)
        identity_B = self.g_a(real_B)

        id_loss_A = self.l1_loss(identity_A,real_A) * 5.0
        id_loss_B = self.l1_loss(identity_B,real_B) * 5.0

        # GAN loss
        fake_A = self.g_b(real_B)
        fake_A_label = self.d_a(fake_A)
        GAN_loss_B2A = self.adversarial_loss(fake_A_label, ones)

        fake_B = self.g_a(real_A)
        fake_B_label = self.d_b(fake_B)
        GAN_loss_A2B = self.adversarial_loss(fake_B_label, ones)

        # Cycle loss
        recovered_A = self.g_b(fake_B)
        cycle_loss_ABA = self.l1_loss(recovered_A, real_A)

        recovered_B = self.g_a(fake_A)
        cycle_loss_BAB = self.l1_loss(recovered_B, real_B)

        # Final Loss
        total_loss = id_loss_A + id_loss_B + GAN_loss_A2B + GAN_loss_B2A + cycle_loss_ABA + cycle_loss_BAB

        total_loss.backward()
        self.optimizer_g.step()

        self.optimizer_d_a.zero_grad()
        self.optimizer_d_b.zero_grad()
        # Update d_a
        real_A_label = self.d_a(real_A)
        err_da_real_a = self.adversarial_loss(real_A_label,ones)

        fake_A = self.fake_a_pool.query(fake_A)
        fake_A_label = self.d_a(fake_A)
        err_da_fake_a = self.adversarial_loss(fake_A_label,zeros)

        loss_da = (err_da_real_a + err_da_fake_a) / 2
        loss_da.backward()
        self.optimizer_d_a.step()

        # update d_b
        real_B_label = self.d_b(real_B)
        err_db_real_b = self.adversarial_loss(real_B_label, ones)

        fake_B = self.fake_b_pool.query(fake_B)
        fake_B_label = self.d_b(fake_B)
        err_db_fake_b = self.adversarial_loss(fake_B_label, zeros)

        loss_db = (err_db_real_b + err_db_fake_b) / 2
        loss_db.backward()
        self.optimizer_d_b.step()

        if i % 100 == 0:
            vutils.save_image(real_A,
                              f"{self.output_folder}/B/real_samples_epoch_{epoch}_{i}.png",
                              normalize=True)
            vutils.save_image(real_B, f"{self.output_folder}/A/real_samples_epoch_{epoch}_{i}.png", normalize=True)

            fake_image_A = 0.5 * (self.g_b(real_B).data + 1.0)
            fake_image_B = 0.5 * (self.g_a(real_A).data + 1.0)

            vutils.save_image(fake_image_A.detach(),
                              f"{self.output_folder}/A/fake_samples_epoch_{epoch}_{i}.png", normalize=True)
            vutils.save_image(fake_image_B.detach(),
                              f"{self.output_folder}/B/fake_samples_epoch_{epoch}_{i}.png", normalize=True)