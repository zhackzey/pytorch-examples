import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = 999

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# parameters
dataroot = "data/celeba"
workers = 0
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1

dataset = dset.ImageFolder(root = dataroot,
                            transform = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                            ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle =True, num_workers = workers)

device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

'''
real_batch = next(iter(dataloader))
plt.figure(figsize = (8,8))
plt.axis('off')
plt.title('Training Images')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],padding = 2,normalize = True).cpu(),(1,2,0)))
plt.pause(10)
'''

# custom weights initialization called on netD and netG
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != - 1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data,0)

# generator
class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8,4,1,0,bias = False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size (ngf * 8) * 4 * 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4,2,1,bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size (ngf * 4) * 8 * 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4,2,1,bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size (ngf * 2) * 16 * 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4,2,1,bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size (ngf ) * 32 * 32
            nn.ConvTranspose2d(ngf, nc, 4,2,1,bias = False),
            nn.Tanh()
            # state size (nc) * 4 * 4
        )
    
    def forward(self,input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super(Discriminator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) * 64 * 64
            nn.Conv2d(nc,ndf,4,2,1,bias = False),
            nn.LeakyReLU(0.2,inplace = True),
            # state size (ndf) * 32 * 32
            nn.Conv2d(ndf, ndf * 2, 4,2,1,bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # state size (ndf *2) * 16 * 16
            nn.Conv2d(ndf * 2, ndf * 4, 4,2,1,bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # state size (ndf * 4) * 8 * 8
            nn.Conv2d(ndf * 4, ndf * 8, 4,2,1,bias = False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            # state size (ndf * 8) * 4 * 4
            nn.Conv2d(ndf * 8, 1 , 4,1,0,bias = False),
            nn.Sigmoid()
        )
    def forward(self,input):
        return self.main(input)

if __name__ == "__main__":
    netG = Generator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG,list(range(ngpu)))

    netG.apply(weights_init)
    print(netG)
    netD = Discriminator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD,list(range(ngpu)))
    netD.apply(weights_init)
    print("hello")
    print(netD)

    # initialize BCE loss function
    # here BCE means: binary cross entropy
    criterion = nn.BCELoss()

    # create a set of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device = device)

    # establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # setup adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))

    # Training Loop
    # Lists to keep track of progress

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop ...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in dataloader
        for i, data in enumerate(dataloader, 0):
            ################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ################################
            ## Train with all-real batch

            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device) 
            b_size = real_cpu.size(0)
            label = torch.full((b_size,),real_label,device = device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output,label)
            # Calculate gradients for D in backward pass
            errD_real.backward(retain_graph = True)
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device = device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on all-fake batch
            errD_fake = criterion(output,label)
            # Calculate gradients for this batch
            errD_fake.backward(retain_graph = True)
            D_G_z1 =  output.mean().item()

            # Add the gradients from all-real and all-fake batches
            errD = errD_fake + errD_real
            # Update D
            optimizerD.step()

            ################################
            # (2) Update G network: maximize log(D(G(z)))
            ################################
            netG.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            ouput = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output,label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # save losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # check how the generator is doing by saving G's output on fixed_noise 
            if (iters % 500 == 0)  or ((epoch == num_epochs - 1 ) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding = 2,normalize = True))
            
            iters += 1

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.pause(30)

    #%%capture
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()