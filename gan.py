import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from mlp import MLP
import numpy as np

import matplotlib.pyplot as plt

batch_size = 128
disc_train_k = 1
gen_train_k = 1
epochs = 100
noise_dim = 100
data_dim = 28*28

def save_samples_as_images( S: '9x784', filename ):
    batch, n = S.shape
    assert(batch == 9)
    assert(n == 784)

    for i in range(3):
        for j in range(3):
            i_no = i * 3 + j
            image = S[i_no,...].reshape((28,28))
            image = (image + 1.0) / 2.0 # normalize to [0,1]
            plt.subplot(3, 3, i_no + 1)
            plt.imshow(image, cmap='gray')

    plt.savefig(filename + '.png')
    plt.close()

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist = torchvision.datasets.MNIST('./mnist', train=True, transform=trans, download=True)
mnistdl = torch.utils.data.DataLoader(mnist, shuffle=True, batch_size=batch_size//2, drop_last=True,
    pin_memory=True)

if __name__ == '__main__':
    gen = MLP(noise_dim, [256, 512, 1024, data_dim],
        activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Tanh()])
    disc = MLP(data_dim, [512,256,1], bn=False,
        activations=[nn.LeakyReLU(0.2), nn.LeakyReLU(0.2), torch.sigmoid])
    
    if torch.cuda.is_available():
        gen = gen.cuda()
        disc = disc.cuda()

    optim_gen = optim.Adam(gen.parameters(), lr=1e-4)
    optim_disc = optim.Adam(disc.parameters(), lr=1e-4)

    BCEcrit = nn.BCELoss()

    GenLosses = []
    DiscLosses = []

    # labels are always same
    disc_labels = torch.cat([torch.ones(batch_size//2, 1),
                   torch.zeros(batch_size//2, 1)], dim=0)
    gen_labels = torch.ones(batch_size//2, 1)
    
    if torch.cuda.is_available():
        disc_labels = disc_labels.cuda()
        gen_labels = gen_labels.cuda()
        
    for e in range(epochs):
        # print a msg
        print('epoch {0} starts'.format(e))
        for i, (data, _) in enumerate(mnistdl):
            data = data.resize_(batch_size//2, data_dim) * 2.0 - 1.0 # -1 TO +1
            noise = torch.from_numpy(
                    np.random.normal(size=(batch_size//2, noise_dim)).astype(np.float32)
                )
            
            if torch.cuda.is_available():
                data = data.cuda()
                noise = noise.cuda()

            for _ in range(disc_train_k):
                optim_disc.zero_grad()

                # two forward passes
                disc_out_data = disc(data)
                disc_out_noise = disc(gen(noise))
                disc_out = torch.cat([disc_out_data, disc_out_noise], dim=0)

                # discriminator loss
                disc_loss = BCEcrit(disc_out, disc_labels)

                # back-prop
                disc_loss.backward()
                # weight update
                optim_disc.step()

            for _ in range(gen_train_k):
                optim_gen.zero_grad()

                # forward pass
                gen_out_noise = disc(gen(noise))

                # generator loss
                gen_loss = BCEcrit(gen_out_noise, gen_labels)

                # back-prop
                gen_loss.backward()
                # weight update
                optim_gen.step()

            if i % 100 == 0:
                if torch.cuda.is_available():
                    gloss = gen_loss.cpu().data.item()
                    dloss = disc_loss.cpu().data.item()
                else:
                    gloss = gen_loss.data.item()
                    dloss = disc_loss.data.item()

                GenLosses.append(gloss)
                DiscLosses.append(dloss)

                # generate samples
                with torch.no_grad():
                    noise_ = torch.from_numpy(np.random.normal(size=(9, noise_dim)).astype(np.float32))
                    if torch.cuda.is_available():
                        noise_ = noise_.cuda()
                    sample_ = gen(noise_)
                    if torch.cuda.is_available():
                        save_samples_as_images(sample_.cpu().data.numpy(), 'samples/sample_'+str(i))
                    else:
                        save_samples_as_images(sample_.data.numpy(), 'samples/sample_'+str(i))

                print('disc loss: {0} | gen loss {1}'.format(DiscLosses[-1], GenLosses[-1]))

                # see the distribution of discriminator's output prob
                if torch.cuda.is_available():
                    d_mean = disc_out.cpu().data.numpy().mean()
                    d_std = disc_out.cpu().data.numpy().std()
                else:
                    d_mean = disc_out.data.numpy().mean()
                    d_std = disc_out.data.numpy().std()
                print('\'disc_out\' distribution: mu={0}, std={1}'.format(d_mean ,d_std))

        # time to save generator model at each epoch
        print('saving model')
        torch.save(gen.state_dict(), './model/genmodel.pt')
    
    # save the losses
    GenLosses = np.array(GenLosses)
    DiscLosses = np.array(DiscLosses)
    np.save('./log/genloss.npy', GenLosses)
    np.save('./log/discloss.npy', DiscLosses)