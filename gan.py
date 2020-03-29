import torch, os
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from mlp import MLP
import numpy as np

import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=False, default=128)
parser.add_argument('--kD', type=int, required=False, default=3)
parser.add_argument('--kG', type=int, required=False, default=1)
parser.add_argument('-e', '--epochs', type=int, required=False, default=100)
parser.add_argument('--base', type=str, required=False, default='.')
args = parser.parse_args()

batch_size = args.batch_size
disc_train_k = args.kD
gen_train_k = args.kG
epochs = args.epochs

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
data_root = os.path.join(args.base, 'mnist')
mnist = torchvision.datasets.MNIST(data_root, train=True, transform=trans, download=True)
mnistdl = torch.utils.data.DataLoader(mnist, shuffle=True, batch_size=batch_size//2, drop_last=True,
    pin_memory=True)

if __name__ == '__main__':
    gen = MLP(noise_dim, [256, 512, 1024, data_dim],
        activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Tanh()])
    disc = MLP(data_dim, [512,256,1], bn=False,
        activations=[nn.LeakyReLU(0.2), nn.LeakyReLU(0.2), lambda x: x])
    
    if torch.cuda.is_available():
        gen = gen.cuda()
        disc = disc.cuda()

    optim_gen = optim.Adam(gen.parameters(), lr=1e-4)
    optim_disc = optim.Adam(disc.parameters(), lr=1e-4)

    BCEcrit = nn.BCELoss()

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

                # # discriminator loss
                # disc_out = torch.cat([disc_out_data, disc_out_noise], dim=0)
                # disc_loss = BCEcrit(disc_out, disc_labels)
                disc_loss = torch.mean(disc_out_noise) - torch.mean(disc_out_data)

                # back-prop
                disc_loss.backward()
                # weight update
                optim_disc.step()

                for p in disc.parameters():
                    p.data.clamp_(-0.01, 0.01) # gradient clip

            for _ in range(gen_train_k):
                optim_gen.zero_grad()

                # forward pass
                gen_out_noise = disc(gen(noise))

                # generator loss
                # gen_loss = BCEcrit(gen_out_noise, gen_labels)
                gen_loss = - torch.mean(gen_out_noise)

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

                # generate samples
                with torch.no_grad():
                    noise_ = torch.from_numpy(np.random.normal(size=(9, noise_dim)).astype(np.float32))
                    if torch.cuda.is_available():
                        noise_ = noise_.cuda()
                    sample_ = gen(noise_)
                    sample_dir = os.path.join(args.base, 'samples')
                    if torch.cuda.is_available():
                        save_samples_as_images(sample_.cpu().data.numpy(), sample_dir + '/sample_'+str(i))
                    else:
                        save_samples_as_images(sample_.data.numpy(), sample_dir + '/sample_'+str(i))

                print('Disc loss: {0:.5f} | Gen loss {1:.5f}'.format(dloss, gloss))

        # time to save generator model at each epoch
        print('saving model')
        torch.save(gen.state_dict(), args.base + '/genmodel.pt')