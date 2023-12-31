## This code was stolen from: https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder


# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.transforms as transforms

# Matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt

# OS
import os
import argparse

# My own implementations
from gaeLDALoss import gae_lda_loss

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model(embDim = 100):
    autoencoder = Autoencoder(embDim=embDim)
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Autoencoder(nn.Module):
    def __init__(self, embDim=100):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.compressor = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=embDim),
            nn.ReLU()
        )
        self.uncompressor = nn.Sequential(
            nn.Linear(in_features=embDim, out_features=768),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(48, 4, 4))
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        compressed = self.compressor(encoded)
        uncompressed = self.uncompressor(compressed)
        decoded = self.decoder(uncompressed)
        return encoded, decoded

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    parser.add_argument("-e", "--epochs", default=1, type=int,
                        help='Number of epochs to train the AE on [default: 1]')
    parser.add_argument("-b", "--batchSize", default=64, type=int,
                        help="Batch size of the data used for training and testing [default: 64]")
    parser.add_argument("-z", "--embDim", default=100, type=int,
                        help="Number of dimensions for the low dimension representation [default: 100]")
    args = parser.parse_args()

    print(args.epochs)


    numEpochs = args.epochs
    batchSize = args.batchSize
    embeddingDim = args.embDim

    # Create model
    autoencoder = create_model(embDim=embeddingDim)

    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.valid:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load("./models/autoencoder.pkl"))
        dataiter = iter(testloader)
        images, labels = dataiter._next_data()
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        imshow(torchvision.utils.make_grid(images))

        images = get_torch_vars(images)

        decoded_imgs = autoencoder(images)[1]
        imshow(torchvision.utils.make_grid(decoded_imgs.data))

        exit(0)

    # Define an optimizer and criterion
    optimizer = optim.Adam(autoencoder.parameters())

    for epoch in range(numEpochs):

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = get_torch_vars(inputs)

            # ============ Forward ============
            encoded, outputs = autoencoder(inputs)

            # ============ LDA loss ============
            ldaLoss = gae_lda_loss(inputs=inputs, outputs=outputs, labels=labels)

            # ============ Backward ============
            optimizer.zero_grad()
            ldaLoss.backward()
            optimizer.step()

            # ============ Logging ============
            running_loss += ldaLoss.data
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        
        print('Finished epoch {}'.format(str(epoch)))

    print('Finished Training')
    print('Saving Model...')
    if not os.path.exists('./models'):
        os.mkdir('./models')
    torch.save(autoencoder.state_dict(), "./models/gae_autoencoder.pkl")


if __name__ == '__main__':
    main()
