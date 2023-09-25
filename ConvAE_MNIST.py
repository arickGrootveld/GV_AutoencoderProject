# Script to use our Convolutional AE on the MNIST dataset
# Also mostly stolen from: https://saturncloud.io/blog/convolutional-autoencoder-in-pytorch-for-dummies/#:~:text=Convolutional%20autoencoders%20are%20widely%20used,image%20compression%2C%20and%20image%20generation.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from ConvAEImp import ConvAutoencoder


transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    for data in train_loader:
        img, _ = data
        img = img.to(device)

        # Forward Pass
        output = model(img)

        # Compute Loss
        loss = criterion(output, img)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * img.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))