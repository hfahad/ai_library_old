from argparse import ArgumentParser
from DataGeneration import DataGeneration
from NeuralNet import NeuralNet
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('./runs/shape')


def Train(batch_size, nb_epochs, data_dir):

    # collecting a data
    trainData = DataGeneration(data_dir, train=True)
    testData = DataGeneration(data_dir, train=False)
    # Training and Validation Split
    trainData, valData = train_test_split(trainData,
                                          random_state=10,
                                          test_size=0.25)
    # Data Loader
    train_Loader = DataLoader(trainData, shuffle=True,	batch_size=batch_size)
    val_Loader = DataLoader(valData, shuffle=True, batch_size=batch_size)
    test_Loader = DataLoader(testData, batch_size=batch_size)

    # Define a model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = NeuralNet(in_channels=1, out_channels=3, kernel=5)
    Net.train().to(device)
    optimizer = optim.Adam(params=Net.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    for n in range(nb_epochs):
        correct = 0
        running_loss = 0
        print(f'Epoch: {n+1}/{nb_epochs}')

        for (data, target) in tqdm(train_Loader):
            data = data.unsqueeze(1)
            target = target.squeeze(1)
            data, target = data.to(device), target.to(device)
            output = Net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(output.data, 1)
            correct += (pred == target).sum().item()
            running_loss += loss.item()

        print('\nAverage training Loss: {:.4f}, training Accuracy: \
            {}/{} ({:.3f}%)\n'.format(
            loss, correct, len(train_Loader.dataset),
            100. * correct / len(train_Loader.dataset)))
        # Tensorboard defining
        writer.add_scalar('training loss',
                          running_loss / len(train_Loader.dataset),
                          n)

        writer.add_scalar('Accuracy',
                          correct / len(train_Loader.dataset),
                          n)

        with torch.no_grad():
            Net.eval()
            loss = 0
            correct = 0

            for data, target in val_Loader:
                data = data.unsqueeze(1)
                target = target.squeeze(1)
                data, target = data.to(device), target.to(device)

                output = Net(data)
                loss += F.cross_entropy(output, target, reduction='sum').item()

                _, pred = torch.max(output.data, 1)
                correct += (pred == target).sum().item()

            loss /= len(val_Loader.dataset)

            print('Average Val Loss: {:.4f}, Val Accuracy: {}/{} \
                ({:.3f}%)\n'.format(
                loss, correct, len(val_Loader.dataset),
                100. * correct / len(val_Loader.dataset)))

    Path = './model/model_shape.pth'
    torch.save(Net.state_dict(), Path)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        dest="data_dir", default='./data/',
                        help="data folder")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=100,
                        help="number of iterations")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=64,
                        help="batch_size")

    args = parser.parse_args()
    Train(**vars(args))
