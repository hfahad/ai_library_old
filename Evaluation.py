from DataGeneration import DataGeneration
from NeuralNet import NeuralNet
from NeuralNet import NeuralNet
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F


def Evaluation(data_dir, model_path):
    device = "cpu"
    Net = NeuralNet(1, 3, 5).to(device)
    Net.load_state_dict(torch.load(os.path.join(
        model_path, "model_shape.pth"), map_location='cpu'))
   # collecting a data
    trainData = DataGeneration(data_dir, train=True)
    # Training and Validation Split
    trainData, valData = train_test_split(trainData,
                                          random_state=10,
                                          test_size=0.25)
    # Data Loader
    val_Loader = DataLoader(valData, shuffle=True, batch_size=1)
    with torch.no_grad():
        Net.eval()
        loss = 0
        correct = 0
        y_pred = []
        y_true = []
        for data, target in tqdm(val_Loader):
            data = data.unsqueeze(1)
            target = target.squeeze(1)
            data, target = data.to(device), target.to(device)

            output = Net(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()

            _, pred = torch.max(output.data, 1)
            correct += (pred == target).sum().item()

            y_pred.extend(pred.cpu().numpy())
            y_true.extend(target.numpy())

        loss /= len(val_Loader.dataset)

        print('Average Val Loss: {:.4f}, Val Accuracy: {}/{} \
            ({:.3f}%)\n'.format(
            loss, correct, len(val_Loader.dataset),
            100. * correct / len(val_Loader.dataset)))
        np.savetxt('./results/y_pred.txt', y_pred, fmt='%d')
        np.savetxt('./results/y_true.txt', y_true, fmt='%d')

        Confusion_matrix(y_true, y_pred)


def Confusion_matrix(y_true, y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred)
    class_names = ('tri', 'cir', 'rec')
    dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)
    sns.heatmap(dataframe, annot=True, cbar=None, cmap='YlGnBu', fmt='d')
    plt.title('Comfusion Matrix'),
    plt.tight_layout()

    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig('/results/cm_matrix.jpg', bbox_inches='tight')


if __name__ == "__main__":

    Evaluation('/data', '/model')
