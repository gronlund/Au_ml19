import torch
import torch.utils.data as torch_data
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import bidict
import json
from sklearn.preprocessing import normalize

def fit_model(net, train_loader, epochs=10):
    """ fit the neural net using BCEWithLogitsLoss i.e. logistic loss of sigmoud(x), y """
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    optimizer = optim.Adam(net.parameters())
    history = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_total = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs.shape, outputs.dtype, labels.shape, labels.dtype)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * len(labels)
            running_total += len(labels)        
        
            mean_loss = running_loss / running_total
            print(f'epoch {epoch}: batch {i} loss: {mean_loss} \r', end='')        
        history.append(mean_loss)
    print('\nFinished Training')
    return history

    
class KNN():
    """ Simple K nearest neighbour data structure """
    def __init__(self, embedding, word_to_idx):
        print('Create KNN')
        self.embedding = normalize(embedding.numpy(), axis=1)
        self.word_to_idx = word_to_idx

    def query(self, idx, k=5):
        tmp = {}
        for i in idx:
            tmp[i] = self.get_most_similar(i, k)
        return tmp

    def print_nearest(self, words, k=5):
        for x in words:
            idx = self.word_to_idx[x]
            k_near_idx = self.get_most_similar(idx, k)
            similar_words = [self.word_to_idx.inverse[z] for z in k_near_idx]
            print('Most Similar to {0}:'.format(x), ', '.join(similar_words))

    def get_most_similar(self, i, k):
        """ Get the indexes of the most similar embedding vectors 
    
            Args:
                i: int
                k: int
            Returns 
                k_nearest: list    
        """
        embed_i = self.embedding[i, :].reshape(-1, 1)
        scores = (self.embedding @ embed_i).ravel()
        ordered_sims = np.argsort(scores)[::-1]
        k_nearest = ordered_sims[1:k + 1] # i is probably includes
        assert ordered_sims[0] == i
        return k_nearest



def load_skipgram_data(path, data_size=-1):
    """ Loads the skip gram data"""
    with open(path, "r") as in_file:
        data = json.load(in_file)
    word_idx_mapping = bidict.bidict(data["dictionary"])
    if data_size > 0:
        X = data["X"][:data_size]
        Y = data["y"][:data_size]
    else: 
        X = data["X"]
        Y = data["y"]

    tX = torch.from_numpy(np.array(X))
    tY = torch.tensor(Y)

    dataset = torch_data.TensorDataset(tX, tY)
    dataloader = torch_data.DataLoader(dataset, batch_size=64, shuffle=False)     

    return dataset, dataloader, word_idx_mapping

def load_cbow_data(path, data_size=-1):
    """ Loads the skip gram data"""
    with open(path, "r") as in_file:
        data = json.load(in_file)
    word_idx_mapping = bidict.bidict(data["dictionary"])

    if data_size > 0:
        X = data["X"][:data_size]
        Y = data["y"][:data_size]
    else: 
        X = data["X"]
        Y = data["y"]

    tX = torch.from_numpy(np.array(X))
    tY = torch.tensor(Y)

    dataset = torch_data.TensorDataset(tX, tY)
    dataloader = torch_data.DataLoader(dataset, batch_size=64, shuffle=False)    
    return dataset, dataloader, word_idx_mapping



