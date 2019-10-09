import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


data_path = './torch_data'
### YOUR CODE HERE set path to where to store data
### END CODE
_trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transform)
_trainloader = torch.utils.data.DataLoader(_trainset, batch_size=32,
                                          shuffle=True, num_workers=4)

_testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=transform)
_testloader = torch.utils.data.DataLoader(_testset, batch_size=32,
                                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

### An example network
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        """ Implement the forward pass
            
            Args:
            dat: torch.tensor shape (mini_batch_size, channels, width, height) which is (-1, 3, 32, 32)
            
            return x, torch.tensor shape (mini_batch_size, output_size)
        """

        x = x.view(-1, 32 *32 * 3) # reshape data from image style to standard matrix of data in rows
        x = self.fc1(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
### YOUR CODE HERE - Implement a convolutional network        
### END CODE

class FullNet(nn.Module):
    def __init__(self):
        super(FullNet, self).__init__()
### YOUR CODE HERE - Implement a fully connected network of at least two layers with 128 neurons
### END CODE




def score(net, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc

def fit(net, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    optimizer = optim.Adam(net.parameters())
    history = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:   # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print(f'epoch {epoch} done')
        train_acc = score(net, train_loader)
        print('Train Accuracy:', train_acc)
        test_acc = score(net, test_loader)
        print('Test Accuracy:', test_acc)
        
        history.append((train_acc, test_acc))
    print('Finished Training')

    return history

def test_net(net, epochs=10):
    linear_hist = fit(net, _trainloader, _testloader, epochs=epochs)
    fig, ax = plt.subplots(1, 1)
    xc = list(range(len(linear_hist)))
    ax.plot(xc, [x[0] for x in linear_hist], 'b-', label='train accuracy')
    ax.plot(xc, [x[1] for x in linear_hist], 'r-', label='validation accuracy')
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_ylim([0,1])
    ax.legend()
    ax.set_title('Cifar 10 Accuracy per epoch', fontsize=20)
    plt.show()

#print('Testing Linear Model')
#net = LinearNet()
#test_net(net)
#print('Testing fully connected network')
#fnet = FullNet()
#test_net(fnet)
print('testing convolutional network')
cnet = ConvNet()
test_net(cnet)
