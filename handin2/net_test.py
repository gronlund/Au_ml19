import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from net_classifier import NetClassifier, get_init_params
from argparse import ArgumentParser
from h2_util import load_digits_train_data, load_digits_test_data


def export_fig(fig, name):
    result_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print('outputting to file', name)
    my_path = os.path.join(result_path, name)
    fig.savefig(my_path)

def digits_test(hidden_size=256, epochs=10, batch_size=32, lr=0.1, c=1e-4):
    net = NetClassifier()
    digits, labels = load_digits_train_data()
    digits_train, digits_val, labels_train, labels_val = train_test_split(digits, labels, test_size=0.15, random_state=42)
    init_params = get_init_params(digits.shape[1], hidden_size, 10)    
    hist = net.fit(digits_train, labels_train, digits_val, labels_val, init_params, batch_size=batch_size, epochs=epochs, lr=lr, c=c)
    print('in sample accuracy', net.score(digits, labels))
    test_digits, test_labels = load_digits_test_data()
    print('test sample accuracy', net.score(test_digits, test_labels))
    fig, ax = plt.subplots(1, 2, figsize=(20, 16))  
    idx = list(range(len(hist['train_loss'])))
    ax[0].plot(idx, hist['train_loss'], 'r-', linewidth=2, label='train loss')
    ax[0].plot(idx, hist['val_loss'], 'b-', linewidth=2, label='val loss')
    ax[0].set_title('Loss Per Epoch', fontsize=20)
    ax[0].set_ylabel('Loss', fontsize=16)
    ax[0].set_xlabel('Epoch', fontsize=16)
    ax[0].set_ylim([0, 1])
    ax[1].plot(idx, hist['train_acc'], 'r-', linewidth=2, label='train acc')
    ax[1].plot(idx, hist['val_acc'], 'b-', linewidth=2, label='val acc')
    ax[1].set_title('Acccuracy Per Epoch', fontsize=20)
    ax[1].set_ylim([0.5, 1])
    ax[1].set_ylabel('Accuracy', fontsize=16)
    ax[1].set_xlabel('Epoch', fontsize=16)
    plt.legend(fontsize=12)
    export_fig(fig, 'epoch_plots.png')
    return net

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-lr', dest='lr', type=float, default=-1)
    parser.add_argument('-bs', type=int, dest='batch_size', default=-1)
    parser.add_argument('-epochs', dest='epochs', type=int, default=-1)    
    parser.add_argument('-hidden', dest='hidden', type=int, default=-1)    

    args = parser.parse_args()
    print('args', args)
    kwargs = {}
    if args.lr >= 0:
        kwargs['lr'] = args.lr
    if args.batch_size >= 0:
        kwargs['batch_size'] = args.batch_size
    if args.epochs >= 0:
        kwargs['epochs'] = args.epochs            
    if args.hidden >= 0:
        kwargs['hidden_size'] = args.hidden

    digits_test(**kwargs)

