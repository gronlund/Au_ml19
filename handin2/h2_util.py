import numpy as np

from sklearn.datasets import fetch_openml

def load_digits_train_data():
    Xm, ym = fetch_openml('mnist_784', version=1, return_X_y=True)
    #mnist = fetch_mldata('MNIST original')
    #Xm = mnist.data
    #ym = mnist.target
    X_train = Xm[0:60000, :]/256.0
    y_train = ym[0:60000].squeeze().astype(int)
    return X_train, y_train

def load_digits_test_data():
    Xm, ym = fetch_openml('mnist_784', version=1, return_X_y=True)
    X_train = Xm[60001:, :]/256.0
    y_train = ym[60001:].squeeze().astype(int)
    return X_train, y_train


def print_score(classifier, X_train, X_test, y_train, y_test):
    """ Simple print score function that prints train and test score of classifier - almost not worth it"""
    print('In Sample Score: ',
          classifier.score(X_train, y_train))
    print('Test Score: ',
          classifier.score(X_test, y_test))


def numerical_grad_check(f, x):
    """ Numerical Gradient Checker """
    eps = 1e-6
    h = 1e-4
    # d = x.shape[0]
    cost, grad = f(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        dim = it.multi_index
        tmp = x[dim]
        x[dim] = tmp + h
        cplus, _ = f(x)
        x[dim] = tmp - h 
        cminus, _ = f(x)
        x[dim] = tmp
        num_grad = (cplus-cminus)/(2*h)
        print('grad, num_grad, grad-num_grad', grad[dim], num_grad, grad[dim]-num_grad)
        assert np.abs(num_grad - grad[dim]) < eps, 'numerical gradient error index {0}, numerical gradient {1}, computed gradient {2}'.format(dim, num_grad, grad[dim])
        it.iternext()

    
            
            
    
