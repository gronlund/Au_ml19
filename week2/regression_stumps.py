import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pdb 
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pydotplus


def plot_tree(dtree, feature_names):
    """ helper function """
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    print('exporting tree to dtree.png')
    graph.write_png('dtree.png')


class RegressionStump():
    
    def __init__(self):
        """ The state variables of a stump"""
        self.idx = None
        self.val = None
        self.left = None
        self.right = None
    
    def fit(self, data, targets):
        """ Fit a decision stump to data
        
        Find the best way to split the data in feat  minimizig the cost (0-1) loss of the tree after the split 
    
        Args:
           data: np.array (n, d)  features
           targets: np.array (n, ) targets
    
        sets self.idx, self.val, self.left, self.right
        """
        # update these three
        self.idx = 0
        self.val = None
        self.left = None
        self.right = None
        ### YOUR CODE HERE
        ### END CODE

    def predict(self, X):
        """ Regression tree prediction algorithm

        Args
            X: np.array, shape n,d
        
        returns pred: np.array shape n,  model prediction on X
        """
        pred = None
        ### YOUR CODE HERE
        ### END CODE
        return pred
    
    def score(self, X, y):
        """ Compute accuracy of model

        Args
            X: np.array, shape n,d
            y: np.array, shape n, 

        returns out: scalar - means least scores cost
        """
        out = None
        ### YOUR CODE HERE
        ### END CODE
        return out
        

### YOUR CODE HERE
### END CODE


def main():
    """ Simple method testing """
    boston = load_boston()
    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(boston.data,
                                                        boston.target,
                                                        test_size=0.2)

    baseline_accuracy = np.mean((y_test-np.mean(y_train))**2)
    print('Least Squares Cost of learning mean of training data:', baseline_accuracy) 
    print('Lets see if we can do better with just one question')
    D = RegressionStump()
    D.fit(X_train, y_train)
    print('idx, val, left, right', D.idx, D.val, D.left, D.right)
    print('Feature name of idx', boston.feature_names[D.idx])
    print('Score of model', D.score(X_test, y_test))
    print('lets compare with sklearn decision tree')
    dc = DecisionTreeRegressor(max_depth=1)
    dc.fit(X_train, y_train)
    dc_score = ((dc.predict(X_test)-y_test)**2).mean()
    print('dc score', dc_score)
    print('feature names - for comparison', list(enumerate(boston.feature_names)))
    plot_tree(dc, boston.feature_names)

if __name__ == '__main__':
    main()