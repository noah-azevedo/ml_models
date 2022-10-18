def distance(x1, x2):
        """ Calculates the l2 distance between two vectors """
        distance = 0
        # Squared distance between each coordinate
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i]), 2)
        return math.sqrt(distance)

class HomemadeKNeighborsClassifier(AibtMachineLearningModel):
    """ K Nearest Neighbors classifier.
    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the 
        sample that we wish to predict.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def decision_rule(self, neighbor_labels):
        """ Return the most common class among the neighbor samples """
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()
        
    
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = np.empty(X_test.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([distance(test_sample, x) for x in self.X_train])[:self.n_neighbors]
            # Extract the labels of the K nearest neighboring training samples
            #k_nearest_neighbors = np.array([self.y_train[i] for i in idx])
            k_nearest_neighbors = self.y_train[idx]
            # Label sample as the most common class label
            y_pred[i] = self.decision_rule(k_nearest_neighbors)

        return y_pred