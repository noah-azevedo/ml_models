class HomemadeSimpleLinearRegression(AibtMachineLearningModel):
    """ Logistic Regression classifier.
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If
        false then we use batch optimization by least squares.
    """
    def __init__(self, learning_rate=1e-3):
        self.param = None
        self.bias = None
        self.learning_rate = learning_rate

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))
        self.bias = np.random.uniform(-limit, limit, (1,))
    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X) # Initially the theta parameters are chosen randomly.
        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make a new prediction
            y_pred = X.dot(self.param) + np.ones(X.shape[0])*self.bias
            # Move against the gradient of the loss function with
            # respect to the parameters to minimize the loss
            self.param -= self.learning_rate * (-(y - y_pred).dot(X))
            self.bias -= self.learning_rate * np.mean((-(y - y_pred)))
    def predict(self, X):
        y_pred = X.dot(self.param) + np.ones(X.shape[0])*self.bias
        return y_pred