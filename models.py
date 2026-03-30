import nn

class PerceptronModel(object):
    def __init__(self, dimension):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimension` is the dimensionality of the data.
        For example, dimension=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimension)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, point):
        """
        Calculates the score assigned by the perceptron to a data point `point`.

        Inputs:
            point: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        dotprod = nn.DotProduct(self.get_weights(), point)
        return dotprod


    def get_prediction(self, point):
        """
        Calculates the predicted class for a single data point `point`.

        Returns: -1 or 1
        """
        "*** YOUR CODE HERE ***"
        dotprod = self.run(point)
        c = nn.as_scalar(dotprod)
        
        if c >= 0: return 1
        return -1

    def train_model(self, data):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        # x are the datapoints, y are the corresponding values
        # Loop until the model makes no mistakes in a full pass over the data
        while True:
            complete = True  # Assume no mistakes this iteration
            # Iterate through the dataset one example at a time
            # x = input point, y = true label (+1 or -1)
            for x, y in data.iterate_once(1):
                # Convert label node to scalar value
                true = nn.as_scalar(y)
                # Get model prediction for current input
                pred = self.get_prediction(x)
                # If prediction is incorrect, update weights
                if true != pred:
                    complete = False  # Mark that we made a mistake this pass
                    # Perceptron update rule:
                    # w = w + (true_label * x)
                    # This shifts weights toward correctly classifying this example
                    self.w.update(true, x)
            # If no mistakes were made in the entire pass, training is done
            if complete:
                break
        

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Hidden size
        hidden_size = 300
        # Batch Size
        self.batches = 20
        # Learning rate
        self.learning_rate = 0.01

        # Layer 1
        self.w1 = nn.Parameter(1, hidden_size)
        self.b1 = nn.Parameter(1, hidden_size)
        # Layer 2
        self.w2 = nn.Parameter(hidden_size,1)
        self.b2 = nn.Parameter(1,1)

        # lists of (layer,bias)
        self.sets = [(self.w1, self.b1), (self.w2, self.b2)]


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Let f be equal to the input
        f = x
        # For every weight and bias pair in the set of (layer, bias) pairs
        # Let z equal to the bias(b) added to f transformed by the weight(w)
        # Then let f be equal to ReLU(z)
        for w, b in self.sets:
            z = nn.AddBias(b, nn.Linear(f, w))
            f = nn.ReLU(z)

        # return the result
        return z
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Get the predicted y values by running on the input x
        y_pred = self.run(x)
        # return the squared loss between the predicted and actual y values
        return nn.SquareLoss(y, y_pred)

    def train_model(self, data):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # Flatten the list of (weight, bias) pairs into a single list of parameters
        para = []
        for pair in self.sets:
            para.append(pair[0])
            para.append(pair[1])

        # Total number of parameters
        n = len(para)

        running = True
        while running:

            # Set the loss value to zero
            loss_val = 0
            # Iterate through the dataset one batch at a time
            for x,y in data.iterate_once(self.batches):

                # Compute the loss for the current batch
                loss = self.get_loss(x,y)
                # Compute the loss as a scalar
                loss_val = nn.as_scalar(loss)
                # Compute gradients of the loss with respect to each parameter
                grads = nn.gradients(para, loss)

                # Update each parameter using gradient descent
                # param = param - learning_rate * gradient
                for i in range(n):
                    para[i].update(-self.learning_rate, grads[i])
                    
            # Stop training once the loss value is less than 0.001
            if loss_val <= 0.001:
                    running = False
                

                
                







class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Hidden size
        hidden_size = 300
        # Batch Size
        self.batches = 50
        # Learning rate
        self.learning_rate = 0.1

        # Layer 1
        self.w1 = nn.Parameter(784, hidden_size)
        self.b1 = nn.Parameter(1, hidden_size)
        # Layer 2
        self.w2 = nn.Parameter(hidden_size,10)
        self.b2 = nn.Parameter(1,10)

        # lists of (layer,bias)
        self.sets = [(self.w1, self.b1), (self.w2, self.b2)]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # Let f be equal to the input
        f = x

        # Let z equal to the bias(b1) added to f transformed by the weight(w1),
        # where w1 and b1 are the weight and bias of the first layer
        z = nn.AddBias(self.sets[0][1], nn.Linear(f, self.sets[0][0]))
        # Let f be equal to ReLU(z)
        f = nn.ReLU(z)
        # Let z equal to the bias(b2) added to f transformed by the weight(w2),
        # where w2 and b2 are the weight and bias of the second layer
        z = nn.AddBias(self.sets[1][1], nn.Linear(f, self.sets[1][0]))

        # return the result
        return z

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Get the predicted y values by running on the input x
        y_pred = self.run(x)
        # return the Softmax loss between the predicted and actual y values
        return nn.SoftmaxLoss(y_pred, y)

    def train_model(self, data):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # Flatten the list of (weight, bias) pairs into a single list of parameters
        para = []
        for pair in self.sets:
            para.append(pair[0])
            para.append(pair[1])

        # Total number of parameters
        n = len(para)

        running = True
        while running:
            
            # Iterate through the dataset one batch at a time
            for x,y in data.iterate_once(self.batches):

                # Compute the loss for the current batch
                loss = self.get_loss(x,y)
                # Compute gradients of the loss with respect to each parameter
                grads = nn.gradients(para, loss)

                # Update each parameter using gradient descent
                # param = param - learning_rate * gradient
                for i in range(n):
                    para[i].update(-self.learning_rate, grads[i])

            # Stop training once the validation accuracy is greater than %97.7
            if  data.get_validation_accuracy() >= 0.977:
                    running = False

