import nn
import backend

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

        while True:
            complete = True
            for x,y in data.iterate_once(1):
                true = nn.as_scalar(y)
                pred = self.get_prediction(x)
                if true != pred:
                    complete = False
                    self.w.update(true, x)

            if complete == True:
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
        hidden_size = 300
        self.batches = 100
        self.learning_rate = 0.01
        # 1 layer
        self.w1 = nn.Parameter(1, hidden_size)
        self.b1 = nn.Parameter(1, hidden_size)
        #layer 2
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
        f = x
        for w, b in self.sets:
            z = nn.AddBias(b, nn.Linear(f, w))
            f = nn.ReLU(z)
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
        y_pred = self.run(x)
        return nn.SquareLoss(y, y_pred)

    def train_model(self, data):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        para = []
        for pair in self.sets:
            para.append(pair[0])
            para.append(pair[1])
        n = len(para)

        running = True
        while running:
            
            loss_val = 0

            for x,y in data.iterate_once(self.batches):
                
                loss = self.get_loss(x,y)
                loss_val = nn.as_scalar(loss)
                # list of gradient changes WRT to the parameters in order
                grads = nn.gradients(para, loss)

                for i in range(n):
                    para[i].update(-self.learning_rate, grads[i])
            
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
        hidden_size = 300
        self.batches = 100
        self.learning_rate = 0.1
        # 1 layer
        self.w1 = nn.Parameter(784, hidden_size)
        self.b1 = nn.Parameter(1, hidden_size)
        #layer 2
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
        f = x
        
        z = nn.AddBias(self.sets[0][1], nn.Linear(f, self.sets[0][0]))
        f = nn.ReLU(z)
        z = nn.AddBias(self.sets[1][1], nn.Linear(f, self.sets[1][0]))

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
        
        y_pred = self.run(x)
        return nn.SoftmaxLoss(y_pred, y)

    def train_model(self, data):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        para = []
        for pair in self.sets:
            para.append(pair[0])
            para.append(pair[1])
        n = len(para)

        running = True
        while running:
            
            loss_val = 0

            for x,y in data.iterate_once(self.batches):
                
                loss = self.get_loss(x,y)
                loss_val = nn.as_scalar(loss)
                # list of gradient changes WRT to the parameters in order
                grads = nn.gradients(para, loss)

                for i in range(n):
                    para[i].update(-self.learning_rate, grads[i])

            print(data.get_validation_accuracy())
            if  data.get_validation_accuracy() >= 0.9775:
                    running = False

