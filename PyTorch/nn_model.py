'''the purpose of this script is to perform logistic regression'''

import torch
import torch.nn as nn

class BinaryClassification(nn.Module):
    '''pytorch model with linear layers and a sigmoid activation function for binary awarness questions and a softmax function of likert scale questions'''
    def __init__(self, input_size, hidden_size):
        super(BinaryClassification, self).__init__()
        #note the layers are specified as linear because the model will perform a linear transformation of the input data. The linear transformation allows the model to perform matrix multiplication to deterimine the weight and bias. The weight is the pattern being picked up in the 2nd layer and the bias is how high weighted sum needs to be before going through the activation function.
        self.input_layer = nn.Linear(input_size, hidden_size) # this is the first layer of the NN, the number of features = input size
        self.hidden_layer = nn.Linear(hidden_size, hidden_size) # layers between the input and output layer that process the data by applying complex non-linear functions. The hidden layers could consist of one or more
        
        self.output_layer = nn.Linear(hidden_size, 1) # the last layer has the number of specified outputs (questions), each with an activation (between 0 and 1). The last layer is the predicitons

        # activation functions
        self.relu = nn.ReLU() # Applies the rectified linear unit function element-wise. This model uses the ReLu activation function for the hidden layers , not the output layer.
        self.sigmoid = nn.Sigmoid() # the sigmoid (aka logistic curve) is the logistic curve that'll compress the real number line down to an activation between 0 and 1. The sigmoid activation function will be used on the final output layer.

    def forward(self, x):
        '''Forward function to define how the input features move through the neural net and how to final output is transformed'''
        x = self.input_layer(x)
        x = self.relu(x) #introduce non-linearity
        x = self.hidden_layer(x)
        x = self.relu(x) #maintain the non-linearity throughout the network

        output = self.output_layer(x)
        output = self.sigmoid(output) #squish output to 0 and 1
        
        return output