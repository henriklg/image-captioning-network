from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


######################################################################################################################
class imageCaptionModel(nn.Module):
    def __init__(self, config):
        super(imageCaptionModel, self).__init__()
        """
        "imageCaptionModel" is the main module class for the image captioning network
        
        Args:
            config: Dictionary holding neural network configuration

        Returns:
            self.Embedding  : An instance of nn.Embedding, shape[vocabulary_size, embedding_size]
            self.inputLayer : An instance of nn.Linear, shape[VggFc7Size, hidden_state_sizes]
            self.rnn        : An instance of RNN
            self.outputLayer: An instance of nn.Linear, shape[hidden_state_sizes, vocabulary_size]
        """
        self.config = config
        self.vocabulary_size = config['vocabulary_size']
        self.embedding_size = config['embedding_size']
        self.VggFc7Size = config['VggFc7Size']
        self.hidden_state_sizes = config['hidden_state_sizes']
        self.num_rnn_layers = config['num_rnn_layers']
        self.cell_type = config['cellType']

        # TODO: task 1e - a
        self.Embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.inputLayer = nn.Linear(self.VggFc7Size, self.hidden_state_sizes)
        self.rnn = RNN(self.embedding_size, self.hidden_state_sizes, self.num_rnn_layers, self.cell_type)
        self.outputLayer = nn.Linear(self.hidden_state_sizes, self.vocabulary_size)
        return

    def forward(self, vgg_fc7_features, xTokens, is_train, current_hidden_state=None):
        """
        Args:
            vgg_fc7_features    : Features from the VGG16 network, shape[batch_size, VggFc7Size]
            xTokens             : Shape[batch_size, truncated_backprop_length]
            is_train            : "is_train" is a flag used to select whether or not to use estimated token as input
            current_hidden_state: If not None, "current_hidden_state" should be passed into the rnn module
                                  shape[num_rnn_layers, batch_size, hidden_state_sizes]

        Returns:
            logits              : Shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_hidden_state: shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # TODO: task 1e - b
        # Get "initial_hidden_state" shape[num_rnn_layers, batch_size, hidden_state_sizes].
        # Remember that each rnn cell needs its own initial state.

        batch_size = xTokens.shape[0]
        if current_hidden_state is None:
            inputt = torch.tanh(self.inputLayer(vgg_fc7_features))
            initial_hidden_state = inputt.repeat(repeats=(self.num_rnn_layers, 1, 1))
        else:
            initial_hidden_state = current_hidden_state

        # use self.rnn to calculate "logits" and "current_hidden_state"
        logits, current_hidden_state_out = self.rnn.forward(xTokens, initial_hidden_state, self.outputLayer, self.Embedding, is_train)

        return logits, current_hidden_state_out


######################################################################################################################
class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, cell_type='RNN'):
        super(RNN, self).__init__()
        """
        Args:
            input_size (Int)        : embedding_size
            hidden_state_size (Int) : Number of features in the rnn cells (will be equal for all rnn layers) 
            num_rnn_layers (Int)    : Number of stacked rnns
            cell_type               : Whether to use vanilla or GRU cells
            
        Returns:
            self.cells              : A nn.ModuleList with entities of "RNNCell" or "GRUCell"
        """
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers = num_rnn_layers
        self.cell_type = cell_type

        # ToDo
        # Your task is to create a list (self.cells) of type "nn.ModuleList" and populated it with cells of type "self.cell_type".
        if self.cell_type == "RNN":
            cellList = [RNNCell(hidden_state_size, input_size)]
            new_input_size = hidden_state_size
            cellList.extend([RNNCell(hidden_state_size, new_input_size) for i in range(num_rnn_layers - 1)])

            self.cells = nn.ModuleList(cellList)
        elif self.cell_type == "GRU":
            cellList = [GRUCell(hidden_state_size, input_size)]
            new_input_size = hidden_state_size
            cellList.extend([GRUCell(hidden_state_size, new_input_size) for i in range(num_rnn_layers - 1)])

            self.cells = nn.ModuleList(cellList)
        else:
            print('Error: cell type not recognized')

        return


    def forward(self, xTokens, initial_hidden_state, outputLayer, Embedding, is_train=True):
        """
        Args:
            xTokens:        shape [batch_size, truncated_backprop_length]
            initial_hidden_state:  shape [num_rnn_layers, batch_size, hidden_state_size]
            outputLayer:    handle to the last fully connected layer (an instance of nn.Linear)
            Embedding:      An instance of nn.Embedding. This is the embedding matrix.
            is_train:       flag: whether or not to feed in the predicated token vector as input for next step

        Returns:
            logits        : The predicted logits. shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_state : The hidden state from the last iteration (in time/words).
                            Shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """

        if is_train:
            seqLen = xTokens.shape[1]  # truncated_backprop_length
        else:
            seqLen = 40  # Max sequence length to be generated

        # ToDo: task 1d
        # While iterate through the (stacked) rnn, it may be easier to use lists instead of indexing the tensors.
        # You can use "list(torch.unbind())" and "torch.stack()" to convert from pytorch tensor to lists and back again.


        listed_states = list(torch.unbind(initial_hidden_state))
        num_rnn_layers = list(initial_hidden_state.size())[0]
        batch_size = list(xTokens.size())[0]
        # get input embedding vectors
        embedding_vectors = Embedding(xTokens)
        # not sure how to get out of embed

        # Use for loops to run over "seqLen" and "self.num_rnn_layers" to calculate logits

        if is_train == True:
            listed_logits = []
            for i in range(seqLen):
                data_prev_RNN = embedding_vectors[:, i, :]
                for j in range(num_rnn_layers):
                    data_RNN = self.cells[j](data_prev_RNN, listed_states[j])
                    listed_states[j] = data_RNN
                    data_prev_RNN = data_RNN
                listed_logits.append(outputLayer(data_prev_RNN))

            # Produce outputs

            # restack

            logits = torch.stack(listed_logits, dim=1)  # initiate tensor of size [batch_size,seqLen,vocabulary_size]
            current_state = torch.stack(listed_states)  # next state?
        else:
            listed_logits = []
            data_prev_RNN = embedding_vectors[:, 0, :]
            for i in range(seqLen):
                for j in range(num_rnn_layers):
                    data_RNN = self.cells[j](data_prev_RNN, listed_states[j])
                    listed_states[j] = data_RNN
                    data_prev_RNN = data_RNN
                logit = outputLayer(data_prev_RNN)
                listed_logits.append(logit)
                values, max_index = torch.max(logit, dim=1, keepdim=False)
                data_prev_RNN = Embedding(max_index)

            # Produce outputs

            # restack

            logits = torch.stack(listed_logits, dim=1)  # initiate tensor of size [batch_size,seqLen,vocabulary_size]
            current_state = torch.stack(listed_states)  # next state?


        return logits, current_state


########################################################################################################################
class GRUCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(GRUCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight_u: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean. 

            self.weight_r: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean. 

            self.weight: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean. 

            self.bias_u: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

            self.bias_r: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_sizes = hidden_state_size
        self.input_size = input_size

        # TODO: task 1b
        n = hidden_state_size + input_size

        self.weight_u = torch.nn.Parameter(torch.randn(n, hidden_state_size) / np.sqrt(n))
        self.bias_u = torch.nn.Parameter(torch.zeros(1, hidden_state_size))

        self.weight_r = torch.nn.Parameter(torch.randn(n, hidden_state_size) / np.sqrt(n))
        self.bias_r = torch.nn.Parameter(torch.zeros(1, hidden_state_size))

        self.weight = torch.nn.Parameter(torch.randn(n, hidden_state_size) / np.sqrt(n))
        self.bias = torch.nn.Parameter(torch.zeros(1, hidden_state_size))
        return

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        # TODO: task 1b
        gam_u = torch.sigmoid(torch.cat((x, state_old), 1).mm(self.weight_u) + self.bias_u)
        gam_r = torch.sigmoid(torch.cat((x, state_old), 1).mm(self.weight_r) + self.bias_r)
        h_hat = torch.tanh(torch.cat((x, (gam_r*state_old)), 1).mm(self.weight) + self.bias)
        h = gam_u * state_old + (1 - gam_u) * h_hat

        return h   # h = state_new


######################################################################################################################
class RNNCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(RNNCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight: A nn.Parameter with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean.

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_size = hidden_state_size
        self.input_size = input_size

        # TODO: task 1a
        n = hidden_state_size+input_size
        self.weight = torch.nn.Parameter(torch.randn(n, hidden_state_size)/np.sqrt(n))
        self.bias = torch.nn.Parameter(torch.zeros(1, hidden_state_size))
        return

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        # TODO: task 1a
        state_new = torch.tanh(torch.cat((x, state_old), 1).mm(self.weight) + self.bias)
        return state_new


######################################################################################################################
def loss_fn(logits, yTokens, yWeights):
    """
    Weighted softmax cross entropy loss.

    Args:
        logits          : shape[batch_size, truncated_backprop_length, vocabulary_size]
        yTokens (labels): Shape[batch_size, truncated_backprop_length]
        yWeights        : Shape[batch_size, truncated_backprop_length]. Add contribution to the total loss only from words exsisting 
                          (the sequence lengths may not add up to #*truncated_backprop_length)

    Returns:
        sumLoss: The total cross entropy loss for all words
        meanLoss: The averaged cross entropy loss for all words

    Tips:
        F.cross_entropy
    """
    eps = 0.0000000001  # used to not divide on zero

    # TODO: task 1c
    c = np.shape(logits)[2]                        # number of classes

    inputs = logits.view(-1, c)                    # (N,C)
    targets = yTokens.view(-1)                     # (N)
    loss = F.cross_entropy(inputs, targets, reduction='none')

    sumLoss = torch.sum(loss*yWeights.view(-1))
    meanLoss = sumLoss/(yWeights.view(-1).nonzero().size()[0] + eps)

    return sumLoss, meanLoss
