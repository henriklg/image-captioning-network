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
        self.vocabulary_size    = config['vocabulary_size']
        self.embedding_size     = config['embedding_size']
        self.VggFc7Size         = config['VggFc7Size']
        self.hidden_state_sizes = config['hidden_state_sizes']
        self.num_rnn_layers     = config['num_rnn_layers']
        self.cell_type          = config['cellType']

        # ToDo
        self.Embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)

        self.inputLayer = nn.Linear(self.VggFc7Size, self.hidden_state_sizes)

        self.rnn = RNN(self.embedding_size, self.hidden_state_sizes, self.num_rnn_layers, cell_type=self.cell_type)

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
        # ToDO
        # Get "initial_hidden_state" shape[num_rnn_layers, batch_size, hidden_state_sizes].
        # Remember that each rnn cell needs its own initial state.

        # use self.rnn to calculate "logits" and "current_hidden_state"
        
        if current_hidden_state is None: 
            in_put = torch.tanh(self.inputLayer(vgg_fc7_features))
            current_hidden_state = in_put.repeat(repeats=(self.num_rnn_layers, 1, 1))
        
        logits, current_hidden_state_out = self.rnn.forward(xTokens, current_hidden_state, self.outputLayer, self.Embedding, is_train)

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
        self.input_size        = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers    = num_rnn_layers
        self.cell_type         = cell_type

        # ToDo
        # Your task is to create a list (self.cells) of type "nn.ModuleList" and populated it with cells of type "self.cell_type".
        if self.cell_type == "RNN":
            cellList = [RNNCell(hidden_state_size, input_size)]
            new_input_size = hidden_state_size
            cellList.extend([RNNCell(hidden_state_size, new_input_size) for i in range(num_rnn_layers-1)])
            
            self.cells = nn.ModuleList(cellList)
        elif self.cell_type == "GRU":
            cellList = [GRUCell(hidden_state_size, input_size)]
            new_input_size = hidden_state_size
            cellList.extend([GRUCell(hidden_state_size, new_input_size) for i in range(num_rnn_layers-1)])
            
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
        if is_train==True:
            seqLen = xTokens.shape[1] #truncated_backprop_length
        else:
            seqLen = 40 #Max sequence length to be generated
            
        h_old = initial_hidden_state
        L = xTokens.shape[1]
        log_list = []
        x_emb = Embedding(xTokens) #shape = [batch_size, seqLen, input_len] 

        
        if is_train:   
            # Iterate over time steps
            for ii in range(0, seqLen):
                
                x = x_emb[:, ii, :]
                h_new = torch.zeros_like(h_old)
                
                # Pass through layers
                for layer in range(self.num_rnn_layers):
                    out = self.cells[layer].forward(x, h_old[layer, :, :])
                    h_new[layer, :, :] = out
                    
                    x = out
                    
                log_list.append(outputLayer(x))
                h_old = h_new
                
        else:
            x = x_emb[:, 0, :]
            for ii in range(0, seqLen):
                h_new = torch.zeros_like(h_old)
                
                # Pass through layers
                for layer in range(self.num_rnn_layers):
                    out = self.cells[layer].forward(x, h_old[layer, :, :])
                    h_new[layer, :, :] = out
                    
                    x = out
                    
                # Get new word
                logits = outputLayer(out)
                _, x = torch.max(logits, dim=1)
                x = Embedding(x)
                    
                log_list.append(logits)
                h_old = h_new
                
            #for ii in range(0, seqLen):
        logits = torch.stack(log_list, dim=1)

                
            

        # ToDo
        # While iterate through the (stacked) rnn, it may be easier to use lists instead of indexing the tensors.
        # You can use "list(torch.unbind())" and "torch.stack()" to convert from pytorch tensor to lists and back again.

        # get input embedding vectors

        # Use for loops to run over "seqLen" and "self.num_rnn_layers" to calculate logits

        # Produce outputs
        #logits        = 
        current_state = h_new
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

        # TODO:
        self.weight_u = torch.nn.Parameter(torch.randn(hidden_state_size+input_size, hidden_state_size)/np.sqrt(hidden_state_size+input_size))
        self.bias_u   = torch.nn.Parameter(torch.zeros(1, hidden_state_size))

        self.weight_r = torch.nn.Parameter(torch.randn(hidden_state_size+input_size, hidden_state_size)/np.sqrt(hidden_state_size+input_size))
        self.bias_r   = torch.nn.Parameter(torch.zeros(1, hidden_state_size))

        self.weight = torch.nn.Parameter(torch.randn(hidden_state_size+input_size, hidden_state_size)/np.sqrt(hidden_state_size+input_size))
        self.bias   = torch.nn.Parameter(torch.zeros(1, hidden_state_size))
        return

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """

        gate_update = torch.sigmoid(torch.cat((x, state_old), 1).mm(self.weight_u)+self.bias_u)
        gate_reset = torch.sigmoid(torch.cat((x, state_old), 1).mm(self.weight_r)+self.bias_r)
        state_hat = torch.tanh(torch.cat((x, gate_reset*state_old), 1).mm(self.weight)+self.bias)
        state_new = gate_update*state_old+(1-gate_update)*state_hat
        return state_new

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

        self.weight = torch.nn.Parameter(torch.randn(hidden_state_size+input_size, hidden_state_size)/np.sqrt(hidden_state_size+input_size))
        self.bias   = torch.nn.Parameter(torch.zeros(1, hidden_state_size))
        return


    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        # TODO:
        state_new = nn.functional.tanh(torch.cat((x, state_old), 1).mm(self.weight) + self.bias)    
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
    eps = 0.0000000001 #used to not divide on zero
    
    # TODO:
    #print(logits.shape)
    
    _, _, v_size = logits.shape
    #*yWeights.view(-1)[:, None]
    #*yWeights.long().view(-1)

    sumLoss = torch.sum(F.cross_entropy(input=logits.view(-1, v_size), target=yTokens.view(-1), reduction='none')*yWeights.view(-1))
    
    N = yWeights.view(-1).nonzero().size()[0]
    meanLoss = sumLoss/N+eps

    return sumLoss, meanLoss


