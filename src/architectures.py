import torch.nn as nn
import torch.nn.functional as F
import torch

class FNN(nn.Module):

    name = "FNN"

    def __init__(self, input_size = 5, num_neurons = 10, num_classes = 2, initialization = "random", init_scales=None):
        super(FNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_classes)
        )

        if init_scales is not None:
            self.linear_relu_stack[0].weight.data = torch.mul(self.linear_relu_stack[0].weight.data, init_scales[0])
            self.linear_relu_stack[2].weight.data = torch.mul(self.linear_relu_stack[2].weight.data, init_scales[1])
            

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output

    def plot_weights(self, ax, title = None):
        for layer in self.linear_relu_stack:
            if isinstance(layer, nn.Linear):
                ax.hist(layer.weight.data.flatten())
                ax.set_title(layer)
                
    @classmethod
    def add_args(cls,parser):
        parser.add_argument(f'--{cls.name}__num_neurons', type=int, help='Number of neurons in the hidden layer.', default=10)


class FNN_2layer(nn.Module):

    name = "FNN_2layer"

    def __init__(self, input_size = 5, num_neurons_layer1 = 10, num_neurons_layer2 = 10, num_classes = 2, initialization = "random", activation_func="relu", lsoftmax = False):
        super(FNN_2layer, self).__init__()
        #self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(input_size, num_neurons_layer1) 
        self.fc2 = nn.Linear(num_neurons_layer1, num_neurons_layer2) 
        self.fc3 = nn.Linear(num_neurons_layer2, num_classes)
        self.lsoftmax = lsoftmax

        if activation_func == "relu":
            self.activation = F.relu
        elif activation_func == "sigmoid":
            self.activation = F.sigmoid
        else:
            raise ValueError("Activation function not supported.")

        if initialization == "Kaiming":
            nn.init.kaiming_normal_(self.fc1.weight)
            #self.fc1.weight.data = torch.mul(self.fc1.weight.data, args.scales[0])
            nn.init.kaiming_uniform_(self.fc1.weight)
            #self.fc1.weight.data = torch.mul(self.fc1.weight.data, args.scales[1])
            nn.init.kaiming_normal_(self.fc2.weight)
            #self.fc1.weight.data = torch.mul(self.fc1.weight.data, args.scales[0])
            nn.init.kaiming_uniform_(self.fc2.weight)
            #self.fc1.weight.data = torch.mul(self.fc1.weight.data, args.scales[1])


    def forward(self, x):

        #print("Begin at: ", x.shape)
        x = self.activation(self.fc1(x))
        #print("1st layer: ", x.shape)
        x = self.activation(self.fc2(x))
        #print("2nd layer: ", x.shape)
        #x = self.drop_out(x)
        x = self.fc3(x)

        if self.lsoftmax:
            return F.log_softmax(x,dim=1)
        else:
            return x

    @classmethod
    def add_args(cls,parser):
        parser.add_argument(f'--{cls.name}__num_neurons_layer1', type=int, help='Number of neurons in the first hidden layer.', default=10)
        parser.add_argument(f'--{cls.name}__num_neurons_layer2', type=int, help='Number of neurons in the second hidden layer.', default=10)

    def reparameterize(self,alpha1,alpha2,alpha3):
        """Reparameterize the network such that it hase the same output function, but is in a different context of loss landscape."""
        alphas = torch.Tensor([alpha1,alpha2,alpha3])
        assert alphas.prod() == 1, "Alphas must factor to 1."
        self.fc1.weight.data = torch.mul(self.fc1.weight.data, alpha1)
        self.fc2.weight.data = torch.mul(self.fc2.weight.data, alpha2)
        self.fc3.weight.data = torch.mul(self.fc3.weight.data, alpha3)

        self.fc1.bias.data = torch.mul(self.fc1.bias.data, alpha1)
        self.fc2.bias.data = torch.mul(self.fc2.bias.data, alpha1 * alpha2)

    def scale(self,factor):
        self.fc1.weight.data = torch.mul(self.fc1.weight.data, factor)
        self.fc2.weight.data = torch.mul(self.fc2.weight.data, factor)
        self.fc3.weight.data = torch.mul(self.fc3.weight.data, factor)

        self.fc1.bias.data = torch.mul(self.fc1.bias.data, factor)
        self.fc2.bias.data = torch.mul(self.fc2.bias.data, factor)
        self.fc3.bias.data = torch.mul(self.fc3.bias.data, factor)


class CNN(nn.Module):
    name = "CNN"

    def __init__(self, input_size=1, num_classes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3, #5
                stride=1,
                padding=0, #2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,  # 5
                stride=1,
                padding=0,  # 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # fully connected layer, output num_classes classes
        self.out = nn.Linear(32 * 5 * 5, num_classes)

    def forward(self, x):
        x = self.conv1(x.float())
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 5 * 5)
        x = x.view(x.size(0), -1)  #can use to visualize
        output = self.out(x)
        return output
    
def model_l2_norm(model):
    parameters = model.parameters()
    l2_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2.0) for g in parameters]), 2.0)
    return l2_norm.item()