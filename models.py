# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchviz
from IPython.display import display
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Define a class for the AlphabetModel
class AlphabetModel():
    def __init__(self, X_train, y_train, X_test, y_test):
        # Get the input, hidden, and output sizes
        self.input_size = len(X_train.columns)
        self.hidden_size1 = len(X_train.columns) * 2 + 3
        self.hidden_size2 = len(X_train.columns) * 2 - 3
        self.output_size = len(set(y_train))
        
        alphabet = {}
        c = 0
        for char in range(ord('A'), ord('Z') + 1):
            alphabet[chr(char)] = c
            c += 1
            
        y_train = y_train.map(alphabet)
        y_test = y_test.map(alphabet)
        # Convert training and testing data to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train.to_numpy())
        self.y_train = torch.LongTensor(y_train.to_numpy())
        
        # Create a training dataset
        self.train = TensorDataset(self.X_train, self.y_train)
        
        self.X_test = torch.FloatTensor(X_test.to_numpy())
        self.y_test = torch.LongTensor(y_test.to_numpy())
        
        # Create a testing dataset
        self.test = TensorDataset(self.X_test, self.y_test)
        
        # Initialize a neural network
        self.nn = NeuralNet(self.input_size, self.hidden_size1, self.hidden_size2, self.output_size)
    
    # Method to train the model
    def fit(self, stop_earlier=25, max_iter = 200, batch_size=32):
        # Define the loss function (CrossEntropyLoss) and optimizer (Adam)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.nn.parameters(), lr=0.001)
        
        # Create a data loader for training data
        train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=True)
        
        losses = []
        c = 0
        epoch = 0
        while True:
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.nn(batch_x)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')
            
            if(len(losses) != 0):
                if(loss.item() > min(losses)):
                    c += 1
                else:
                    c = 0
            losses.append(loss.item())
            
            epoch += 1
            
            if(c == stop_earlier):
                print(f"Stopping earlier because no improvement in {stop_earlier} epochs")
                break
            elif(epoch == max_iter):
                break
        return epoch, losses
    
    # Method to evaluate the model's accuracy
    def score(self, batch_size=32):
        # Set the model to evaluation mode
        self.nn.eval()
        correct = 0
        total = 0

        # Create a data loader for testing data
        test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=True)
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.nn(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

    def predict(self, batch_size=32):
        # Set the model to evaluation mode
        self.nn.eval()
        
        # Create a data loader for testing data
        test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=True)
        y_pred = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.nn(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.append(predicted)
        
        return y_pred
    
    # Method to visualize the model using torchviz
    def __repr__(self):
        # Create a dummy input tensor with the same shape as your actual input data
        dummy_input = torch.randn(1, 63)
        # Use torchviz to visualize the model
        display(torchviz.make_dot(self.nn(dummy_input), params=dict(self.nn.named_parameters())))
       
        return ""

# Define a neural network class (NeuralNet) as a subclass of nn.Module
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNet, self).__init__()
        
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    # Forward pass through the network
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.softmax(out)

        return out
