# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchviz
from sklearn.model_selection import train_test_split
from IPython.display import display
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Define a class for the AlphabetModel
class AlphabetModel():
    def __init__(self):
        pass
    
    def set_nn(self, X, y):
        # Get the input, hidden, and output sizes
        self.input_size = len(X.columns)
        self.hidden_size1 = len(X.columns) * 2 + 3
        self.hidden_size2 = len(X.columns) * 2 - 3
        self.output_size = len(set(y))
        
        # Initialize a neural network
        self.nn = NeuralNet(self.input_size, self.hidden_size1, self.hidden_size2, self.output_size)
        
    def __set_dataset(self, X, y):
        
        alphabet = {}
        c = 0
        for char in range(ord('A'), ord('Z') + 1):
            alphabet[chr(char)] = c
            c += 1
            
        y = y.map(alphabet)
        
        # Convert training and testing data to PyTorch tensors
        X = torch.FloatTensor(X.to_numpy())
        y = torch.LongTensor(y.to_numpy())
        
        # Create a training dataset
        return TensorDataset(X, y)
    
    # Method to train the model
    def fit(self, X_train, y_train, val_size = 0.1, stop_earlier=50, max_iter = 200, batch_size=32):
        self.set_nn(X_train, y_train)
        
        # Divide train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=10)
        
        self.train = self.__set_dataset(X_train, y_train)
        self.val = self.__set_dataset(X_val, y_val)
        # Define the loss function (CrossEntropyLoss) and optimizer (Adam)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.nn.parameters(), lr=0.001)
        
        # Create a data loader for training data
        train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=True)
        
        # Create a data loader for val data
        val_loader = DataLoader(self.val, batch_size=batch_size, shuffle=False)
        
        losses = []
        accuracies = []
        val_losses = []
        val_accuracies = []
        
        c = 0
        epoch = 0
        
        while True:
            # Evaluate the model on the training set
            for batch_x, batch_y in train_loader:
                outputs = self.nn(batch_x)

                # Backward pass and optimization
                optimizer.zero_grad()
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
            t_loss = loss.item()
            acc = (predicted == batch_y).sum().item() / batch_y.size(0)
            accuracies.append(acc)
                 
            # Evaluate the model on the validation set
            val_acc = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = self.nn(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    _, predicted = torch.max(outputs.data, 1)
                v_loss = loss.item()
                total = batch_y.size(0)
                correct = (predicted == batch_y).sum().item()
                val_acc += correct / total
                val_accuracies.append(acc)
                val_losses.append(v_loss)  
               
            print('Epoch [{}], Loss:{:.4f}, Validation Loss:{:.4f}, Accuracy:{:.2f}, Validation Accuracy:{:.2f}'.format(
                epoch+1, t_loss, v_loss, acc, val_acc))
            
            if(len(losses) != 0):
                if(t_loss > min(losses)):
                    c += 1
                else:
                    c = 0
            losses.append(t_loss)
            
            epoch += 1
            
            if(c == stop_earlier):
                print(f"Stopping earlier because no improvement in {stop_earlier} epochs")
                break
            elif(epoch == max_iter):
                break
        return epoch, losses, val_losses
    
    # Method to evaluate the model's accuracy
    def score(self, X_test, y_test, batch_size=32):
        # Set the model to evaluation mode
        self.nn.eval()
        
        correct = 0
        total = 0

        self.test = self.__set_dataset(X_test, y_test)
        
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

    def predict(self, X_test, y_test, batch_size=32):
        # Set the model to evaluation mode
        self.nn.eval()
        
        self.test = self.__set_dataset(X_test, y_test)
        
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
