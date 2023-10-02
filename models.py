import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

class AlphabetModel():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.input_size = len(X_train.columns)
        self.hidden_size = len(X_train.columns)*2 + 3
        self.output_size = len(set(y_train))
        
        self.X_train = torch.FloatTensor(X_train.to_numpy())
        self.y_train = torch.LongTensor(LabelEncoder().fit_transform(y_train.to_numpy()))
        
        self.train = TensorDataset(self.X_train, self.y_train)
        
        self.X_test = torch.FloatTensor(X_test.to_numpy())
        self.y_test = torch.LongTensor(LabelEncoder().fit_transform(y_test.to_numpy()))
        
        self.test = TensorDataset(self.X_test, self.y_test)
        
        self.nn = NeuralNet(self.input_size, self.hidden_size, self.output_size)
    
        
    def fit(self, num_epoch=100, batch_size=64):
        criterion = nn.CrossEntropyLoss()  # Mean Squared Error loss (you can choose an appropriate loss function)
        optimizer = optim.Adam(self.nn.parameters(), lr=0.001)  # Adam optimizer
        
        train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=True)
        
        losses = []
        for epoch in range(num_epoch):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.nn(batch_x)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')

        return range(num_epoch), losses
    
    def score(self, batch_size=512):
        # Evaluation
        self.nn.eval()
        correct = 0
        total = 0

        test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=True)
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.nn(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
                
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.softmax(out)
        
        # out = torch.argmax(out, dim=1)
        return out
    