import time

import numpy as np
import torch 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


TRAIN_SIZE = 10000
epochs = 3
lr = 1e-3
batch_size = 4
num_epochs = 3
data_dir = "./pytorch_data"

torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.device.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081)),]
)

train_dataset = datasets.MNIST(
    root=data_dir, train=True, transform=transform, download=True
)

test_dataset = datasets.MNIST(
    root=data_dir, train=False, transform=transform, download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Pre-allocate tensors of the appropriate size
train_data = torch.zeros(len(train_dataset), 1, 28, 28)
train_labels = torch.zeros(len(train_dataset), dtype=torch.long)
test_data = torch.zeros(len(test_dataset), 1, 28, 28)
test_labels = torch.zeros(len(test_dataset), dtype=torch.long)

# Load data into RAM
for idx, (data, label) in enumerate(train_loader):
    start_idx = idx * batch_size
    end_idx = start_idx + data.size(0)
    train_data[start_idx:end_idx] = data
    train_labels[start_idx:end_idx] = label

print("Train Data Shape:", train_data.shape)
print("Train Data Type:", train_data.dtype)


# Load all test data into RAM
for idx, (data, label) in enumerate(test_loader):
    start_idx = idx * batch_size
    end_idx = start_idx + data.size(0)
    test_data[start_idx:end_idx] = data
    test_labels[start_idx:end_idx] = label

print("Train Data Shape:", test_data.shape)
print("Train Data Type:", test_data.dtype)

iters_per_epoch = TRAIN_SIZE // batch_size


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = x.reshape(batch_size, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

model = MLP(in_features=784, hidden_features=256, num_classes=10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

#train job
def train(model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for i in range(iters_per_epoch):
        optimizer.zero_grad()
        data = train_data[i*batch_size:(i+1)*batch_size].to(device)
        target = train_labels[i*batch_size: (i+1)*batch_size].to(device)

        start = time.time()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        end = time.time()
        running_loss += loss.item()
        if 1%100 == 99 or i ==0:
            print(f"Epoch: {epoch+1}, Iter: {i+1}, Loss: {loss}")
            print(f"Iteration Time: {(end - start) * 1e3:.4f} sec")
            running_loss = 0.0



#Evaluate the function
def evaluate(model, test_data, test_labels):
    model.to(device)
    model.eval()

    total_batch_accuracy = torch.tensor(0.0, device=device)
    num_batches = 0

    with torch.no_grad():
        for i in range(len(test_data)//batch_size):
            data = test_data[i*batch_size:(i+1)*batch_size].to(device)
            labels = test_labels[i*batch_size:(i+1)*batch_size].to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            correct_batch = (predicted == labels).sum().item()
            total_batch = labels.size(0)
            if total_batch != 0:
                batch_accuracy = correct_batch / total_batch
                total_batch_accuracy += batch_accuracy
                num_batches += 1


    avg_batch_accuracy = total_batch_accuracy / num_batches
    print(f"Average Batch Accuracy:{avg_batch_accuracy*100:.2f}%")