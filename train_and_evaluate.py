import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# --------------------------
# 1. Load Filtered Data
# --------------------------
features_path = r"C:\Users\PMYLS\How-to-download-ABIDE-Preprocessed-dataset-for-autism-detection\preprocessed_dataset\cnn_features_filtered.pt"
labels_path   = r"C:\Users\PMYLS\How-to-download-ABIDE-Preprocessed-dataset-for-autism-detection\preprocessed_dataset\labels_filtered.pt"

features = torch.load(features_path)    # Shape: [297, 32, 56, 56]
labels   = torch.load(labels_path)        # Shape: [297]

print("Filtered Features Shape:", features.shape)
print("Filtered Labels Shape:", labels.shape)

# Create the full dataset
dataset = TensorDataset(features, labels)

# --------------------------
# 2. Split the Data (80% train, 20% test)
# --------------------------
total_samples = len(dataset)        # 297
train_size = int(0.8 * total_samples) # 80% for training
test_size = total_samples - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

# Create DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# --------------------------
# 3. Define a Simple Model
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # reduces H and W by half
        self.fc1 = nn.Linear(64 * 28 * 28, 2)  # 2 outputs for binary classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        return x

model = SimpleCNN()
print("Model initialized!")

# --------------------------
# 4. Train the Model on the Training Set
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")

print("Training completed!")

# Save the trained model
model_save_path = r"C:\Users\PMYLS\How-to-download-ABIDE-Preprocessed-dataset-for-autism-detection\preprocessed_dataset\vit_model.pt"
torch.save(model.state_dict(), model_save_path)
print("Model saved successfully!")

# --------------------------
# 5. Evaluate the Model on the Test Set
# --------------------------
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")
