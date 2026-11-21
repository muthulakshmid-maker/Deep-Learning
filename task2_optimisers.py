import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

print("Libraries imported successfully!")


X, y = make_moons(
    n_samples=1000,
    noise=0.2,
    random_state=42
)

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

print("Training samples:", X_train.shape)
print("Validation samples:", X_val.shape)

print("Dataset prepared successfully!")



class MoonNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(2, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.network(x)


model = MoonNetwork()

print("\n========== NEURAL NETWORK ==========")
print(model)



sample_input = X_train[:5]

output = model(sample_input)

print("\nInput shape:", sample_input.shape)
print("Output shape:", output.shape)

print("\nForward pass completed successfully!")



from torch.utils.data import TensorDataset, DataLoader
import copy

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

criterion = nn.CrossEntropyLoss()


def train_model(model, optimizer, epochs=100):

    losses = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for inputs, labels in train_loader:

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)

        losses.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/100] "
                f"Loss: {epoch_loss:.4f}"
            )

    return losses



print("\n========== SGD OPTIMISER ==========")

base_model = MoonNetwork()
initial_weights = copy.deepcopy(base_model.state_dict())

sgd_model = MoonNetwork()
sgd_model.load_state_dict(initial_weights)

sgd_optimizer = optim.SGD(
    sgd_model.parameters(),
    lr=0.01
)

sgd_losses = train_model(
    sgd_model,
    sgd_optimizer,
    epochs=100
)


print("\n========== SGD WITH MOMENTUM ==========")

momentum_model = MoonNetwork()
momentum_model.load_state_dict(initial_weights)

momentum_optimizer = optim.SGD(
    momentum_model.parameters(),
    lr=0.01,
    momentum=0.9
)

momentum_losses = train_model(
    momentum_model,
    momentum_optimizer,
    epochs=100
)


print("\n========== ADAM OPTIMISER ==========")

adam_model = MoonNetwork()
adam_model.load_state_dict(initial_weights)

adam_optimizer = optim.Adam(
    adam_model.parameters(),
    lr=0.01
)

adam_losses = train_model(
    adam_model,
    adam_optimizer,
    epochs=100
)


print("\nAll three optimisers trained successfully!")


plt.figure(figsize=(10, 6))

plt.plot(sgd_losses, label="SGD")
plt.plot(momentum_losses, label="SGD with Momentum")
plt.plot(adam_losses, label="Adam")

plt.xlabel("Epochs")
plt.ylabel("Training Loss")

plt.title("Optimiser Comparison")

plt.legend()
plt.grid(True)

from pathlib import Path

file_path = Path(__file__).resolve().parent / "task2_optimiser_comparison.png"

plt.savefig(
    file_path,
    dpi=300,
    bbox_inches="tight"
)

print("Graph saved at:", file_path)

plt.show()

print("Optimiser comparison graph saved successfully!")


class RegularizedMoonNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(2, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.network(x)


print("\nRegularized network:")
regularized_model = RegularizedMoonNetwork()
print(regularized_model)


def calculate_validation_loss(model):

    model.eval()

    with torch.no_grad():

        outputs = model(X_val)

        loss = criterion(outputs, y_val)

    return loss.item()


print("\n========== VALIDATION LOSS COMPARISON ==========")

print("SGD Validation Loss:",
      round(calculate_validation_loss(sgd_model), 4))

print("Momentum Validation Loss:",
      round(calculate_validation_loss(momentum_model), 4))

print("Adam Validation Loss:",
      round(calculate_validation_loss(adam_model), 4))


regularized_model = RegularizedMoonNetwork()

regularized_optimizer = optim.Adam(
    regularized_model.parameters(),
    lr=0.01
)

regularized_losses = train_model(
    regularized_model,
    regularized_optimizer,
    epochs=100
)

regularized_val_loss = calculate_validation_loss(
    regularized_model
)

print("\nRegularized Validation Loss:",
      round(regularized_val_loss, 4))


def train_with_early_stopping(model, optimizer, epochs=100, patience=10):

    best_val_loss = float("inf")
    best_weights = None
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_dataset)
        train_losses.append(train_loss)

        val_loss = calculate_validation_loss(model)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    print("Best validation loss:", round(best_val_loss, 4))

    return train_losses, val_losses


early_stop_model = RegularizedMoonNetwork()

early_stop_optimizer = optim.Adam(
    early_stop_model.parameters(),
    lr=0.01
)

early_train_losses, early_val_losses = train_with_early_stopping(
    early_stop_model,
    early_stop_optimizer,
    epochs=100,
    patience=10
)

plt.figure(figsize=(10, 6))

plt.plot(early_train_losses, label="Training Loss")
plt.plot(early_val_losses, label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Early Stopping - Training vs Validation Loss")
plt.legend()
plt.grid(True)

from pathlib import Path

file_path = Path(__file__).resolve().parent / "task2_early_stopping.png"

plt.savefig(file_path, dpi=300, bbox_inches="tight")
print("Graph saved at:", file_path)

plt.show()





# BONUS TASK:

scheduler_model = MoonNetwork()

scheduler_optimizer = optim.Adam(
    scheduler_model.parameters(),
    lr=0.01
)

scheduler = optim.lr_scheduler.StepLR(
    scheduler_optimizer,
    step_size=20,
    gamma=0.5
)

scheduler_losses = []
learning_rates = []

for epoch in range(100):

    scheduler_model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:

        outputs = scheduler_model(inputs)
        loss = criterion(outputs, labels)

        scheduler_optimizer.zero_grad()
        loss.backward()
        scheduler_optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    scheduler_losses.append(epoch_loss)

    current_lr = scheduler_optimizer.param_groups[0]["lr"]
    learning_rates.append(current_lr)

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"Loss = {epoch_loss:.4f}, "
            f"Learning Rate = {current_lr:.6f}"
        )


from pathlib import Path

plt.figure(figsize=(10, 6))

plt.plot(range(1, 101), learning_rates, marker="o")

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("StepLR Learning Rate Schedule")
plt.grid(True)

file_path = Path(__file__).resolve().parent / "task2_stepLR.png"

plt.savefig(file_path, dpi=300, bbox_inches="tight")

print("StepLR graph saved at:", file_path)

plt.show()
