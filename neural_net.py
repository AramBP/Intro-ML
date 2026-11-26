# TODO : Implement dropout -> nn.Dropout() (done)
# TODO : Implement regularization (done)
# TODO : Option to use ADAM instead of SGD (done)
# TODO : Hyperparameter tuning on GPU (mee bezig)
# TODO : Save the model (containing the model parameters)
# TODO : Add sklearn for regressions (done)

import torch
import numpy as np
import pandas as pd

from itertools import product
from preprocess import preprocess
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

torch.manual_seed(0)

def get_data(train_ds, valid_ds, batch_size, shuffle=False):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle),
        DataLoader(valid_ds, batch_size=batch_size * 2)
    )

def get_model(in_features, out_features, learning_rate, n_hidden_layers, weight_decay, neurons_hidden=None, dropout_rate=0.2):

    if neurons_hidden is None:
        neurons_hidden = int((in_features + out_features) / 2)

    layers = [nn.Linear(in_features, neurons_hidden), nn.ReLU(), nn.Dropout(dropout_rate)]

    for _ in range(n_hidden_layers):
        layers.extend([nn.Linear(neurons_hidden, neurons_hidden), nn.ReLU(), nn.Dropout(dropout_rate)])

    layers.append(nn.Linear(neurons_hidden, out_features))

    model = nn.Sequential(*layers)

  # opt = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return (model, opt)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
  train_losses = []
  valid_losses = []
  for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
      loss, _ = loss_batch(model, loss_func, xb, yb, opt)
      train_losses.append(loss)

    model.eval()
    with torch.no_grad():
      losses, nums = zip(
        *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
      )
    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    valid_losses.append(val_loss)
  return (valid_losses, train_losses)


data = pd.read_csv("data/fire_response_train.csv")
# Split training data 80/20
perm = torch.randperm(data.shape[0])
data = data.iloc[perm].reset_index(drop=True)

split = int(0.8 * data.shape[0])
data_train = data.iloc[:split]
data_valid = data.iloc[split:]

X_train, train_scaling_stats = preprocess(
    data_train.drop(columns=["Response"])
)
X_valid, _ = preprocess(
    data_valid.drop(columns=["Response"]),
    train_scaling_stats
)
X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)

X_train = torch.from_numpy(X_train.to_numpy(dtype="float32"))
X_valid = torch.from_numpy(X_valid.to_numpy(dtype="float32"))
y_train = torch.from_numpy(data_train[['Response']].to_numpy(dtype="float32"))
y_valid = torch.from_numpy(data_valid[['Response']].to_numpy(dtype="float32"))

y_mean, y_std = y_train.mean(), y_train.std()
y_train = (y_train - y_mean) / y_std
y_valid = (y_valid - y_mean) / y_std

# Model hyper parameters
batch_size = 64
epochs = 30

train_ds = TensorDataset(X_train, y_train)
valid_ds = TensorDataset(X_valid, y_valid)

train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
in_features = X_train.size(1)
out_features = y_train.size(1)

loss_func = nn.MSELoss()

# Hyperparameter tuning
param_grid = {
    "learning_rate": [1e-4, 1e-3, 1e-2],
    "weight_decay": [0, 1e-4, 1e-3],
    "hidden_layers": [1, 2, 3],
    "neurons_hidden": [32, 64, 128],
    "dropout_rate": [0.0, 0.2, 0.4]
}


best_loss = float("inf")
best_params = None
best_model = None

for lr, wd, layers, neurons, dropout in product(
    param_grid["learning_rate"],
    param_grid["weight_decay"],
    param_grid["hidden_layers"],
    param_grid["neurons_hidden"],
    param_grid["dropout_rate"]
):

    print(f"Testing lr={lr}, "f"wd={wd}, "f"layers={layers}")

    model, opt = get_model(
        in_features=in_features,
        out_features=out_features,
        learning_rate=lr,
        n_hidden_layers=layers,
        weight_decay=wd,
        neurons_hidden=neurons,
        dropout_rate=dropout)

    valid_losses, _ = fit(epochs, model, loss_func, opt, train_dl, valid_dl)

    final_val_loss = valid_losses[-1]
    print(f"lr={lr}, wd={wd}, layers={layers}, "f"val_loss={final_val_loss:.6f}")

    if final_val_loss < best_loss:
        best_loss = final_val_loss
        best_params = {"learning_rate": lr, "weight_decay": wd, "hidden_layers": layers, "neurons_hidden": neurons, "dropout_rate": dropout}
        best_model = model

print("\nBest parameters:")
print(best_params)

print("\nBest validation loss:")
print(best_loss)

# Evaluate best model
model = best_model
model.eval()

with torch.no_grad():
    nn_pred = model(X_valid)

nn_pred = nn_pred * y_std + y_mean
nn_pred = nn_pred.numpy().flatten()

print("\n=== Neural Network ===")
print("MSE:", mean_squared_error(data_valid["Response"], nn_pred))
print("R² :", r2_score(data_valid["Response"], nn_pred))

# Benchmark regressions
# Linear Regression, Ridge and Lasso regression for comparison (no cross validation yet)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

for name, model in models.items():
    model.fit(X_train, data_train["Response"])
    pred = model.predict(X_valid)

    print(f"\n=== {name} ===")
    print("MSE:", mean_squared_error(data_valid["Response"], pred))
    print("R² :", r2_score(data_valid["Response"], pred))