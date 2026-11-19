# TODO : Implement dropout -> nn.Dropout()
# TODO : Implement regularization
# TODO : Option to use ADAM instead of SGD
# TODO : Hyperparameter tuning on GPU
# TODO : Save the model (containing the model parameters)
# TODO : Add sklearn for regressions

import torch
import numpy as np

from preprocess import train
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

def get_data(train_ds, valid_ds, batch_size, shuffle=False):
  return (
    DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle),
    DataLoader(valid_ds, batch_size=batch_size*2)
  )

def get_model(in_features, out_features, learning_rate, n_hidden_layers, neurons_hidden = None):
  # Use rule of thumb for number of hidden neurons if no argument is provided
  if neurons_hidden == None:
    neurons_hidden = int((in_features + out_features) / 2)

  model = nn.Sequential(
    nn.Linear(in_features, neurons_hidden),
    nn.ReLU()
  )
  hidden_layer = nn.Sequential(nn.Linear(neurons_hidden, neurons_hidden), nn.ReLU())
  for _ in range (n_hidden_layers):
    model.extend(hidden_layer)
  model.extend(nn.Sequential(nn.Linear(neurons_hidden, out_features)))

  opt = optim.SGD(model.parameters(), lr = learning_rate)

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
    print(epoch, val_loss)
    valid_losses.append(val_loss)
  return (valid_losses, train_losses)

X = torch.from_numpy(train.drop(columns=['Response']).to_numpy(dtype="float32"))
y = torch.from_numpy(train[['Response']].to_numpy(dtype="float32"))

# Split training data 80/20
gen = torch.Generator()
gen.manual_seed(0)
perm = torch.randperm(X.size(0), generator=gen)
X, y = X[perm], y[perm]

split = int(X.size(0) * 0.8)
X_train = X[:split,]
X_valid = X[split:,]
y_train = y[:split,]
y_valid = y[split:,]

y_mean, y_std = y_train.mean(), y_train.std()
y_train = (y_train - y_mean) / y_std
y_valid = (y_valid - y_mean) / y_std

loss_func = nn.MSELoss(reduction='mean')

# Model hyper parameters
hidden_layers = 3
learning_rate = 1e-3
batch_size = 64
epochs = 100

train_ds = TensorDataset(X_train, y_train)
valid_ds = TensorDataset(X_valid, y_valid)

train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
in_features = X.size(1)
out_features = y.size(1)

model, opt = get_model(in_features, out_features, learning_rate, hidden_layers)
valid_losses, train_losses = fit(epochs, model, loss_func, opt, train_dl, valid_dl)
