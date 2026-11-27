import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime

from itertools import product
from preprocess import preprocess
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

torch.manual_seed(0)

def get_data(train_ds, valid_ds, batch_size, shuffle=True):
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

  # opt = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    opt = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    return (model, opt)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, patience = 10):
  train_losses = []
  valid_losses = []

  best_val_loss = float("inf")
  best_weights = None
  patience_counter = 0
  
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

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = {
            k: v.clone() 
            for k, v in model.state_dict().items()
        }
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

  # Restore best weights
  if best_weights is not None:
      model.load_state_dict(best_weights)

  return (valid_losses, train_losses)


data = pd.read_csv("data/fire_response_train.csv")
# Split training data 80/20
perm = torch.randperm(data.shape[0])
data = data.iloc[perm].reset_index(drop=True)

split = int(0.8 * data.shape[0])
data_train = data.iloc[:split]
data_valid = data.iloc[split:]

batch_size = 64
epochs = 30

loss_func = nn.MSELoss()

kf = KFold(
    n_splits=10,
    shuffle=True,
    random_state=0
)

# Hyperparameter tuning
param_grid = {
    "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2],
    "weight_decay": [0, 1e-5, 1e-4, 1e-3, 1e-2],
    "hidden_layers": [1, 2, 3, 4],
    "neurons_hidden": [16, 32, 64, 128, 256, 512],
    "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 
}

best_loss = float("inf")
best_params = None

for lr, wd, layers, neurons, dropout in product(
    param_grid["learning_rate"],
    param_grid["weight_decay"],
    param_grid["hidden_layers"],
    param_grid["neurons_hidden"],
    param_grid["dropout_rate"]
):

    fold_losses = []

    print(
    f"lr={lr}, "
    f"wd={wd}, "
    f"layers={layers}, "
    f"neurons={neurons}, "
    f"dropout={dropout}"
)

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(kf.split(data_train), 1):

        train_fold = data_train.iloc[train_idx]
        valid_fold = data_train.iloc[valid_idx]

        X_train, scaling_stats = preprocess(
            train_fold.drop(columns=["Response"])
        )
        X_valid, _ = preprocess(
            valid_fold.drop(columns=["Response"]),
            scaling_stats
        )
        X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)

        X_train = torch.from_numpy(X_train.to_numpy(dtype="float32"))
        X_valid = torch.from_numpy(X_valid.to_numpy(dtype="float32"))
        y_train = torch.from_numpy(train_fold[['Response']].to_numpy(dtype="float32"))
        y_valid = torch.from_numpy(valid_fold[['Response']].to_numpy(dtype="float32"))

        y_mean, y_std = y_train.mean(), y_train.std()
        y_train = (y_train - y_mean) / y_std
        y_valid = (y_valid - y_mean) / y_std


        train_ds = TensorDataset(X_train, y_train)
        valid_ds = TensorDataset(X_valid, y_valid)

        train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
        in_features = X_train.size(1)
        out_features = y_train.size(1)

        model, opt = get_model(
        in_features=in_features,
        out_features=out_features,
        learning_rate=lr,
        n_hidden_layers=layers,
        weight_decay=wd,
        neurons_hidden=neurons,
        dropout_rate=dropout)

        valid_losses, _ = fit(epochs, model, loss_func, opt, train_dl, valid_dl, patience = 10)

        fold_loss = min(valid_losses)

        fold_losses.append(
                    fold_loss
                )

        print(
                    f"Fold {fold}: "
                    f"{fold_loss:.6f}"
                )

    mean_loss = np.mean(
            fold_losses
        )

    std_loss = np.std(
            fold_losses
        )

    print(
            f"CV Loss = "
            f"{mean_loss:.6f} "
            f"± "
            f"{std_loss:.6f}"
        )

    if mean_loss < best_loss:

            best_loss = mean_loss

            best_params = {
                "learning_rate": lr,
                "weight_decay": wd,
                "hidden_layers": layers,
                "neurons_hidden": neurons,
                "dropout_rate": dropout
            }


print("\n================")
print("BEST PARAMETERS")
print("================")
print(best_params)

print(
    f"Best CV Loss: "
    f"{best_loss:.6f}"
)

X_train_final, train_scaling_stats = preprocess(
    data_train.drop(columns=["Response"])
)

X_train_final = torch.tensor(
    X_train_final.to_numpy(
        dtype=np.float32
    )
)

y_train_final = torch.tensor(
    data_train[["Response"]]
    .to_numpy(dtype=np.float32)
)

y_mean = y_train_final.mean()
y_std = y_train_final.std()

y_train_final = (
    y_train_final - y_mean
) / y_std

train_ds = TensorDataset(
    X_train_final,
    y_train_final
)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True
)

final_model, final_opt = get_model(
    in_features=X_train_final.shape[1],
    out_features=1,
    learning_rate=best_params[
        "learning_rate"
    ],
    n_hidden_layers=best_params[
        "hidden_layers"
    ],
    weight_decay=best_params[
        "weight_decay"
    ],
    neurons_hidden=best_params[
        "neurons_hidden"
    ],
    dropout_rate=best_params[
        "dropout_rate"
    ]
)

for epoch in range(epochs):

    final_model.train()

    for xb, yb in train_dl:

        loss_batch(
            final_model,
            loss_func,
            xb,
            yb,
            final_opt
        )

print("\nFinal model trained.")

#Save model and hyperparameters
timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

checkpoint = {
    "model_state_dict": final_model.state_dict(),
    "hyperparameters": best_params,
    "cv_loss": best_loss,
    "y_mean": y_mean.item(),
    "y_std": y_std.item(),
    "train_scaling_stats": train_scaling_stats
}

model_filename = (
    f"fire_response_model_{timestamp}.pth"
)

params_filename = (
    f"best_hyperparameters_{timestamp}.json"
)

torch.save(
    checkpoint,
    model_filename
)

with open(
    params_filename,
    "w"
) as f:
    json.dump(
        {
            "hyperparameters": best_params,
            "cv_loss": best_loss
        },
        f,
        indent=4
    )

print(
    f"\nModel saved to: {model_filename}"
)

print(
    f"Hyperparameters saved to: "
    f"{params_filename}"
)

# Evaluate best model

X_test, _ = preprocess(
    data_valid.drop(
        columns=["Response"]
    ),
    train_scaling_stats
)

X_test = X_test.reindex(
    columns=preprocess(
        data_train.drop(
            columns=["Response"]
        )
    )[0].columns,
    fill_value=0
)

X_test = torch.tensor(
    X_test.to_numpy(
        dtype=np.float32
    )
)

y_test = data_valid[
    "Response"
].values

final_model.eval()


with torch.no_grad():
    nn_pred = final_model(X_test)

nn_pred = nn_pred * y_std + y_mean
nn_pred = nn_pred.numpy().flatten()

print("\n=== Neural Network ===")
print("MSE:", mean_squared_error(data_valid["Response"], nn_pred))
print("R² :", r2_score(data_valid["Response"], nn_pred))

X_train_sk, scaling_stats = preprocess(
    data_train.drop(columns=["Response"])
)

X_valid_sk, _ = preprocess(
    data_valid.drop(columns=["Response"]),
    scaling_stats
)

X_valid_sk = X_valid_sk.reindex(
    columns=X_train_sk.columns,
    fill_value=0
)
# Benchmark regressions
# Linear Regression, Ridge and Lasso regression for comparison (no cross validation yet)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

for name, model in models.items():
    model.fit(X_train_sk, data_train["Response"])
    pred = model.predict(X_valid_sk)

    print(f"\n=== {name} ===")
    print("MSE:", mean_squared_error(data_valid["Response"], pred))
    print("R² :", r2_score(data_valid["Response"], pred))