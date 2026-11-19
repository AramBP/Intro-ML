# TODO : seperate train and test
import pandas as pd
import numpy as np

# Preprocess and return the means and std values that where used to standardize
def preprocess(data, scaling_stats = None):
  features = data
  # Standardize numerical features except: call_month, call_hour
  numeric_features = features.dtypes[features.dtypes!='string'].index
  numeric_features = numeric_features.drop(["call_month", "call_hour"])

  # Log transform because prior incidents is heavily skewed
  prior_incidents_cols = [c for c in features.columns if c.startswith("prior_incidents_")]
  features[prior_incidents_cols] = features[prior_incidents_cols].apply(lambda x : np.log1p(x))

  # Standardize numerical features
  if scaling_stats == None:
    scaling_stats = {
      col: {
          "mean": features[col].mean(),
          "std": features[col].std()
      }
      for col in numeric_features
    }

  features[numeric_features] = features[numeric_features].apply(
      lambda x: (x - scaling_stats[x.name]["mean"]) / scaling_stats[x.name]["std"]
  )

  # Encode call_month and call_hour and call_day as sine/cosine pairs. This helps represening the wrap around of time
  # That is, hour 23 is close to hour 0, december is close to january, sunday is close to monday

  def day_to_num(s):
    if s == "Monday": return 0
    elif s == "Tuesday": return 1
    elif s == "Wednesday": return 2
    elif s == "Thursday": return 3
    elif s == "Friday": return 4
    elif s == "Saturday": return 5
    elif s == "Sunday": return 6
  features["call_weekday"] = features["call_weekday"].apply(lambda x : day_to_num(x))

  features["call_weekday_sin"] = np.sin(2 * np.pi * features["call_weekday"] / 7)
  features["call_weekday_cos"] = np.cos(2 * np.pi * features["call_weekday"] / 7)
  features["call_month_sin"] = np.sin(2 * np.pi * (features["call_month"]- 1) / 12)
  features["call_month_cos"] = np.cos(2 * np.pi * (features["call_month"]- 1) / 12)
  features["call_hour_sin"] = np.sin(2 * np.pi * features["call_hour"] / 24)
  features["call_hour_cos"] = np.cos(2 * np.pi * features["call_hour"] / 24)
  features = features.drop(columns=["call_weekday", "call_hour", "call_month"])
  # One hot encoding
  features = pd.get_dummies(features, dummy_na=True)
  return features.copy(), scaling_stats
