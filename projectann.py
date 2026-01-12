import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1) Load dataset (put Churn_Modelling.csv in same folder)
data = pd.read_csv("Churn_Modelling.csv")

# 2) Drop unused columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# 3) Encode Gender
gender_encoder = LabelEncoder()
data["Gender"] = gender_encoder.fit_transform(data["Gender"])

# 4) One-hot encode Geography
geo_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
geo_ohe = geo_encoder.fit_transform(data[["Geography"]])
geo_cols = geo_encoder.get_feature_names_out(["Geography"])
geo_df = pd.DataFrame(geo_ohe, columns=geo_cols)

# 5) Combine + drop original Geography
data = pd.concat([data.drop(["Geography"], axis=1), geo_df], axis=1)

# 6) Split
X = data.drop("Exited", axis=1)
y = data["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7) Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8) Build model
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
    callbacks=[early_stop],
    verbose=1
)

# 9) Save artifacts
model.save("churn_model.h5")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("gender_encoder.pkl", "wb") as f:
    pickle.dump(gender_encoder, f)

with open("geo_encoder.pkl", "wb") as f:
    pickle.dump(geo_encoder, f)

# IMPORTANT: save feature column order
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Saved: churn_model.h5, scaler.pkl, gender_encoder.pkl, geo_encoder.pkl, feature_columns.pkl")
