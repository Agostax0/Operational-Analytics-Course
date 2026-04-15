import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # if using GPU
    torch.backends.cudnn.deterministic = True  # deterministic ops
    torch.backends.cudnn.benchmark = False     # disable auto-tuner


def create_dataset(data, seq_len):
    xs, ys = [], []
    for i in range(len(data)- seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)


# Model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=24, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
        batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

if __name__ == "__main__":
    # Load dataset
    # df = pd.read_csv("BoxJenkins.csv", usecols=[1])
    # data = df.values.astype(float)

    df = pd.read_csv('M3C_monthly.csv')
    rawdata_x = np.arange(len(df.iloc[505, 6:].values))
    rawdata_y = df.iloc[505, 6:].values.astype(float)
    rawdata = pd.DataFrame(rawdata_y.transpose(), rawdata_x.transpose())
    df = rawdata
    data = rawdata

    # Normalize, needed
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    lag = 12
    X, y = create_dataset(data_scaled, lag)
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    set_seed(666)
    model = LSTM(num_layers=2, hidden_size=48)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Training
    for epoch in range(100):
        model.train()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Forecast next 12 values
    model.eval()
    fore = []
    # unsqueeze adds a dimension, it is a pytorch unfortunate requirement
    seq = torch.tensor(X[-1], dtype=torch.float32).unsqueeze(0)
    for _ in range(12):
        with torch.no_grad():
            pred = model(seq)
            fore.append(pred.item())
            seq = torch.cat([seq[:, 1:], pred.unsqueeze(1)], dim=1)
    # Inverse scale
    forecast = scaler.inverse_transform(np.array(fore).reshape(-1, 1))
    # Plot

    plt.plot(df.values, label="Actual")
    plt.plot(range(len(df) - 12 - 1, len(df) - 1), forecast, label="Forecast", color='red')
    plt.legend()
    plt.title("BoxJenkins LSTM series")
    plt.show(block=True)  # block ensures it writes the window properly, it waits for it
    print("fine")