import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import root_mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
if __name__ == '__main__':
    # load the dataset, split into input (X) and output (y) variables
    df = pd.read_csv('M3C_monthly.csv')
    rawdata_x = np.arange(len(df.iloc[490, 6:].values))
    rawdata_y = df.iloc[490, 6:].values.astype(float)
    rawdata = pd.DataFrame(rawdata_y.transpose(), rawdata_x.transpose())
    print(f"df2: {rawdata.shape}")


    #df2 = pd.read_csv('BoxJenkins.csv', usecols=[1])
    #print(f"df2: {df2.shape}")
    #print(df2)
    #dataset = df2.values.astype(float) # COLUMN VECTOR !!!
    dataset = rawdata.values.astype(float)
    # split into train and test sets
    train_size = int(len(dataset) - 12)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print("Len train={0}, len test={1}".format(len(train), len(test)))
    # reshape into X=t and Y=t+1
    look_back = 12
    testdata = np.concatenate((train[-look_back:], test))
    trainX, trainy = create_dataset(train, look_back)

    X = torch.FloatTensor(trainX)
    y = torch.FloatTensor(trainy)
    # define the model (2 hidden layers, 1 output neuron)
    model = nn.Sequential(nn.Linear(12, 10), nn.ReLU(), nn.Linear(10, 8), nn.ReLU(),
                          nn.Linear(8, 1)  # no activation function, allowing for negative outputs
                          )
    print(model)
    # train the model
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # mind the learning rate!
    print(f"Optimizer type: {type(optimizer)}")
    n_epochs = 200
    batch_size = 1
    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):  # sometimes batches are of help
            Xbatch = X[i:i + batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i + batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()  # clear old gradients
            loss.backward()  # here the learning
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

    #alternative1 : model evaluation as RMSE
    model.eval()  # tells the model we are evaluating, not training
    with torch.no_grad():  # no_grad: needed to stop tracking gradient computations
        y_pred = model(X)  # y_pred is a torch tensor
    rmse = root_mean_squared_error(trainy, y_pred.numpy())
    print(f"Test RMSE: {rmse:.4f}")

    # alternative 2: model evaluation as AIC
    # Count number of parameters (k)
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Get predictions on training data
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        rss = torch.sum((y - y_pred) ** 2).item()  # both tensors
        n = trainX.shape[0]
        aic = n * np.log(rss / n) + 2 * k
        print(f"AIC: {aic:.2f}")


    # generate predictions for training and forecast for plotting
    trainPredict = model(X)
    predictions = trainPredict.detach().cpu().numpy()

    # Recursive forecasting for testY.Starting from last window from train ( or first from testX)
    input_seq = trainX[-1].tolist()
    testForecast = []
    model.eval()
    with torch.no_grad():
        for _ in range(len(test)):
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # shape [1,look_back]
            pred = model(input_tensor).item()  # get scalar prediction
            testForecast.append(pred)
            # Roll the input window forward by 1 step
            input_seq = input_seq[1:] + [pred]
    print(testForecast)
    # plot train/test data and predictions
    plt.plot(dataset, label="actual")
    plt.plot(range(12, len(predictions) + 12), predictions, label="predict")
    plt.plot(range(len(predictions) + 12, len(predictions) + 24), testForecast, label="forecast")
    plt.legend()
    plt.show(block=True)