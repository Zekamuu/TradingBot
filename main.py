import math
import random

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import requests
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import IPython


stocksTicker = "AAPL"
multiplier = 1
timespan = "day"
fromDate = "2024-01-01"
toDate = "2024-12-31"
key = "d8mn4ws669bjtEqDziawWgoobb5cjUO9"

trainModel = True

link = "https://api.polygon.io/v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{fromDate}/{toDate}?adjusted=true&sort=asc&apiKey={key}"


# Format the link with the actual values
formatted_link = link.format(
    stocksTicker=stocksTicker,
    multiplier=multiplier,
    timespan=timespan,
    fromDate=fromDate,
    toDate=toDate,
    key=key
)

# Fetch the data from the API
response = requests.get(formatted_link)
data = response.json()


# Lines 39 to 51 may be faulty
# Check if the API response contains 'results'
if 'results' not in data or not data['results']:
    raise ValueError("API response does not contain 'results' or it is empty")

# Convert the data to a DataFrame
df = pd.DataFrame(data['results'])

# Ensure required columns exist
required_columns = ['h', 'l', 'c']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is missing from the data")



# Calculate Pivot Points
for index, row in df.iterrows():
    df.at[index, 'pivot'] = (row['h'] + row['l'] + row['c']) / 3
    df.at[index, 'r1'] = df.at[index, 'pivot'] + (0.382 * (row['h'] - row['l']))
    df.at[index, 'r2'] = df.at[index, 'pivot'] + (0.618 * (row['h'] - row['l']))
    df.at[index, 'r3'] = df.at[index, 'pivot'] + (1.000 * (row['h'] - row['l']))
    df.at[index, 's1'] = df.at[index, 'pivot'] - (0.382 * (row['h'] - row['l']))
    df.at[index, 's2'] = df.at[index, 'pivot'] - (0.618 * (row['h'] - row['l']))
    df.at[index, 's3'] = df.at[index, 'pivot'] - (1.000 * (row['h'] - row['l']))

# Calculate EMA
df['ema_9'] = df['c'].ewm(span=9, adjust=False).mean()
df['ema_50'] = df['c'].ewm(span=50, adjust=False).mean()

# Calculate RSI
delta = df['c'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Calculate ATR
tr1 = df['h'] - df['l']
tr2 = abs(df['h'] - df['c'].shift(1))
tr3 = abs(df['l'] - df['c'].shift(1))
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['atr'] = tr.rolling(window=14).mean()

# Calculate Stochastic RSI
rsi14_min = df['rsi'].rolling(window=14).min()
rsi14_max = df['rsi'].rolling(window=14).max()
df['stoch_rsi'] = (df['rsi'] - rsi14_min) / (rsi14_max - rsi14_min)




# Save the DataFrame to a CSV file with the ticker symbol, timespan, and multiplier in the filename
csv_path = f"StockData/stock_data_{stocksTicker}_{timespan}_{multiplier}.csv"

# Check if the file already exists
if os.path.exists(csv_path):
    # Load the existing data
    existing_df = pd.read_csv(csv_path)
    # Append the new data
    df = pd.concat([existing_df, df]).drop_duplicates().reset_index(drop=True)

# Save the DataFrame to the CSV file
df.to_csv(csv_path, index=False)








#everything below is ML related
if(trainModel):

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display


    plt.ion()

    # Define model architecture
    class DQN(nn.Module):
        def __init__(self, n_observations, n_actions):
            super(DQN, self).__init__()
            self.layer1 = nn.Linear(n_observations, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, n_actions)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)

    # Hyperparameters
    BATCH_SIZE = 256
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    n_actions = 3  # Long, Short, Do Nothing
    n_observations = df.shape[1]

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = deque(maxlen=10000)
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    steps_done = 0

    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = random.sample(memory, BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    # Training loop
    num_episodes = 600 if torch.cuda.is_available() else 50

    # Add this function to plot the training graph
    def plot_durations(show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.clf()
        plt.title('Training...' if not show_result else 'Result')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take a 100-episode average and plot it
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # Pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    # Initialize a list to store episode durations
    episode_durations = []

    # Update the training loop to track durations
    for i_episode in range(num_episodes):
        state = torch.tensor(df.iloc[0].values, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            if t + 1 < len(df):
                next_state = torch.tensor(df.iloc[t + 1].values, dtype=torch.float32, device=device).unsqueeze(0)
                reward = torch.tensor([df.iloc[t + 1]['c'] - df.iloc[t]['c']], device=device, dtype=torch.float32)
            else:
                next_state = None
                reward = torch.tensor([0], device=device, dtype=torch.float32)

            memory.append(Transition(state, action, next_state, reward))

            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if next_state is None:
                episode_durations.append(t + 1)  # Track the duration of the episode
                plot_durations()  # Update the plot after each episode
                break

    print('Training complete')
    plot_durations(show_result=True)  # Show the final result
    plt.ioff()
    plt.show()


    # Download training data from open datasets.
        # training_data = datasets.FashionMNIST(
        #     root="data",
        #     train=True,
        #     download=True,
        #     transform=ToTensor(),
        # )

    # stock_data = pd.read_csv(csv_path)
    # stock_tensor = torch.tensor(stock_data.values, dtype=torch.float32)
    # testing_tensor = torch.tensor(stock_data.iloc[-1].values, dtype=torch.float32)
    

    # Download test data from open datasets.
        # test_data = datasets.FashionMNIST(
        #     root="data",
        #     train=False,
        #     download=True,
        #     transform=ToTensor(),
        # )
    

    # batch_size = len(stock_data)-1 #So that the model can predict the very last price

    # Create data loaders.
    # train_dataloader = DataLoader(stock_tensor, batch_size=batch_size)
    # test_dataloader = DataLoader(testing_tensor, batch_size=1)

    # for X, y in test_dataloader:
    #     print(f"Shape of X [N, C, H, W]: {X.shape}")
    #     print(f"Shape of y: {y.shape} {y.dtype}")
    #     break


    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(f"Using {device} device")

    # Define model
    # class NeuralNetwork(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #                                     # Fix this stuff
    #         # self.flatten = nn.Flatten()
    #         self.linear_relu_stack = nn.Sequential(
    #             nn.Linear(input_size, 512),
    #             nn.ReLU(),
    #             nn.Linear(512, 512),
    #             nn.ReLU(),
    #             nn.Linear(512, 3)
    #         )

    #     def forward(self, x):
    #         # x = self.flatten(x)
    #         logits = self.linear_relu_stack(x)
    #         return logits

    # model = NeuralNetwork().to(device)
    # print(model)




    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)




    # def train(dataloader, model, loss_fn, optimizer):
    #     size = len(dataloader.dataset)
    #     model.train()
    #     for batch, (X, y) in enumerate(dataloader):
    #         X, y = X.to(device), y.to(device)

    #         # Compute prediction error
    #         pred = model(X)
    #         loss = loss_fn(pred, y)

    #         # Backpropagation
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()

    #         if batch % 100 == 0:
    #             loss, current = loss.item(), (batch + 1) * len(X)
    #             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




    # def test(dataloader, model, loss_fn):
    #     size = len(dataloader.dataset)
    #     num_batches = len(dataloader)
    #     model.eval()
    #     test_loss, correct = 0, 0
    #     with torch.no_grad():
    #         for X, y in dataloader:
    #             X, y = X.to(device), y.to(device)
    #             pred = model(X)
    #             test_loss += loss_fn(pred, y).item()
    #             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    #     test_loss /= num_batches
    #     correct /= size
    #     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




    # epochs = 5
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer)
    #     test(test_dataloader, model, loss_fn)
    # print("Done!")

    # torch.save(model.state_dict(), "model.pth")
    # print("Saved PyTorch Model State to model.pth")

    # model = NeuralNetwork().to(device)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))

    # classes = [
    #     "Long",
    #     "Short",
    #     "Do Nothing",
    # ]

    # model.eval()
    # x, y = test_data[0][0], test_data[0][1]
    # with torch.no_grad():
    #     x = x.to(device)
    #     pred = model(x)
    #     predicted, actual = classes[pred[0].argmax(0)], classes[y]
    #     print(f'Predicted: "{predicted}", Actual: "{actual}"')

