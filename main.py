from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from stable_baselines3 import A2C
import torch
from torch import nn, optim, tensor as T
from collections import deque
import random

def process_obs(obs):
    grey = obs.dot([0.07, 0.72, 0.21])
    #Top left: (94, 45)
    #Bottom right: (177, 210)
    cropped = grey[47:207, 95:175]
    downsampled = np.array(Image.fromarray(cropped).resize((80, 80), resample=Image.NEAREST))
    return downsampled

class ExperienceReplay:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, item):
        #item is (s, a, r, sp)
        self.memory.append(item)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QModel(nn.Module):
    def __init__(self, K, memory, lr=1e-2, gamma=0.99):
        super(QModel, self).__init__()
        
        self.lr = lr
        self.gamma = gamma
        self.K = K
        
        self.layers = nn.ModuleList()
        c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2) #80x80x6
        p2 = nn.MaxPool2d(4, 4) #20x20x6
        c3 = nn.Conv2d(6, 16, kernel_size=5) #16x16x16
        p4 = nn.MaxPool2d(2, 2) #8x8x16
        fl = nn.Flatten()
        fc5 = nn.Linear(1024, 256)
        fc6 = nn.Linear(256, 84)
        fc7 = nn.Linear(84, K)
        self.layers.extend([c1, p2, c3, p4, fl, fc5, fc6, fc7])
        
        self.g = nn.ReLU()
        self.memory = memory
        self.op = optim.Adam(self.parameters(), lr=lr)
        self.cost = nn.SmoothL1Loss()

    def forward(self, X):
        a = X.view(-1, 1, 80, 80)
        for l in self.layers:
            z = l(a)
            a = self.g(z)
        return a

    def train(self, target_network, batch_size=128):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = tuple(zip(*transitions)) #turns list of tuples into tuple of lists
        #print(batch[1])
        non_term_mask = T(tuple(map(lambda s: s is not None, batch[3])), dtype=torch.bool)
        non_term_sps = torch.cat([s for s in batch[3] if s is not None])

        s_batch = torch.stack(batch[0])
        a_batch = torch.stack(batch[1])
        r_batch = torch.stack(batch[2])


        yhat = self.forward(s_batch)
        Qs = yhat.gather(1, a_batch.view(-1, 1))

        next_Qs = torch.zeros(batch_size)
        next_Qs[non_term_mask] = target_network(non_term_sps).max(1)[0].detach()

        expected_Qs = r_batch + self.gamma*next_Qs

        l = self.cost(Qs, expected_Qs.unsqueeze(1))
        self.op.zero_grad()
        l.backward()
        self.op.step()

    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            X = T(x)
            return np.argmax(self.forward(X.float()).detach().numpy())
        

def play_one(env, model, tmodel, eps, copy_period):
    global tot_iters
    done = False
    obs = process_obs(env.reset())
    tot_r = 0
    iters = 0

    while not done:
        a = model.sample_action(obs, eps)
        prev_obs = obs
        raw_obs, r, done, _ = env.step(a)
        obs = process_obs(raw_obs)
        
        model.memory.push((T(prev_obs), T(a), T(r), T(obs)))

        model.train(tmodel)

        env.render()
        tot_r += r
        iters += 1
        tot_iters += 1
        if tot_iters % copy_period == 0:
            tmodel.load_state_dict(model.state_dict())

    return tot_r

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#model = A2C('MlpPolicy', env, verbose=1)
#model.learn(total_timesteps=2e5)


#print(obs.shape)
#plt.imshow(obs, cmap='gray')
#plt.show()

memory = ExperienceReplay(10000)
copy_period = 100
model = QModel(env.action_space.n, memory)
tmodel = QModel(env.action_space.n, memory)
tmodel.load_state_dict(model.state_dict())

tot_iters = 0
N = 100
rs = np.zeros(N)
for n in range(N):
    eps = 2/np.sqrt(n+1)
    #eps = np.log(n+8)/np.sqrt(n+5)
    tot_r = play_one(env, model, tmodel, eps, 50)
    rs[n] = tot_r
    print(f"Episode {n}, total reward {tot_r}, avg loss")
print(f"Average reward for final 100 episodes: {rs[-100:].mean()}")

env.close()
