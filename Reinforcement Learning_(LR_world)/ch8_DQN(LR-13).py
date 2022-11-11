import collections
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy as cp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.001
gamma = 0.98
buffer_limit = 5000
batch_size = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class LR_WORLD():
    def __init__(self):
        self.x = 6
        self.count = 0
        self.state = np.zeros(13)

    def step(self, a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if a == 0:
            self.move_left()
            if (self.state == np.array([0,0,0,0,1,3,2,0,0,0,0,0,0])).all():
                reward = 1000
            else:
                reward = -1

        elif a == 1:
            self.move_right()
            reward = + 1

        self.count += 1
        done = self.is_done()
        return self.state, reward, done, self.count

    def move_right(self):
        self.x += 1
        self.state[self.x] += 1

    def move_left(self):
        self.x -= 1
        self.state[self.x] += 1

    def is_done(self):
        if self.count == 6:
            return True
        else:
            return False

    def reset(self):
        self.x = 6
        self.state = np.zeros([13])
        self.count = 0
        return self.state

def plot_results(n_epi,score):
    x = np.linspace(0, n_epi, n_epi+1)
    plt.plot(x, score)
    plt.xlim((0, n_epi))
    plt.ylim((-10, 1000))
    plt.grid()
    plt.show()

def epsilon(n_epi,memo_score):
    epsilon = max(0.01, 0.30 - 0.01 * (n_epi / 2000))
    if memo_score > 44.3:
        epsilon = 0.0
    return epsilon

def main():
    env = LR_WORLD()
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 100
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    memo = []
    memo_score = 0

    for n_epi in range(5000):

        # if n_epi == 0:
        #     epsilon(n_epi,score,print_interval)  # Linear annealing from 8% to 1%1
        #     print(epsilon)
        # epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 2000))
        epsilon = 1
        if n_epi > 1000:
            epsilon = 0

        # try:
        #     if memo_score > 44.3:
        #         epsilon = 0.01
        # except:
        #     pass
        s = env.reset()
        done = False
        while not done:
            # print(type(s))
            # print(torch.from_numpy(s).float())
            s_copy = cp(s)
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            # print(a)
            # print(s,a)
            s_prime, r, done, info = env.step(a)
            # print(s)
            done_mask = 0.0 if done else 1.0
            if memory.size() < 5000:
                memory.put((s_copy, a, r, s_prime, done_mask))
            s = s_prime
            score += r
            # print(score)
            # print(done)
            if done:
                break
        memo.append(score)

        # score = 0.0
        if memory.size() > 100:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0 or n_epi == 0:
            # print(n_epi)
            q_target.load_state_dict(q.state_dict())
            #
        print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
            n_epi, score , memory.size(), epsilon * 100))
            # new_data = pd.DataFrame({"n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
            #     n_epi, score/print_interval , memory.size(), epsilon * 100)})
        memo_score = max(memo_score,score/print_interval)
        score = 0.0

    plot_results(n_epi,memo)
if __name__ == '__main__':
    main()