import collections
import random
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from copy import deepcopy as cp
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torchvision.models as models

# Hyperparameters
learning_rate = 0.01
gamma = 0.98
buffer_limit = 3000
batch_size = 32
k= 1.5

weight = [1, 1, 1, 1, 1, 1, 1]
ready_time = [0, 0, 0]
processing_time = [3, 4, 5, 5, 2, 1, 4,]
due_date = [8, 10, 7, 9, 8, 6, 5]


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
        self.fc1 = nn.Linear(len(processing_time), 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 2)
        else:
            return out.argmax().item()

class Scheduling():
    def __init__(self):
        self.ready_time = np.array(ready_time)
        self.processing_time = np.array(processing_time)
        self.due_date = np.array(due_date)
        self.weight = weight
        self.count = 0
        self.state = np.zeros(len(processing_time))
        self.flow_time = 0

    def step(self,a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if a == 0:
            self.EDD()

        elif a == 1:
            self.STP()

        elif a == 2:
            self.ATC_rule()

        self.cal_complete_time()
        # print(self.cal_complete_time(),'ct')
        reward = self.cal_tardiness()
        # print(reward, 'reward')
        self.count += +1
        done = self.is_done()
        return self.state, reward, done, self.count


# 초기 상태[count(총작업개수)-(한 스텝), 선택한 작업, 작업 가능 여부, 현재 완료 시간 ,processing time, duedate]
    def cal_complete_time(self):
        # print(processing_time[index])
        # print(self.flow_time, 'ct')
        self.flow_time = self.flow_time + processing_time[index]
        return self.flow_time
# reward 계산 부분
    def cal_tardiness(self):
        self.tardiness = due_date[index] - self.flow_time
        if self.tardiness >= 0:
            reward = 0
        else:
            reward = self.tardiness

        return reward

    def EDD(self):
        global index

        val = np.min(ma.masked_where(self.due_date == 0,self.due_date))
        # print(val)
        index = np.argmin(ma.masked_where(self.due_date == 0, self.due_date))
        # print(index)
        self.state[self.count] = index + 1
        self.due_date[index] = 0
        self.processing_time[index] = 0

    def STP(self):
        global index

        val = np.min(ma.masked_where(self.processing_time == 0, self.processing_time))
        # print(val)
        index = np.argmin(ma.masked_where(self.processing_time == 0, self.processing_time))
        self.state[self.count] = index + 1
        self.due_date[index] = 0
        self.processing_time[index] = 0

    def ATC_rule(self):
        global index
        ATC_list = list(np.zeros(len(due_date)))

        for index in range(len(self.due_date)):
            if self.due_date[index] != 0:

                w = self.weight[index]
                slack_time = self.due_date[index] - self.processing_time[index]

                p_bar = (sum(self.processing_time)  / (len(self.due_date) - self.count))  # 1은 최초 시작할때를 의미, 이 후 2,3 ... 씩 진행

                I = w / self.processing_time[index] * math.exp(-max(slack_time, 0) / (k * p_bar))

                ATC_list[index] = I
        index = (ATC_list.index(max(ATC_list)))
        self.state[self.count] = index + 1
        self.processing_time[index] = 0
        self.due_date[index] = 0


    def is_done(self):
        if self.count == len(processing_time):
            return True
        else:
            return False

    def reset(self):
        self.state = np.zeros([len(processing_time)])
        self.count = 0
        self.flow_time = 0
        self.processing_time = np.array(processing_time)
        self.due_date = np.array(due_date)
        return self.state

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

def plot_results(n_epi,score):
    x = np.linspace(0, n_epi, n_epi+1)
    plt.plot(x, score)
    plt.xlim((0, n_epi))
    plt.ylim((min(score)-5,max(score)+5 ))
    plt.grid()
    plt.show()


def main():
    env = Scheduling()
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    memo = []

    print_interval = 5
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(1500):
        epsilon = max(0.7, 0.1 - 0.01 * (n_epi / 2000))  # Linear annealing from 8% to 1%1
        if memory.size() > 2999:
            epsilon = max(0.01, 0.1 - 0.01 * (n_epi / 200))
        s = env.reset()

        done = False
        while not done:
            s_copy = cp(s)
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)

            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            if memory.size() < 3000:
                memory.put((s_copy, a, r , s_prime, done_mask))
            s = s_prime
            score += r

            if done:
                break
        memo.append(score)
        # score = 0.0
        if memory.size() > 1000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0 or n_epi == 0:
            q_target.load_state_dict(q.state_dict())

        print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
            n_epi, score , memory.size(), epsilon * 100))
        score = 0.0
    plot_results(n_epi,memo)

    model = Qnet()
    PATH = 'C:/Users/user/Desktop/2022 summer/Scheduling RL'
    torch.save(model.state_dict(), PATH +'model_state_dict.pt')

if __name__ == '__main__':
    main()
