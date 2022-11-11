import random

class LR_WORLD():
    def __init__(self):
        self.x = 0
        self.his = []
        self.count = 0
        self.dic_state = dict()

    def step(self, a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if a==0:
            self.move_left()
            self.his.append('L')

            if self.x == 83:        # [LRLRLL]
                reward = 1000
            else:
                reward = -1

        elif a==1:
            self.move_right()
            self.his.append('R')

            reward = + 1

        self.count += 1
        done = self.is_done()
        return self.x, reward, done, self.his, self.count

    def move_right(self):
        self.x = self.x * 2 + 2
      
    def move_left(self):
        self.x = self.x * 2 + 1

    def is_done(self):
        if self.count == 6:
            return True
        else:
            return False

    def get_state(self):
        return self.x
      
    def reset(self):
        self.x = 0
        self.count = 0
        self.his = []
        return

    def state(self):
        if self.dic_state.get(self.x) == None:
            self.dic_state[self.x] = self.his[::1]
        x = sorted(self.dic_state.items())
        return dict(x)

class Agent():
    def __init__(self):
        pass        

    def select_action(self):
        coin = random.random()
        if coin < 0.5:
            action = 0
        else:
            action = 1

        return action


def main():

    global x
    env = LR_WORLD()
    agent = Agent()
    data = []
    for i in range(2 ** 7 - 1):
        data.append(0)
    gamma = 1.0
    alpha = 0.001

    for k in range(10000):
        done = False
        history = [(0,0)]

        while not done:
            action = agent.select_action()
            x, reward, done,env.his, env.count = env.step(action)
            history.append((x,reward))
            env.state()

        env.reset()

        cum_reward = 0
        for transition in history[::-1]:
            x, reward = transition
            data[x] = data[x] + alpha*(cum_reward-data[x])
            cum_reward = cum_reward + gamma*reward

    for state, value in enumerate(data[:63]):
        try:
            print(state, value, env.dic_state[state])
        except:
            pass

if __name__ == '__main__':
    main()
