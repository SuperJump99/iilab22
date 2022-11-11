import random
import numpy as np


class LR_WORLD():
    def __init__(self):
        self.x = 0
        self.his = []
        self.count = 0
        self.dic_state = {0: []}

    def step(self, a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if a == 0:
            self.move_left()
            self.his.append('L')

            if self.x == 83:  # [LRLRLL]
                reward = 1000
            else:
                reward = -1

        elif a == 1:
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
        return self.x

    def state(self):
        if self.dic_state.get(self.x) == None:
            self.dic_state[self.x] = self.his[::1]
        x = sorted(self.dic_state.items())
        return dict(x)


class QAgent():
    def __init__(self):
        self.q_table = np.zeros((127,2))        # q벨류를 저장하는 변수. 모두 0으로 초기화.
        self.dic_state = dict()
        self.eps = 0.9
        self.alpha = 0.01

    def select_action(self,s):
        # eps-greedy로 액션을 선택
        x = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0, 1)
        else:
            action_val = self.q_table[x]
            action = np.argmax(action_val)
        return action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        x = s
        next_x = s_prime
        a_prime = self.select_action(s_prime) # S'에서 선택할 액션 (실제로 취한 액션이 아님)
        # SARSA 업데이트 식을 이용
        self.q_table[x,a] = self.q_table[x,a] + 0.1 * (r + self.q_table[next_x,a_prime] - self.q_table[x,a])

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)


    def get_data(self):
        # 학습이 각 위치에서 어느 액션의 q 값이 가장 높았는지 보여주는 함수
        q_lst = self.q_table.tolist()
        data = []
        for i in range(2 ** 7 - 1):
            data.append(0)

        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            data[row_idx] = np.argmax(row)
        return data


      
def main():
    env = LR_WORLD()
    agent = QAgent()

    for n_epi in range(10000):
        done = False

        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime, r, done, env.his, env.count = env.step(a)
            env.state()

            agent.update_table((s,a,r,s_prime))
            s = s_prime
        agent.anneal_eps()

    agent.get_data()
    for state, action in enumerate(agent.get_data()[:63]): # 학습이 끝난 결과를 출력
        if action == 1:
            act = 'R'
        else:
            act = 'L'
        try:
            print(state,  env.dic_state[state],'\t>>', act)
        except:
            pass


if __name__ == '__main__':
    main()