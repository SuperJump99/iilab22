import numpy as np
import copy
import platform
import torch
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel

datetime = "2022_11_11_144602"

load_path = f"./saved_models/Drone/DDPG/{datetime}"

# DDPG를 위한 파라미터 값 세팅
state_size = 9      # 벡터관측정보 m개 x 벡터의 n차원 -> m*n
action_size = 3     # n차원 백터로의 움직임

num_of_inputs = None # 초기 상태의 개수  (num_of_inputs == state_size) ??

test_step = 10000       # 테스트 횟수
print_interval = 10      # 출력 빈도수

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 유니티 환경 경로
game = "Drone"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
    env_name = f"../envs/{game}_{os_name}"

# Actor 클래스 -> DDPG Actor 클래스 정의
class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.mu = torch.nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.mu(x))

# Critic 클래스 -> DDPG Critic 클래스 정의
class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128+action_size, 128)
        self.q = torch.nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.cat((x, action), dim=-1)
        x = torch.relu(self.fc2(x))
        return self.q(x)

# DDPGAgent 클래스 -> DDPG 알고리즘을 위한 다양한 함수 정의
def Load_Model():
    global actor
    actor = Actor().to(device)
    target_actor = copy.deepcopy(actor)
    actor_optimizer = torch.optim.Adam(actor.parameters())
    critic = Critic().to(device)
    target_critic = copy.deepcopy(critic)
    critic_optimizer = torch.optim.Adam(critic.parameters())

    print(f"... Load Model from {load_path}/ckpt ...")
    checkpoint = torch.load(load_path+'/ckpt', map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    target_actor.load_state_dict(checkpoint["actor"])
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    critic.load_state_dict(checkpoint["critic"])
    target_critic.load_state_dict(checkpoint["critic"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

def get_action(actor, state):
    action = actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
    return action


if __name__ == '__main__':

    # TODO: 1
    #  (86~95) 환경 수정 필요
    # 환경 불러오기
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel])
    env.reset()

    # 유니티 브레인 설정
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    dec, term = env.get_steps(behavior_name)

    Load_Model()

    # 학습된 모델의 파라미터 불러오기
    # Inference 시작
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    print("TEST START")
    for input in range(num_of_inputs):          # TODO: input-> 각 state의 초기 상태의 개수
        for step in range(test_step):           # TODO: total calculation = num_of_inputs * test_step
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

            # TODO: 2
            #  (109~119) state action reward 수정 필요
            state = dec.obs[0]      # 환경에서 현재 state 부분가져옴
            action = get_action(actor,state)
            action_tuple = ActionTuple()        # 행동을 담는 튜플 객체
            action_tuple.add_continuous(action)
            env.set_actions(behavior_name, action_tuple)        # 환경에 행동을 전달
            env.step()      # 환경에서 에이전트가 실제로 행동을 수행

            dec, term = env.get_steps(behavior_name)
            done = len(term.agent_id) > 0
            reward = term.reward if done else dec.reward
            next_state = term.obs[0] if done else dec.obs[0]
            score += reward[0]      # 종료시점까지 보상 누적합

            if done:        # 1회 에피소드 종료
                episode += 1
                scores.append(score)        # 1회 에피소드의 보상의 총합 저장
                score = 0       # 보상합 초기화

                # 게임 진행 상황 출력
                if episode % print_interval == 0:
                    mean_score = np.mean(scores)        # 보상합
                    mean_actor_loss = np.mean(actor_losses)
                    mean_critic_loss = np.mean(critic_losses)
                    actor_losses, critic_losses, scores = [], [], []   # 계산 값 초기화

                    print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " + \
                          f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")

    env.close()
