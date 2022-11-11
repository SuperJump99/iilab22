import numpy as np
import random
import copy
import datetime
import platform
import torch
import torch.nn.functional as F
from collections import deque


load_path = f"./saved_models/DDPG/20210709235643"


test_step = 10000       # 테스트 횟수
print_interval = 1      # 출력 빈도수

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_env():
    pass


# 네트워크 모델 가중치만 불러오기, 전체 모델 불러오는 것 X
def load_model():
    print(f"... Load Model from {load_path}/ckpt ...")
    checkpoint = torch.load(load_path + '/ckpt', map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    target_actor.load_state_dict(checkpoint["actor"])
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    critic.load_state_dict(checkpoint["critic"])
    target_critic.load_state_dict(checkpoint["critic"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    return actor, target_actor, actor_optimizer, critic, target_critic, critic_optimizer


if __name__ == '__main__':

    # 환경 불러오기
    load_env()

    # 학습된 모델의 파라미터 불러오기
    load_model()

    # Inference 시작
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    print("TEST START")
    for step in range(test_step):

        # 수정이 필요한 부분들
        state = dec.obs[0]      # 환경에서 현재 state 부분가져옴
        action = actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()      # 현재 state에서 취한 action을 받아옴
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

