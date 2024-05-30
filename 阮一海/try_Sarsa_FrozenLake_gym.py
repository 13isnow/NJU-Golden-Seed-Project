"""基于gym库的Sarsa解决冰湖问题"""

import gym
import numpy as np

env = gym.make("FrozenLake-v1")
obs_n = env.observation_space.n
act_n = env.action_space.n
# 策略评估函数
Q = np.zeros((obs_n, act_n))

# 学习参数
episode = 1000
alpha = 0.1
gamma = 0.8
epsilon = 0.1


def sample(state):
    # epsilon-贪心
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])

    return action


def main():
    for i in range(episode):
        state = env.reset()
        while True:
            action = sample(state)
            next_state, reward, done, _ = env.step(action)
            if done:
                break
            next_action = sample(next_state)
            # Sarsa算法
            Q[state, action] += alpha * (reward + gamma * Q[next_action, next_action] - Q[state, action])
            state, action = next_state, next_action


def test():
    state = env.reset()
    while True:
        env.render()
        action = np.argmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)
        if done:
            break
        state = new_state

    env.render()
    


if __name__ == '__main__':
    main()
    test()
