import gym
import numpy as np

env = gym.make("FrozenLake-v1")
obs_n = env.observation_space.n
act_n = env.action_space.n
# 策略评估函数
Q = np.zeros((obs_n, act_n))

# 学习参数
episode = 5000
alpha = 0.2
gamma = 0.7
epsilon = 1.0
min_epsilon = 0.01


def sample(state):
    # epsilon-贪心
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])

    return action


def e_decay(i, rate=0.005):
    global epsilon
    epsilon = min_epsilon + (epsilon - min_epsilon) * np.exp(-rate * i)


def main():
    for i in range(episode):
        state = env.reset()
        while True:
            action = sample(state)
            next_state, reward, done, _ = env.step(action)
            # Q-learning算法
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, action]) - Q[state, action])
            state = next_state
            if done:
                break

        e_decay(i)


def test():
    state = env.reset()
    while True:
        env.render()
        action = np.argmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)
        state = new_state
        if done:
            break

    env.render()


if __name__ == '__main__':
    main()
    test()
