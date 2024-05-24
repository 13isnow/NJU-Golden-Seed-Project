""" 基于gym库的DP解决FrozenLake问题 """
import numpy as np
import gym

# 初始化环境
env = gym.make('FrozenLake-v1')
env.reset()
n_states = env.observation_space.n
n_actions = env.action_space.n

env = env.unwrapped
# 初始化价值函数，所有状态的初始价值设为0
Values = np.zeros(n_states)
Pi = np.ones([n_states, n_actions]) * (1.0 / n_actions)

# 学习参数
gamma = 0.9  # 折扣因子
theta = 1e-10  # 收敛阈值


def get_value(state, action):
    value = 0.0
    for prob, next_state, reward, done in env.P[state][action]:
        next_value = Values[next_state] * gamma
        if done:
            next_value = 0
        next_value += reward
        next_value *= prob
        value += next_value

    return value


def state_value():
    new_values = np.zeros(n_states)
    for state in range(n_states):
        action_value = np.zeros(n_actions)
        for action in range(n_actions):
            action_value[action] = get_value(state, action)

        new_values[state] = action_value.max()

    return new_values


def pi_up():
    new_pi = np.zeros([n_states, n_actions])
    for state in range(n_states):
        action_value = np.zeros(n_actions)
        for action in range(n_actions):
            action_value[action] = get_value(state, action)
        count = (action_value == action_value.max()).sum()
        for action in range(n_actions):
            if action_value[action] == action_value.max():
                new_pi[state, action] = 1.0 / count
            else:
                new_pi[state, action] = 0
    return new_pi


def main():
    global Values, Pi

    for _ in range(10):
        for _ in range(100):
            new_Values = state_value()
            if np.linalg.norm(Values - new_Values) <= theta:
                break

            Values = new_Values

        Pi = pi_up()


def test():
    global Values, Pi, env
    state = env.reset()
    while True:
        env.render()
        best_action_pros = Pi[state].max()
        actions = np.where(Pi[state] == best_action_pros)[0]
        best_action = dict()
        for action in actions:
            best_action[get_value(state, action)] = action

        best = best_action[max(best_action)]

        state, reward, done, _ = env.step(best)
        if done:
            break

    env.render()


if __name__ == '__main__':
    main()
    test()


