"""基于gym库，利用蒙特卡洛算法解决经典冰湖问题"""

import gym
import numpy as np
import time

env = gym.make("FrozenLake-v1")

# 策略评估函数
Q = np.zeros((env.observation_space.n, env.action_space.n))
# 状态访问次数
n_s_a = np.zeros((env.observation_space.n, env.action_space.n))
# 学习参数
num_episodes = 1000000
epsilon = 0.2


# 一个打印进度条的装饰器
def training_progress_decorator(size, type=''):
    """ print train process by a bar"""

    def bar_process(progress):
        nonlocal size
        bar_length = 20
        filled_length = int(round(bar_length * progress))
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        percent = round(progress * 100, 3)
        print(f'\r{type} Progress: |{bar}| {percent}%', end='', flush=True)

    def decorator(func):
        nonlocal size
        cnt = 0

        def wrapper(*args, **kwargs):
            nonlocal cnt
            cnt += 1
            res = func(*args, **kwargs)
            bar_process(cnt / size)

            return res

        return wrapper

    return decorator


@training_progress_decorator(size=num_episodes)
def print_bar():
    pass


def main():
    for i in range(num_episodes):
        state = env.reset()
        results_list = []
        result_sum = 0.0
        while True:
            # epsilon-贪心
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            new_state, reward, done, _ = env.step(action)
            results_list.append((state, action))
            result_sum += reward
            if done:
                break
            state = new_state

        # 策略评估
        for (state, action) in results_list:
            n_s_a[state, action] += 1.0
            alpha = 1.0 / n_s_a[state, action]
            Q[state, action] += alpha * (result_sum - Q[state, action])

        print_bar()


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
    print('train over')
    time.sleep(2)
    test()
    env.close()
