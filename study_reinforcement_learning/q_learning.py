import gymnasium as gym
# from gym import envs
import numpy as np
import cv2


def get_status(_observation):
    """
        gym envから得られる情報をもとに、座標positionと速度velocityを取得する。
    """
    env_low = env.observation_space.low  # 位置と速度の最小値
    env_high = env.observation_space.high  # 位置と速度の最大値
    env_dx = (env_high - env_low) / 40  # 40等分
    # 0〜39の離散値に変換する
    position = int((_observation[0] - env_low[0])/env_dx[0])
    velocity = int((_observation[1] - env_low[1])/env_dx[1])
    return position, velocity


def update_q_table(_q_table, _action,  _observation, _next_observation, _reward):
    """
        価値関数Qの更新
        Args:
            _q_table: 現在の推定価値関数Q
            _action: ε-greedy法で選択したaction
            _observation: 現在の状態s（の元となる情報）
            _next_observation: 次の状態s（の元となる情報）
            _reward: 報酬（gym envから得られたもの）
        Output:
    """
    alpha = 0.2  # 学習率
    gamma = 0.99  # 時間割引き率
    # 行動後の状態で得られる最大行動価値 Q(s',a')
    next_position, next_velocity = get_status(_next_observation)
    next_max_q_value = max(_q_table[next_position][next_velocity])
    # 行動前の状態の行動価値 Q(s,a)
    position, velocity = get_status(_observation)
    q_value = _q_table[position][velocity][_action]
    # 行動価値関数の更新
    _q_table[position][velocity][_action] = q_value + alpha * (_reward + gamma * next_max_q_value - q_value)
    return _q_table


def get_action(_q_table):
    epsilon = 0.002
    # position, velocity = get_status(observation)
    # _action = np.argmax(_q_table[position][velocity])
    if np.random.uniform(0, 1) > epsilon:
        position, velocity = get_status(observation)
        _action = np.argmax(_q_table[position][velocity])
    else:
        _action = np.random.choice([0, 1, 2])
    return _action


if __name__ == '__main__':
    # env = gym.make('MountainCar-v0', render_mode="human")
    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    # Qテーブルの初期化
    q_table = np.zeros((40, 40, 3))
    # observation = env.reset()
    rewards = []
    # 10000エピソードで学習する
    for episode in range(10000):
        # print(episode)
        total_reward = 0
        observation = env.reset()[0]
        for _ in range(200):
            # ε-greedy法で行動を選択
            action = get_action(q_table)
            # 車を動かし、観測結果・報酬・ゲーム終了FLG・詳細情報を取得
            next_observation, reward, done, _, _ = env.step(action)
            # next_observation, reward, done, _ = env.step(action)[0]
            # Qテーブルの更新
            q_table = update_q_table(q_table, action, observation, next_observation, reward)
            total_reward += reward
            observation = next_observation
            if done:
                # doneがTrueになったら１エピソード終了
                if episode % 100 == 0:
                    print('episode: {}, total_reward: {}'.format(episode, total_reward))
                    print(q_table.max(), q_table.min())
                rewards.append(total_reward)
                break
    #         if episode % 3000 == 0:
    #             img = env.render()
    #             img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
    #             cv2.imshow("test", img)
    #             img2 = (q_table - q_table.min() - 5) * -255/5
    #             # img2 = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
    #             cv2.imshow("test2", img2)
    #             cv2.waitKey(30)
    # q_table = -1  * q_table
    # np.savetxt("nptext.txt", q_table.min(axis=2, ), delimiter=',')