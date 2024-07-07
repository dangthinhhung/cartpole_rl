import matplotlib.pyplot as plt
from environment import CartPoleEnv
import gymnasium as gym
import numpy as np
import pickle
import os


def run(is_training=True, render=False, threshold=1000,
        name='default_model', lr=0.1, df=0.99,eps=1, eps_dr=0.00001):
    # Create the CartPole environment
    # env = gym.make('CartPole-v1', render_mode='human' if render else None)
    env = CartPoleEnv(render_mode='human' if render else None)
    # Divide position, velocity, pole angle, and pole angular velocity into segments
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    Q_tab_path = os.path.join(name, 'Q_table.pkl')

    if is_training:
        try:
            os.mkdir(name)
        except OSError as error:
            print(error)

        # Initialize a 11x11x11x11x2 array for Q-table
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))
    else:
        # Load Q-table from file
        with open(Q_tab_path, 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = lr  # Alpha or learning rate
    discount_factor_g = df  # Gamma or discount factor

    epsilon = eps  # 1 = 100% random actions
    epsilon_decay_rate = eps_dr  # Epsilon decay rate
    rng = np.random.default_rng()  # Random number generator

    rewards_per_episode = []

    i = 0

    # Training loop
    while True:
        state = env.reset()[0]  # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False  # True when reached goal

        rewards = 0

        # Episode loop
        while not terminated and rewards < 10000:
            if is_training and rng.random() < epsilon:
                # Choose random action  (0=go left, 1=go right)
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            if is_training:
                q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :]) - q[state_p, state_v, state_a, state_av, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward

            if not is_training and rewards % 100 == 0:
                print(f'Episode: {i}  Rewards: {rewards}')

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[-100:])

        if is_training and i % 100 == 0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')

        if mean_rewards > threshold:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        i += 1

    env.close()

    # Save Q-table to file
    if is_training:
        with open(Q_tab_path, 'wb') as f:
            pickle.dump(q, f)

    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
    plt.plot(mean_rewards)
    plt.savefig(f'{name}/Rewards_log.png')
