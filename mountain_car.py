import gym
import pybulletgym
from agent import Agent
from tqdm import tqdm
import torch
import argparse
import os
import pickle

PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-e', '--episodes', default=150, type=int, help='number of episodes')
PARSER.add_argument('-m', '--model', default=True, type=bool, help='use existing model(True/False)')
PARSER.add_argument('-d', '--dir', default='./models/model.pth', type=str, help='model directory')
PARSER.add_argument('-t', '--train', default=False, type=bool, help='train for [number of episodes] IF MODEL EXISTS')
ARGS = PARSER.parse_args()
if ARGS.train:

    no_episodes=ARGS.episodes
    env = gym.make('MountainCar-v0')
    Car = Agent(gamma=0.99, eps=0.7, eps_decay=0.95, eps_min=0.3, batch_size=64, n_actions=env.action_space.n,
                input_size=env.observation_space.shape[0], rate=0.03)
    if ARGS.model:
        if os.path.isdir(ARGS.dir):
            torch.load(Car.network,ARGS.dir)
    succesful_episode_durations = []
    target_updation = 20
    count = 0
    #no_episodes = 300
    episode_length = 1000
    state = env.reset()
    max_pos = state[0]
    max_positions = []
    flag = False
    for i in tqdm(range(no_episodes)):
        # Take a random step and initalise the variables
        for t in range(episode_length):
            action = Car.select_action(state)
            new_state, reward, done, _ = env.step(action)
            if new_state[0] > max_pos:
                reward = new_state[0] + 10
                max_positions.append(new_state[0])
                max_pos = new_state[0]
            else:
                reward = new_state[0] + 0.5
            Car.push(state, action, new_state, reward)
            state = new_state
            Car.learn_from_replay()
            count += 1
            if flag and done:
                flag = False
                print("Current Max Position {}".format(max_pos))
                succesful_episode_durations.append(t + 1)
                # plot_durations()
                break
        if i % target_updation == 0:
            Car.target.load_state_dict(Car.network.state_dict())

        state = env.reset()
    print('Complete')

else:
    env = gym.make('MountainCar-v0')
    Car = Agent(gamma=0.99, eps=0.7, eps_decay=0.95, eps_min=0.3, batch_size=64, n_actions=env.action_space.n,
                input_size=env.observation_space.shape[0], rate=0.03)
    episode_length = 1000
    if ARGS.model:
        if os.path.isdir(ARGS.dir):
            print('Load Successful')
            torch.load(Car.network,ARGS.dir)
    for i in range(100):
        state=env.reset()
        env.render()
        for t in range(episode_length):
            action = Car.select_action(state)
            new_state, reward, done, _ = env.step(action)