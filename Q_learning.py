# -*- coding: utf-8 -*-

__author__ = "Zifeng Wang"
__email__ = "wangzf18@mails.tsinghua.edu.cn"
__date__ = "20191109"

import numpy as np
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import pdb
import datetime

np.random.seed(2019)

"define the environment"


class Environment:
    def __init__(self):
        # set the reward matrix
        reward = np.zeros((4, 4))

        # place the cheese
        reward[3, 3] = 4
        reward[3, 0] = 2
        reward[[0, 0, 2], [3, 1, 1]] = 1

        # place the poison
        reward[[1, 1, 3], [3, 1, 1]] = -5

        self.max_row, self.max_col = reward.shape
        self.raw_reward = reward
        self.reward = reward
        self.action_map = {0: [0, -1],  # left
                           1: [0, 1],  # right
                           2: [-1, 0],  # up
                           3: [1, 0],  # down
                           }

    def step(self, action, state):
        """Given an action, outputs the reward.
        Args:
            action: int, an action taken by the agent
            state: str, the current state of the agent

        Outputs:
            next_state: list, next state of the agent
            reward: float, the reward at the next state
            done: bool, stop or continue learning
        """
        done = False
        next_state = json.loads(state)
        state_shift = self.action_map[action]

        next_state = np.array(next_state) + np.array(state_shift)
        next_state = next_state.tolist()

        reward = self.reward[next_state[0], next_state[1]]
        self.reward[next_state[0], next_state[1]] = 0

        next_state = json.dumps(next_state)

        if reward < 0 or reward == 4:
            done = True

        return next_state, reward, done

    def reset(self, is_training=True):
        """Reset the environment, state, and reward matrix.
        Args:
            is_training: bool, if True, the initial state of the agent will be set randomly,
                or the initial state will be (0,0) by default.

        Outputs:
            state: str, the initial state.
        """
        if is_training:
            # randomly init
            while True:
                init_state = np.random.randint(0, 4, (2))
                if self.reward[init_state[0], init_state[1]] >= 0:
                    init_state = init_state.tolist()
                    break
        else:
            init_state = [0, 0]

        self.state = json.dumps(init_state)
        self.reward = self.raw_reward.copy()
        return self.state


"define the agent"


class Agent:
    def __init__(self, *args, **kwargs):
        self.gamma = kwargs["gamma"]
        self.alpha = kwargs["alpha"]
        self.eps = kwargs["eps"]
        self.max_col = kwargs["max_col"]
        self.max_row = kwargs["max_row"]

        # action is [0,1,2,3]: left, right, up, down
        self.action_space = [0, 1, 2, 3]

        # self.Q = defaultdict(lambda: np.array([.0, .0, .0, .0]))
        self.Q = defaultdict(lambda: np.random.rand(4))

    def do(self, state):
        """Know the current state, choose an action.
        Args:
            state: str, as "[0, 0]", "[0, 1]", etc.
        Outputs:
            action: an action the agent decides to take, in [0, 1, 2, 3].
        """

        """Please Fill Your Code Here.
        """
        possible_actions = self.action_space.copy()
        # if '0' in state.split(',')[0]:
        #     possible_actions.remove(2)
        # if '0' in state.split(',')[1]:
        #     possible_actions.remove(0)
        # if '3' in state.split(',')[0]:
        #     possible_actions.remove(3)
        # if '3' in state.split(',')[1]:
        #     possible_actions.remove(1)
        if int(state[1]) == 0:
            possible_actions.remove(2)
        if int(state[4]) == 0:
            possible_actions.remove(0)
        if int(state[1]) == self.max_row:
            possible_actions.remove(3)
        if int(state[4]) == self.max_col:
            possible_actions.remove(1)

        if self.eps > np.random.random():
            action = np.random.choice(possible_actions)
        else:
            Q_state = self.Q[state]
            action = np.argmax(Q_state)
            while action not in possible_actions:
                Q_state[action] = 0
                action = np.argmax(Q_state)

        return action

    def learn(self, state, action, reward, next_state):
        """Learn from the environment.
        Args:
            state: str, the current state
            action: int, the action taken by itself
            reward: float, the reward after taking the action
            next_state: str, the next state

        Outputs:
            None
        """

        """Please Fill Your Code Here.
        """

        Q_state = self.Q[state]
        Q_state_action = Q_state[action]
        Q_state_action = (1-self.alpha)*Q_state_action + self.alpha*(reward + self.gamma*self.Q[next_state][np.argmax(self.Q[next_state])])
        Q_state[action] = Q_state_action
        self.Q[state] = Q_state

        return

    def getQtable(self):
        return self.Q



def calculateValueDistribution(Qtable):
    reward_table = np.zeros((4, 4))
    reward_table[3, 3] = 4
    reward_table[3, 0] = 2
    reward_table[[0, 0, 2], [3, 1, 1]] = 1
    reward_table[[1, 1, 3], [3, 1, 1]] = -5

    avg_reward = np.zeros((4, 4))
    for entry in Qtable:
        x = entry.split(',')[0]
        x = x.strip('[')
        y = entry.split(',')[1]
        y = y.strip(']')
        avg_reward[int(x)][int(y)] = np.average(Qtable[entry])

    mse = np.square(avg_reward - reward_table).mean()
    print('Mean Square Error is: ', mse)
    return

if __name__ == "__main__":

    "define the parameters"
    agent_params = {
        "gamma": 0.8,  # discounted rate
        "alpha": 0.1,  # learning rate
        "eps": 0.9,  # initialize the e-greedy
    }

    max_iter = 10000

    # initialize the environment & the agent
    env = Environment()

    # the mouse cannot jump out of the grid
    agent_params["max_row"] = env.max_row - 1
    agent_params["max_col"] = env.max_col - 1
    smart_mouse = Agent(**agent_params)

    "start learning your policy"
    step = 0
    for step in range(max_iter):
        current_state = env.reset()
        while True:
            # act
            action = smart_mouse.do(current_state)

            # get reward from the environment
            next_state, reward, done = env.step(action, current_state)

            # learn from the reward
            smart_mouse.learn(current_state, action, reward, next_state)

            current_state = next_state

            if done:
                print("Step {} done, reward {}".format(step, reward))
                break

                # epsilon decay
        if step % 100 == 0 and step > 0:
            eps = - 0.99 * step / max_iter + 1.0
            eps = np.maximum(0.1, eps)
            print("Eps:", eps)
            smart_mouse.eps = eps

    "evaluate your smart mouse"
    current_state = env.reset(False)
    trajectory_mat = env.reward.copy()
    trajectory_mat[0, 0] = 5
    smart_mouse.eps = 0.0

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.ion()
    epoch = 0
    while True:
        action = smart_mouse.do(current_state)
        next_state, reward, done = env.step(action, current_state)
        current_state = next_state
        trajectory_mat[json.loads(current_state)[0], json.loads(current_state)[1]] = 5

        ax.matshow(trajectory_mat, cmap="coolwarm")
        plt.pause(0.5)

        print("===> [Eval] state:{}, reward:{} <===".format(current_state, reward))
        print("state value: \n", smart_mouse.Q[current_state])
        epoch += 1

        if done:
            Qtable = smart_mouse.getQtable()
            calculateValueDistribution(Qtable)
            plt.title(datetime.datetime.now().ctime())
            plt.savefig("1.png")
            print("***" * 10)
            if reward < 0 or epoch > 10:
                print("Your code does not work.")
            else:
                print("Your code works.")
            print("***" * 10)
            break
