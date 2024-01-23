import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

# Set plotting options
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)
'''
# Create an environment and set random seed
env = gym.make('MountainCar-v0')
env.seed(505);

state = env.reset()
score = 0
for t in range(200):
    action = env.action_space.sample()
    env.render()
    state, reward, done, _ = env.step(action)
    score += reward
    if done:
        break
print('Final score:', score)
env.close()
'''

def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    # TODO: Implement this
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
    for l, h, b, splits in zip(low, high, bins, grid):
        print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    # TODO: Implement this
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension

import matplotlib.collections as mc

def visualize_samples(samples, discretized_samples, grid, low=None, high=None):
    """Visualize original and discretized samples on a given 2-dimensional grid."""

    fig, ax = plt.subplots(figsize=(10, 10))

    # Show grid
    ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
    ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
    ax.grid(True)

    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Otherwise use first, last grid locations as low, high (for further mapping discretized samples)
        low = [splits[0] for splits in grid]
        high = [splits[-1] for splits in grid]

    # Map each discretized sample (which is really an index) to the center of corresponding grid cell
    grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T))  # add low and high ends
    grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2  # compute center of each grid cell
    locs = np.stack(grid_centers[i, discretized_samples[:, i]] for i in range(len(grid))).T  # map discretized samples

    ax.plot(samples[:, 0], samples[:, 1], 'o')  # plot original samples
    ax.plot(locs[:, 0], locs[:, 1], 's')  # plot discretized samples in mapped locations
    ax.add_collection(mc.LineCollection(list(zip(samples, locs)), colors='orange'))  # add a line connecting each original-discretized sample
    ax.legend(['original', 'discretized'])
    plt.show()

'''
#visualize_samples(samples, discretized_samples, grid, low, high)

#env = gym.make('Pendulum-v1')
#env = gym.make('MountainCarContinuous-v0')
env = gym.make('Hopper-v3')
env.seed(505);
# Explore state (observation) space
print("State space:", env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)


# Generate some samples from the state space
print("State space samples:")
print(np.array([env.observation_space.sample() for i in range(10)]))

# Explore the action space
print("Action space:", env.action_space)

# Generate some samples from the action space
print("Action space samples:")
print(np.array([env.action_space.sample() for i in range(10)]))
#exit()
low = np.array([-10,-10,-10,-10,-10,-10,-10,-10,-10,-10])
high = np.array([10,10,10,10,10,10,10,10,10,10])
# Create a grid to discretize the state space
state_grid = create_uniform_grid(low, high, bins=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100))

# Obtain some samples from the space, discretize them, and then visualize them
state_samples = np.array([env.observation_space.sample() for i in range(10)])
#print("state samples : ",state_samples[:,0:5])
print("state samples : ",state_samples)

#discretized_state_samples = np.array([discretize(sample, state_grid) for sample in state_samples[:,0:5]])
discretized_state_samples = np.array([discretize(sample, state_grid) for sample in state_samples])

print("discretized state samples ",discretized_state_samples)

exit()
print(discretized_state_samples[0])
a,b,c,d,e = discretized_state_samples[0]
#[46 54 53 44 41]
print(state_grid[0][a])
print(state_grid[1][b])
print(state_grid[2][c])
print(state_grid[3][d])
print(state_grid[4][e])
#visualize_samples(state_samples, discretized_state_samples, state_grid,
#                  env.observation_space.low, env.observation_space.high)
#plt.xlabel('position'); plt.ylabel('velocity');  # axis labels for MountainCar-v0 state space

# Create a grid to discretize the action space
act_grid = create_uniform_grid(low, high, bins=(100,100,100))

# Obtain some samples from the space, discretize them, and then visualize them
act_samples = np.array([env.action_space.sample() for i in range(10)])
print(act_samples)

discretized_act_samples = np.array([discretize(sample, act_grid) for sample in act_samples])
print(discretized_act_samples)
'''


env = gym.make('Humanoid-v3')
env = gym.make('Swimmer-v3')
env.seed(505);
# Explore state (observation) space
print("State space:", env.action_space)
print("- low:", env.action_space.low)
print("- high:", env.action_space.high)

low = env.action_space.low
high = env.action_space.high

# Create a grid to discretize the action space
act_grid = create_uniform_grid(low, high, bins=(100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100))

# Obtain some samples from the space, discretize them, and then visualize them
act_samples = np.array([env.action_space.sample() for i in range(10)])
print(act_samples)

discretized_act_samples = np.array([discretize(sample, act_grid) for sample in act_samples])
print(discretized_act_samples)
