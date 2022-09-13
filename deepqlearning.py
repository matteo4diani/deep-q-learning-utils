from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.activation import ReLU

import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from IPython import display as ipythondisplay
from pathlib import Path

from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
from baselines_wrappers import Monitor, DummyVecEnv, SubprocVecEnv

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()
print(f"""
If you are having problems running this notebook outside of Google Colab, check that: 
- torch has version {torch.__version__} 
- gym has version {gym.__version__}
""")
##################################
# Model Constants (taken from:   #
# "Human-level control through   #
# deep reinforcement learning")  #
# with some additions (Double Q) #
##################################

# Use Double-Q Learning as defined in:
# "Deep Reinforcement Learning with Double-Q Learning"
USE_DOUBLE = True
# Discount rate
GAMMA = 0.99
# How many transitions to sample from
BATCH_SIZE = 32
# How many transitions we're gonna store before overwrite
BUFFER_SIZE = int(1e6)
# How many transitions to accumulate before we start the actual training
MIN_REPLAY_SIZE = 50000
# Starting value of epsilon (probability of taking random action)
EPSILON_START = 1.0
# Final value of epsilon
EPSILON_END = 0.1
# Number of steps taken for EPSILON_START to become EPSILON_END
EPSILON_DECAY = int(1e6)
# Number of batch elements (environments created)
N_ENVS = 4
# Periodicity for target updates with the online values
TARGET_UPDATE_FREQ = 10000 // N_ENVS
# Learning Rate
LEARNING_RATE = 5e-5
# If True force taking action 1 at the start of each round to initiate gameplay.
# Setting FORCE_START to True may alter the learning process and sho
FORCE_START = False

#####################
# Utility Constants #
#####################

SAVE_PATHS = {True:  '/content/gdrive/MyDrive/deep-q-learning-atari/checkpoints/atari_model.pack', 
              False: 'checkpoints/atari_model.pack'}
LOG_DIRS = {True: '/content/gdrive/MyDrive/deep-q-learning-atari/tensorboard/atari_model',
            False: 'tensorboard/atari_model'}
# Use your personal Google Drive for parameter serialization and logs
USE_DRIVE = False
# Save parameters to disk/drive
SAVE_PARAMS = True
# Reload parameters from disk/drive
RELOAD_PARAMS = False
# Path for network parameters serialization
SAVE_PATH = SAVE_PATHS[USE_DRIVE]
SAVE_INTERVAL = 10000
# Path for TensorBoard logging
LOG_DIR = LOG_DIRS[USE_DRIVE]
LOG_INTERVAL = 1000

########################
# Network Architecture #
########################

def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
  """
  CNN architecture as defined in 'Human-level Control through 
  deep reinforcement learning'
  """
  # Get the number of input channels
  n_input_channels = observation_space.shape[0]

  cnn = nn.Sequential(
      nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
      nn.ReLU(),
      nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
      nn.ReLU(),
      nn.Flatten())
  # Compute shape by doing one forward pass through the cnn
  # and looking at the output shape of the tensor
  with torch.no_grad():
    # We are not passing this tensor to the gpu:
    # Our NNs will still be on the CPU when nature_cnn(...) is called.
    n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())
  
  return out

# Class representing the neural network, implements PyTorch nn.Module interface
class Network(nn.Module):
  def __init__(self, env, device, double):
    super().__init__()
    # Use Double-Q Learning
    self.double = double
    # Enable GPU support with explicit tensor/model allocation
    self.device = device
    # Number of actions available to the agent
    self.num_actions = env.action_space.n
    # Get Nature CNN instance
    conv_net = nature_cnn(env.observation_space)
    # Create network stacking the Nature CNN and a last layer 
    # dependent on the game environment 
    # (different num_actions, not knowable a-priori)
    self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))
  # Forward function is part of the interface for nn.Module
  def forward(self, x):
    return self.net(x)

  def act(self, obses, epsilon):
    # Convert observations to PyTorch tensor
    obses_tensor = torch.as_tensor(obses, 
                                   dtype=torch.float32, 
                                   device=self.device)
    
    # PyTorch already expects a batch of samples so we pass the tensor as-is
    # and we get a prediction from the Q-Network
    q_values = self(obses_tensor)

    # Get argmaxes of actions with best q
    max_q_indices = torch.argmax(q_values, dim=1)
    # Cast tensor into list of ints
    actions = max_q_indices.detach().tolist()

    # Implement epsilon-greedy policy.
    # We get P(random action) = epsilon by P(randint(0,1) <= epsilon) = epsilon
    for i in range(len(actions)):
      rnd_sample = random.random()
      if rnd_sample <= epsilon:
        actions[i] = random.randint(0, self.num_actions - 1)
    
    return actions

  def compute_loss(self, transitions, target_net):
    # Comb data and turn to numpy array for faster runs
    obses = [t[0] for t in transitions]
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = [t[4] for t in transitions]
    
    # If using frame-stacking use helper get_frames() to get numpy compliant obj
    if isinstance(obses[0], PytorchLazyFrames):
      obses = np.stack([o.get_frames() for o in obses])
      new_obses = np.stack([o.get_frames() for o in new_obses])
    else:
      obses = np.asarray(obses)
      new_obses = np.asarray(new_obses)

    # Turn to PyTorch tensor
    obses_tensor = torch.as_tensor(obses, 
                                   dtype=torch.float32, 
                                   device=self.device)
    # We unsqueeze(-1) to wrap each action/rew/... in an additional dimension
    actions_tensor = torch.as_tensor(actions,
                                     dtype=torch.int64,
                                     device=self.device).unsqueeze(-1)
    rewards_tensor = torch.as_tensor(rewards, 
                                     dtype=torch.float32,
                                     device=self.device).unsqueeze(-1)
    dones_tensor = torch.as_tensor(dones, 
                                   dtype=torch.float32,
                                   device=self.device).unsqueeze(-1)
    new_obses_tensor = torch.as_tensor(new_obses, 
                                       dtype=torch.float32,
                                       device=self.device)
    with torch.no_grad():
      if self.double:
        # We modify the network to use the online net for action selection
        # And the target net to compute Q-Values of actions.
        # Taken from "Deep Reinforcement Learning with Double-Q Learning"
        # By H. van Hasselt, A. Guez, and D. Silver from Google DeepMind.
        targets_online_q_values = self(new_obses_tensor)
        targets_online_best_q_indices = targets_online_q_values.argmax(dim=1, 
                                                                       keepdim=True)
        targets_target_q_values = target_net(new_obses_tensor)
        targets_selected_q_values = torch.gather(input=targets_target_q_values,
                                                 dim=1,
                                                 index=targets_online_best_q_indices)
        targets = rewards_tensor + GAMMA * (1 - dones_tensor) * targets_selected_q_values

      else:
        # Compute targets for loss function
        # We use the target net to predict target q-values for new obses
        # For each new observation we have a set of q-values
        # We need to condense this set to the one highest q-value per observation
        # (N.B. pytorch tensors .max() return argmax at index 1)
        target_q_values = target_net(new_obses_tensor)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        # Compute r + gamma*max(Q) ("if done -> r" is obtained via "1 - dones_tensor")
        targets = rewards_tensor + GAMMA * (1 - dones_tensor) * max_target_q_values

    # Compute Loss
    q_values = self(obses_tensor)

    # Get q-values for the actions we took
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_tensor)

    # Compute l1 loss
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    return loss

  def save(self, save_path):
    """Serialize network parameters to disk or Google Drive"""

    # We call .cpu() to transfer the tensor
    # from the gpu when converting to np array
    params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()}
    # Serialize network parameter dictionary with msgpack
    params_data = msgpack.dumps(params)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
      f.write(params_data)

  def load(self, load_path):
    """Load network parameters from disk or Google Drive"""
    if not os.path.exists(load_path):
      raise FileNotFoundError(load_path)

    with open(load_path, 'rb') as f:
      params_numpy = msgpack.load(f)
      # Convert to PyTorch tensors and load into network
      params = {k: torch.as_tensor(v, device=self.device) for k,v in params_numpy.items()}
      self.load_state_dict(params)

###############
# Model Setup #
###############

# Enable GPU support
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

# Load Breakout environment
# We use a custom wrapper made by @brthor
# The wrapper applies all the preprocessing steps described in
# "Human-level control through deep reinforcement learning"
# before the agent sees the observation.
# It also transforms [Height, Width, Channel] -> [C, H, W] (PyTorch format)
# Where H, W identify the pixel and channel is R, G or B.
# We wrap make_atari_deepmind in a Monitor object that enriches the info
# returned by env.step()
make_env = lambda: Monitor(make_atari_deepmind('BreakoutNoFrameskip-v4',
                                               scale_values=True), 
                           allow_early_resets=True)

# Double configuration for VecEnv: sequential (dummy) and parallel (subproc)
vec_env = DummyVecEnv([make_env for _ in range(N_ENVS)])
#env = SubprocVecEnv([make_env for _ in range(N_ENVS)])

# We implement frame-stacking via another custom wrapper by @brthor
# It's a VecEnv wrapper, so it wraps the vec_env directly,
# not in the builder lambda (make_env) like Monitor.
# BatchedPytorchFrameStack returns a PytorchLazyFrames instance
# when env.step() is called, instead of a numpy array. 
# The use of lazy frames avoids duplicating memory when frame-stacking.
env = BatchedPytorchFrameStack(vec_env, k=4)

# We create Doubly Ended Queue (deque) for fast append and pop (O(1))
# Transition Buffer
replay_buffer = deque(maxlen=BUFFER_SIZE)
# Episode Info Buffer
ep_infos_buffer = deque([0.0], maxlen=100)

episode_count = 0

# Implement TensorBoard logging
summary_writer = SummaryWriter(LOG_DIR)

online_net = Network(env, device=device, double=USE_DOUBLE)
target_net = Network(env, device=device, double=USE_DOUBLE)

# Delegate networks to GPU (if device = 'cpu' this does nothing)
online_net = online_net.to(device)
target_net = online_net.to(device)

# When at risk of OOM errors enable RELOAD_PARAMS to get back to last checkpoint
if RELOAD_PARAMS:
   online_net.load(SAVE_PATH)

# We set the target net parameters equal to the online_net params
# As specified in "Human-level control through deep reinforcement learning"
target_net.load_state_dict(online_net.state_dict())

# Create optimizer for gradient descent
optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

# Initialize Replay Buffer
obses = env.reset()

# If do_init_action[i] is True then environment i executes action 1 (start game)
do_init_action = [True for _ in range(N_ENVS)]

######################
# Replay Buffer Loop #
######################

for _ in range(MIN_REPLAY_SIZE):
  # Select random actions
  actions = [env.action_space.sample() for _ in range(N_ENVS)]

  # If we are reloading parameters after a notebook disconnect or OOM error
  # We build the replay set from the last network saved
  if RELOAD_PARAMS:
    # Epsilon decays linearly in time until reaching its final value
    epsilon = np.interp(int(1e5) * N_ENVS, 
                        [0, EPSILON_DECAY],
                        [EPSILON_START, EPSILON_END])
    # Get the actions from the online network.
    # If we are using frame-stacking with the custom wrapper
    # we unwrap observations and stack frames before passing them to net.act(...).
    # Epsilon-greedy policy is implemented in the net.act method
    if isinstance(obses[0], PytorchLazyFrames):
      act_obses = np.stack([o.get_frames() for o in obses])
      actions = online_net.act(act_obses, EPSILON_START)
    else:
      actions = online_net.act(obses, EPSILON_START)

  
  # In the breakout game, we need to call action 1 each time a new game starts
  # to release the projectile from the player's platform.
  # We can help the agent by performing this action for them.
  if FORCE_START:
    actions = [1 if do_init_action[i] else a for i, a in enumerate(actions)]
  
  # We step the environment with the selected actions
  new_obses, rewards, dones, infos = env.step(actions)
  do_init_action = list(dones)

  # We zip together all the info related to the current transition
  # and iterate over the resulting collection.
  # Experiences from all batches are grouped together in a common pool.
  for obs, action, reward, done, new_obs, info in zip(obses, 
                                                actions, 
                                                rewards, 
                                                dones, 
                                                new_obses,
                                                infos):
    # We group all this info in a 'transition' tuple
    # We put the tuple in the replay buffer to accumulate experience
    # If an episode is done the VecEnv will env.reset() for us
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)

  # We set the current observations as past obses for the new cycle
  obses = new_obses

######################
# Main Training Loop #
######################

# After the random-actions loop we reset the environment and start training
obses = env.reset()
do_init_action = [True for _ in range(N_ENVS)]

# Step the loop forward with the itertools.count() int generator
for step in itertools.count():

  # Epsilon decays linearly in time until reaching its final value
  epsilon = np.interp(step * N_ENVS, 
                      [0, EPSILON_DECAY],
                      [EPSILON_START, EPSILON_END])
  
  if isinstance(obses[0], PytorchLazyFrames):
    act_obses = np.stack([o.get_frames() for o in obses])
    actions = online_net.act(act_obses, epsilon)
  else:
    actions = online_net.act(obses, epsilon)
  
  if FORCE_START:
    actions = [1 if do_init_action[i] else a for i, a in enumerate(actions)]
    
  # The training loop goes on as in the random-actions regime
  new_obses, rewards, dones, infos = env.step(actions)
  do_init_action = list(dones)

  for obs, action, reward, done, new_obs, info in zip(obses, 
                                                actions, 
                                                rewards, 
                                                dones, 
                                                new_obses,
                                                infos):
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)

    # When an episode is done we append the episode info to the buffer.
    # We pass the info to TensorBoard for out-of-the-box interactive graphs.
    if done:
      ep_infos_buffer.append(info['episode'])
      episode_count += 1

  obses = new_obses

  # Start Gradient Step
  transitions = random.sample(replay_buffer, BATCH_SIZE)
  # Compute loss
  loss = online_net.compute_loss(transitions, target_net)
  # Gradient Descent
  optimizer.zero_grad()
  # Back-Propagation
  loss.backward()
  optimizer.step()

  # Update Target Network with the online net weights
  if step % TARGET_UPDATE_FREQ == 0:
    target_net.load_state_dict(online_net.state_dict())

  # Logging
  if step % LOG_INTERVAL == 0:
    if isinstance(ep_infos_buffer[0], dict):  
      reward_mean = np.mean([e['r'] for e in ep_infos_buffer]) or 0
      length_mean = np.mean([e['l'] for e in ep_infos_buffer]) or 0
      # Log data to TensorBoard graphs
      summary_writer.add_scalar('AvgRew', reward_mean, global_step=step)
      summary_writer.add_scalar('AvgEpLen', length_mean, global_step=step)
      summary_writer.add_scalar('Episodes', episode_count)
    else:
      reward_mean = 'N/A'
      length_mean = 'N/A'

    print()
    print('Step', step)
    print('Avg Reward', reward_mean)
    print('Avg Episode Length', length_mean)
    print('Episodes', episode_count)

  # Save Network Parameters
  if SAVE_PARAMS:
    if step % SAVE_INTERVAL == 0 and step != 0:
      print('Saving...')
      online_net.save(SAVE_PATH)
