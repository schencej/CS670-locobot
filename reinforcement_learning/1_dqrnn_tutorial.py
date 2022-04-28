#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tqdm import tqdm

display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

num_iterations = 400000
fc_layer_params = (256,256,128,64)

initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 5e-6
log_interval = 5000

num_eval_episodes = 5
eval_interval = 10000

from Discrete_Target_Search import DiscreteTargetSearchEnv
env = DiscreteTargetSearchEnv()

env.reset()
_ = PIL.Image.fromarray(env.render())

train_py_env = DiscreteTargetSearchEnv()
eval_py_env = DiscreteTargetSearchEnv()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

qconf = [
    [128],
    [128,128,128],
    [64]
]
policy_dir = "trained_LSTM_policy"
cpt_dir = "trained_LSTM_cpt"

gru_layers = []
for nunits in qconf[0]:
    gru_layers.append(dense_layer(nunits))
for nunits in qconf[1]:
    gru_layers.append(tf.keras.layers.LSTM(nunits,return_sequences=True,return_state=True))
for nunits in qconf[2]:
    gru_layers.append(dense_layer(nunits))
    
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(gru_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)
agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    rets = []
    for _ in range(num_episodes):

        time_step = environment.reset()
        policy_state = policy.get_initial_state(environment.batch_size)
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step,policy_state)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return
        rets.append(float(episode_return))
    avg_return = sum(rets)/len(rets)
    return avg_return, rets

table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2)

py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)
iterator = iter(dataset)

agent.train = common.function(agent.train)

agent.train_step_counter.assign(0)

avg_return,rets = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [rets]
print(returns)
train_py_env._env.generator.n_obstacles = 0
time_step = train_py_env.reset()
collect_driver = py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)
policy_state = collect_policy.get_initial_state(env.batch_size)

with tqdm(range(num_iterations)) as t:
    for _ in t:
        time_step, policy_state = collect_driver.run(time_step,policy_state)
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()
        if step % log_interval == 0:
            t.set_postfix(step=step,loss=float(train_loss),avgret=float(avg_return),minret=min(rets),maxret=max(rets))

        if step % eval_interval == 0:
            avg_return, rets = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            t.set_postfix(step=step,loss=float(train_loss),avgret=float(avg_return),minret=min(rets),maxret=max(rets))

            returns.append(rets)

with open("RETURNS.txt",'w') as ofh:
    for ret in returns:
        ofh.write(str(np.mean(ret))+'\n')

def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)

def create_policy_eval_video(policy, filename, num_episodes=5, fps=30,good_eps=True):
    
    filename = filename + ".mp4"
    frames = []
    reached = []
    for _ in tqdm(range(num_episodes)):
        time_step = eval_env.reset()
        policy_state = policy.get_initial_state(eval_env.batch_size)
        frames.append([])
        frames[-1].append(eval_py_env.render())
        curr_reached = False
        while not time_step.is_last():
            action_step = policy.action(time_step, policy_state)
            time_step = eval_env.step(action_step.action)
            frames[-1].append(eval_py_env.render())
            if float(time_step.reward) > 0:
                curr_reached = True
                break
        reached.append(curr_reached)
    with imageio.get_writer(filename, fps=fps) as video:
        for i in range(len(reached)):
            for frame in frames[i]:
                video.append_data(frame)
    return embed_mp4(filename)
v = create_policy_eval_video(agent.policy, "trained-agent",num_episodes=30)

from tf_agents.policies import policy_saver
tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)
