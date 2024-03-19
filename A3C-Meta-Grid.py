#!/usr/bin/env python
# coding: utf-8

# # Meta Reinforcement Learning with A3C - Rainbow Gridworld
# 
# This iPython notebook includes an implementation of the [A3C algorithm capable of Meta-RL](https://arxiv.org/pdf/1611.05763.pdf).
# 
# For more information see the accompanying [Medium post](https://medium.com/p/b15b592a2ddf).

# In[ ]:


import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont

import scipy.signal
get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *

from random import choice
from time import sleep
from time import time
from gridworld import *


# ### Actor-Critic Network
class AC_Network():
    def __init__(self, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # 输入和视觉编码层
            self.state = tf.placeholder(shape=[None, 5, 5, 3], dtype=tf.float32)
            # 输入状态，这里是一个5x5的RGB图像，形状是(None, 5, 5, 3)，None表示可以是任意数量的图像
            self.conv = slim.fully_connected(slim.flatten(self.state), 64, activation_fn=tf.nn.elu)
            # 使用全连接层对输入状态进行编码，输出64维的向量，激活函数为elu

            # 前一个奖励、前一个动作、时间步等输入
            self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            # 上一个时间步的奖励，形状为(None, 1)，None表示可以是任意数量的奖励
            self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)
            # 上一个时间步的动作，形状为(None,)，None表示可以是任意数量的动作
            self.timestep = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            # 时间步，形状为(None, 1)，None表示可以是任意数量的时间步
            self.prev_actions_onehot = tf.one_hot(self.prev_actions, a_size, dtype=tf.float32)
            # 将上一个动作转换为独热编码形式，长度为a_size，表示动作的种类数

            # 将所有输入拼接成一个向量
            hidden = tf.concat([slim.flatten(self.conv), self.prev_rewards, self.prev_actions_onehot, self.timestep], 1)
            # 将上述所有输入向量拼接成一个隐藏层向量

            # 处理时间相关性的循环网络
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(48, state_is_tuple=True)
            # 使用基本的LSTM单元，48表示LSTM单元的大小，即输出维度
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            # 初始化LSTM单元的c状态
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            # 初始化LSTM单元的h状态
            self.state_init = [c_init, h_init]
            # 将c和h状态组合成一个初始化状态
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            # 输入的LSTM单元的c状态,这行代码创建了一个TensorFlow的占位符（placeholder），用于接收LSTM单元的细胞状态。在模型训练过程中，我们会把实际的细胞状态传递给这个占位符，以便在每个时间步更新LSTM单元的状态。
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            # 输入的LSTM单元的h状态
            self.state_in = (c_in, h_in)
            # 将输入状态组合成一个元组
            rnn_in = tf.expand_dims(hidden, [0])
            # 在 hidden 的第一个维度上添加了一个维度
            step_size = tf.shape(self.prev_rewards)[:1]
            # 获取时间步的形状
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            # 将输入状态组合成LSTM状态元组
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            # 使用动态RNN计算LSTM的输出和状态
            lstm_c, lstm_h = lstm_state
            # 分离LSTM的c和h状态
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            # 输出LSTM第一个时间步的c和h状态
            rnn_out = tf.reshape(lstm_outputs, [-1, 48])
            # 将LSTM的输出展平成一维向量

            # 动作输入和输出层
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            # 输入的动作，形状为(None,)，None表示可以是任意数量的动作
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            # 将动作转换为独热编码形式

            # 策略和价值评估的输出层
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            # 输出策略，使用softmax激活函数
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)
            # 输出价值，没有激活函数

            # 只有worker网络需要损失函数和梯度更新操作
            if scope != 'global':
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                # 目标值，形状为(None,)，None表示可以是任意数量的目标值
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
                # 优势，形状为(None,)，None表示可以是任意数量的优势

                # 计算策略损失
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                # 计算动作的输出
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                # 价值损失，使用均方误差
                self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                # 熵正则化，鼓励探索
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7) * self.advantages)
                # 策略损失，使用梯度上升方法
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05
                # 总损失，包括价值损失、策略损失和熵正则化

                # 获取本地网络的梯度
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                # 获取本地网络的可训练变量
                self.gradients = tf.gradients(self.loss, local_vars)
                # 计算损失函数对于本地网络参数的梯度
                self.var_norms = tf.global_norm(local_vars)
                # 计算本地网络的参数的全局范数
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 50.0)
                # 对梯度进行截断，防止梯度爆炸

                # 将本地梯度应用到全局网络
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                # 获取全局网络的可训练变量
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
                # 将截断后的梯度应用到全局网络


# ### Worker Agent
class Worker():
    def __init__(self, game, name, a_size, trainer, model_path, global_episodes):
        # 初始化Worker对象
        # game: 游戏环境
        # name: 工作器的名称
        # a_size: 动作空间的大小
        # trainer: 用于更新参数的优化器
        # model_path: 模型保存路径
        # global_episodes: 全局的训练轮数
        self.name = "worker_" + str(name)  # 设置工作器的名称
        self.number = name  # 设置工作器的编号
        self.model_path = model_path  # 设置模型保存路径
        self.trainer = trainer  # 设置优化器
        self.global_episodes = global_episodes  # 设置全局的训练轮数
        self.increment = self.global_episodes.assign_add(1)  # 每次增加全局的训练轮数
        self.episode_rewards = []  # 记录每个回合的奖励
        self.episode_lengths = []  # 记录每个回合的步数
        self.episode_mean_values = []  # 记录每个回合的状态值的均值
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))  # 用于记录训练日志的写入器

        # 创建本地网络的副本以及将全局参数复制到本地网络的操作
        self.local_AC = AC_Network(a_size, self.name, trainer)  # 创建本地AC网络
        self.update_local_ops = update_target_graph('global', self.name)  # 创建操作，将全局参数复制到本地网络
        self.env = game  # 设置游戏环境

    def train(self, rollout, sess, gamma, bootstrap_value):
        # 对网络进行训练
        # rollout: 回合数据
        # sess: TensorFlow会话
        # gamma: 折扣因子
        # bootstrap_value: 用于计算优势的引导值
        rollout = np.array(rollout)  # 将回合数据转换为NumPy数组
        states = rollout[:, 0]  # 状态序列
        actions = rollout[:, 1]  # 动作序列
        rewards = rollout[:, 2]  # 奖励序列
        timesteps = rollout[:, 3]  # 时间步序列
        prev_rewards = [0] + rewards[:-1].tolist()  # 前一个时间步的奖励
        prev_actions = [0] + actions[:-1].tolist()  # 前一个时间步的动作
        values = rollout[:, 5]  # 状态值

        # 计算回报和优势
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # 更新全局网络的参数
        rnn_state = self.local_AC.state_init
        feed_dict = {
            self.local_AC.target_v: discounted_rewards,
            self.local_AC.state: np.stack(states, axis=0),
            self.local_AC.prev_rewards: np.vstack(prev_rewards),
            self.local_AC.prev_actions: prev_actions,
            self.local_AC.actions: actions,
            self.local_AC.timestep: np.vstack(timesteps),
            self.local_AC.advantages: advantages,
            self.local_AC.state_in[0]: rnn_state[0],
            self.local_AC.state_in[1]: rnn_state[1]
        }
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                                self.local_AC.policy_loss,
                                                self.local_AC.entropy,
                                                self.local_AC.grad_norms,
                                                self.local_AC.var_norms,
                                                self.local_AC.apply_grads],
                                               feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver, train):
        # 工作函数，用于执行训练
        episode_count = sess.run(self.global_episodes)  # 获取当前的全局训练轮数
        total_steps = 0  # 总步数
        print("Starting worker " + str(self.number))  # 打印工作器开始工作
        with sess.as_default(), sess.graph.as_default():  # 设置默认的会话和图
            while not coord.should_stop():  # 当协调器未停止时
                sess.run(self.update_local_ops)  # 更新本地网络参数
                episode_buffer = []  # 初始化回合缓存
                episode_values = []  # 初始化回合的状态值
                episode_frames = []  # 初始化回合的帧
                episode_reward = 0  # 初始化回合奖励
                episode_step_count = 0  # 初始化回合步数
                d = False  # 初始化回合结束标志
                r = 0  # 初始化当前时间步的奖励
                a = 0  # 初始化动作
                t = 0  # 初始化时间步
                reward_color = [np.random.uniform(), np.random.uniform(), np.random.uniform()]  # 随机生成奖励颜色
                # reward_color = [1,0,0]
                s, s_big = self.env.reset(reward_color)  # 重置游戏环境并获取初始状态
                rnn_state = self.local_AC.state_init  # 初始化RNN状态

                while not d:  # 当回合未结束时
                    # 使用策略网络输出的概率采样动作
                    a_dist, v, rnn_state_new = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                                         feed_dict={
                                                             self.local_AC.state: [s],
                                                             self.local_AC.prev_rewards: [[r]],
                                                             self.local_AC.timestep: [[t]],
                                                             self.local_AC.prev_actions: [a],
                                                             self.local_AC.state_in[0]: rnn_state[0],
                                                             self.local_AC.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])  # 根据概率选择动作
                    a = np.argmax(a_dist == a)  # 获取动作索引

                    rnn_state = rnn_state_new  # 更新RNN状态
                    s1, s1_big, r, d, _, _ = self.env.step(a)  # 执行动作，获取下一个状态、奖励等信息

                    # 记录回合数据
                    episode_buffer.append([s, a, r, t, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    episode_frames.append(set_image_gridworld(s1_big, reward_color, episode_reward, t))  # 将当前状态的图像添加到回合帧列表中
                    total_steps += 1
                    t += 1
                    episode_step_count += 1
                    s = s1  # 更新状态

                    if episode_step_count > 100:  # 如果回合步数超过100步，则结束回合
                        d = True

                # 记录回合奖励、步数和状态值
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # 使用回合数据训练网络
                if len(episode_buffer) != 0 and train:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # 每隔一定回合数保存模型、生成gif动画和记录日志
                if episode_count % 50 == 0 and episode_count != 0:
                    if episode_count % 500 == 0 and self.name == 'worker_0' and train:
                        saver.save(sess, self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print("Saved Model")

                    if self.name == 'worker_0' and episode_count % 50 == 0:
                        time_per_step = 0.25
                        self.images = np.array(episode_frames)
                        make_gif(self.images, './frames/image'+str(episode_count)+'.gif',
                                 duration=len(self.images)*time_per_step, true_image=True)

                    # 计算近50个回合的奖励、步数和状态值的均值，并记录到日志中
                    mean_reward = np.mean(self.episode_rewards[-50:])
                    mean_length = np.mean(self.episode_lengths[-50:])
                    mean_value = np.mean(self.episode_mean_values[-50:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    if train:
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)  # 增加全局训练轮数
                episode_count += 1  # 增加当前回合数


gamma = .95  # 折扣率，用于优势估计和折扣奖励
a_size = 4  # 动作空间的大小
load_model = True  # 是否加载已有模型
train = False  # 是否进行训练
model_path = './model_meta_grid'  # 模型保存路径

# 重置 TensorFlow 默认图
tf.reset_default_graph()

# 如果模型保存路径不存在，则创建该路径
if not os.path.exists(model_path):
    os.makedirs(model_path)

# 如果 './frames' 路径不存在，则创建该路径
if not os.path.exists('./frames'):
    os.makedirs('./frames')

# 将全局回合数数设定为不可训练的整数型变量，并将其初始化为 0
with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    # 使用 Adam 优化器
    trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
    # 创建全局网络
    master_network = AC_Network(a_size, 'global', None)
    # 设置工作进程数量为可用 CPU 线程数
    num_workers = multiprocessing.cpu_count()  # 获取可用的 CPU 核心数
    workers = []  # 创建一个空列表，用于存储 Worker 类的实例对象

    # 循环创建多个工作类实例
    for i in range(num_workers):
        # 创建 Worker 类的实例，并将其添加到 workers 列表中
        # gameEnv(partial=False, size=5, goal_color=[np.random.uniform(), np.random.uniform(), np.random.uniform()]) 用于创建游戏环境实例
        # i 是工作线程的编号
        # a_size 是动作空间的大小
        # trainer 是用于优化的训练器实例
        # model_path 是模型保存路径
        # global_episodes 是全局剧集数
        workers.append(Worker(
            gameEnv(partial=False, size=5, goal_color=[np.random.uniform(), np.random.uniform(), np.random.uniform()]),
            i, a_size, trainer, model_path, global_episodes))

    saver = tf.train.Saver(max_to_keep=5)  # 创建一个 Saver 实例，用于保存模型

with tf.Session() as sess:
    coord = tf.train.Coordinator()  # 创建一个 Coordinator 实例，用于协调多个线程
    if load_model == True:  # 如果设置了 load_model 为 True，则加载模型
        print('Loading Model...')  # 输出提示信息，表示正在加载模型
        ckpt = tf.train.get_checkpoint_state(model_path)  # 获取模型检查点状态
        saver.restore(sess, ckpt.model_checkpoint_path)  # 恢复模型参数
    else:
        sess.run(tf.global_variables_initializer())  # 否则，初始化所有模型参数

    worker_threads = []  # 创建一个空列表，用于存储工作线程
    for worker in workers:
        # 启动工作线程
        worker_work = lambda: worker.work(gamma, sess, coord, saver, train)  # 创建一个 lambda 函数，用于调用 Worker 类的 work 方法
        thread = threading.Thread(target=(worker_work))  # 创建一个线程对象，目标函数为 worker_work
        thread.start()  # 启动线程
        worker_threads.append(thread)  # 将线程对象添加到 worker_threads 列表中

    coord.join(worker_threads)  # 等待所有工作线程结束，防止主线程提前结束



