import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions (state, action, reward, next_state, done).
    Fixed typos in attribute names, tuple array allocations, and returned values.
    """
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0  # memory is finite
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        # overwrites earlier memory with new memory as the memory fills up
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(keras.Model):
    """
    Critic Neural Network estimating Q(s, a).
    Fixed layer initialization syntax, typos (fc2.dims -> fc2_dims), and missing assignment operators.
    """
    def __init__(self, fc1_dims, fc2_dims, name, chkpt_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        q1_action_value = self.fc1(tf.concat([state, action], axis=1))
        q1_action_value = self.fc2(q1_action_value)
        q = self.q(q1_action_value)

        return q


class ActorNetwork(keras.Model):
    """
    Actor Neural Network estimating deterministic continuous policy mu(s).
    Fixed typos in file path generation and layer parameters.
    """
    def __init__(self, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)

        return mu


class Agent:
    """
    TD3 (Twin Delayed Deep Deterministic Policy Gradient) Agent implementation.
    Ties together networks, replay buffer, exploration, target network updates, and learning steps.
    """
    def __init__(self, alpha, beta, input_dims, tau, env,
                 gamma=0.99, update_actor_interval=2, warmup=1000,
                 max_size=1000000, layer1_size=400, layer2_size=300,
                 batch_size=100, noise=0.1, n_actions=2):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(layer1_size, layer2_size, n_actions=n_actions, name='actor')
        self.critic_1 = CriticNetwork(layer1_size, layer2_size, name='critic_1')
        self.critic_2 = CriticNetwork(layer1_size, layer2_size, name='critic_2')

        self.target_actor = ActorNetwork(layer1_size, layer2_size, n_actions=n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(layer1_size, layer2_size, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(layer1_size, layer2_size, name='target_critic_2')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta))

        self.noise = noise
        # Set target network weights equal to starting values of online networks
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = np.random.normal(scale=self.noise, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            mu = self.actor(state)[0].numpy()  # Returns batch size of 1, converted to numpy array

        mu_prime = mu + np.random.normal(scale=self.noise, size=(self.n_actions,))
        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)

        self.time_step += 1
        return mu_prime.numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(states_)
            # Target policy smoothing noise
            target_actions = target_actions + tf.clip_by_value(
                np.random.normal(scale=0.2, size=target_actions.shape), -0.5, 0.5
            )
            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)

            q1_ = self.target_critic_1(states_, target_actions)
            q2_ = self.target_critic_2(states_, target_actions)

            # Shape is [batch_size, 1], squeeze to collapse to [batch_size]
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)

            critic_value_ = tf.math.minimum(q1_, q2_)

            target = rewards + self.gamma * critic_value_ * (1 - tf.cast(dones, tf.float32))

            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)

        critic_1_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        del tape

        self.learn_step_cntr += 1

        # Delayed policy updates
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Update Target Actor
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        # Update Target Critic 1
        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_1.set_weights(weights)

        # Update Target Critic 2
        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_2.set_weights(weights)

    def save_models(self):
        print('Saving model, please wait...')
        os.makedirs(self.actor.checkpoint_dir, exist_ok=True)
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)

    def load_models(self):
        print('Loading models...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file)