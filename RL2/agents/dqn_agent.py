# ---------- agents/dqn_agent.py ----------
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, Flatten, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_shape, action_size, config):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=config['training']['memory_size'])
        self.gamma = config['training']['gamma']
        self.epsilon = config['training']['epsilon_start']
        self.epsilon_min = config['training']['epsilon_end']
        self.epsilon_decay = config['training']['epsilon_decay']
        self.batch_size = config['training']['batch_size']
        self.target_update = config['training']['target_update']
        self.model = self._build_model(config)
        self.target_model = self._build_model(config)
        self.target_model.set_weights(self.model.get_weights())
        self.update_count = 0
        self.optimizer = Adam(learning_rate=config['training'].get('learning_rate', 1e-4))

    def _build_model(self, config):
        # Dual-stream Dueling DQN without no-op attention
        seq_input = Input(shape=self.state_shape)

        # Market (technical) stream
        market_input = Lambda(lambda x: x[:, :, :-6])(seq_input)
        tech = Conv1D(filters=config['model']['cnn_filters'][0], kernel_size=3, activation='relu')(market_input)
        tech = Conv1D(filters=config['model']['cnn_filters'][1], kernel_size=2, activation='relu')(tech)
        tech = Flatten()(tech)

        # Position context stream
        pos_input = Lambda(lambda x: x[:, :, -6:])(seq_input)
        pos = LSTM(config['model']['lstm_units'], return_sequences=True)(pos_input)
        pos = LSTM(config['model']['lstm_units'])(pos)

        # Merge streams
        merged = Concatenate()([tech, pos])
        dense = Dense(config['model']['dense_units'], activation='relu')(merged)

        # Dueling architecture
        if config['model']['dueling']:
            val = Dense(1)(dense)
            adv = Dense(self.action_size)(dense)
            q_vals = val + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))
        else:
            q_vals = Dense(self.action_size)(dense)

        return Model(inputs=seq_input, outputs=q_vals)

    @tf.function
    def _predict(self, state):
        return self.model(state, training=False)

    @tf.function
    def _train_step(self, states, targets):
        with tf.GradientTape() as tape:
            preds = self.model(states, training=True)
            loss = tf.reduce_mean(tf.square(targets - preds))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = np.expand_dims(state, 0)
        q = self._predict(tf.convert_to_tensor(state, dtype=tf.float32))[0]
        return int(tf.argmax(q).numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.bool)

        # Online vs. target models
        next_q_online = self.model(next_states, training=False)
        next_q_target = self.target_model(next_states, training=False)

        current_q = self.model(states, training=False).numpy()

        # --- Fix for dtype mismatch here ---
        next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
        batch_idx    = tf.range(self.batch_size, dtype=tf.int32)
        indices      = tf.stack([batch_idx, next_actions], axis=1)
        max_next_q   = tf.gather_nd(next_q_target, indices)
        # ------------------------------------

        # Build target Q-values
        for i in range(self.batch_size):
            if dones[i]:
                current_q[i, actions[i]] = rewards[i]
            else:
                current_q[i, actions[i]] = rewards[i] + self.gamma * max_next_q[i]

        # Train on updated targets
        self._train_step(states, tf.convert_to_tensor(current_q, dtype=tf.float32))

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periodically update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_model.set_weights(self.model.get_weights())

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)
        self.target_model.set_weights(self.model.get_weights())
