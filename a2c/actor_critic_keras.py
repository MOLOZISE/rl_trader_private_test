from keras import backend as K
from keras.layers import Dense, Input, LSTM, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

class Actor_Critic(object):
    def __init__(self, alpha, beta, gamma=0.9, n_actions=3, input_dims=32, actor_path=None, critic_path=None, policy_path=None):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.imput_dims = input_dims
        self.n_actions = n_actions
        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        if actor_path and critic_path and policy_path is not None:
            self.actor, self.critic, self.policy = self.load_model(actor_path=actor_path, critic_path=critic_path, policy_path=policy_path)
        self.action_space = [i for i in range(self.n_actions)]

    def build_actor_critic_network(self):
        input = Input(shape=(self.imput_dims,))
        delta = Input(shape=[1])
        output = Dense(256, activation='elu', kernel_initializer='he_normal')(input)
        output = Dropout(0.2)(output)
        output = Dense(128, activation='elu', kernel_initializer='he_normal')(output)
        output = Dropout(0.2)(output)
        output = Dense(64, activation='elu', kernel_initializer='he_normal')(output)
        output = Dropout(0.2)(output)
        output = Dense(32, activation='elu', kernel_initializer='he_normal')(output)
        output = Dropout(0.2)(output)
        output = Dense(16, activation='elu', kernel_initializer='he_normal')(output)
        output = Dropout(0.2)(output)
        probs = Dense(self.n_actions, activation='softmax')(output)
        values = Dense(1, activation='linear')(output)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)

        actor = Model(input=[input, delta], output=[probs])

        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        critic = Model(input=[input], output=[values])

        critic.compile(optimizer=Adam(lr=self.beta), loss='mse')

        policy = Model(input=[input], output=[probs])

        return actor, critic, policy

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        prob_action_list = probabilities
        return action, prob_action_list

    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]

        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1.0

        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0)

    # actor, critic, policy 저장필요
    def save_model(self, actor_path=None, critic_path=None, policy_path=None):
        if actor_path and critic_path and policy_path is not None:
            self.actor.save_weights(actor_path, overwrite=True)
            self.critic.save_weights(critic_path, overwrite=True)
            self.policy.save_weights(policy_path, overwrite=True)

    def load_model(self, actor_path=None, critic_path=None, policy_path=None):
        if actor_path and critic_path and policy_path is not None:
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            self.policy.load_weights(policy_path)