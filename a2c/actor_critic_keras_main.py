from actor_critic_keras import Actor_Critic
import environment
import agent
import data_manager
import settings
import os
import numpy as np

def main(stock_code):
    initial_balance = 10000000
    start_time = None
    end_time = None
    reward_threshold = .05
    chart_data, training_data = data_manager.load_data(
        os.path.join(settings.BASE_DIR, 'data/{}/{}_data.txt'.format('a1', stock_code)), ver='a1')
    age = agent.Agent(environment=environment.Environment(chart_data, training_data), min_trading_unit=1,
                      max_trading_unit=max(int(initial_balance / chart_data.iloc[-1]['close']), 1),
                      delayed_reward_threshold=reward_threshold, importance=0.9)
    age.set_balance(initial_balance)
    num_features = age.STATE_DIM
    input_len = age.environment.get_training_data_shape()
    if input_len != 0:
        num_features += input_len  # traing_data만 상태임
    num_actions = age.NUM_ACTIONS
    ac = Actor_Critic(alpha=0.001, beta=0.005, gamma=0.9, n_actions=num_actions, input_dims=num_features)
    score_history = []
    num_episodes = 1000
    print_interval = 100
    start_epsilon = 0.5
    score = 0.
    for n_epi in range(num_episodes):
        done = False
        age.environment.reset()
        age.reset()
        sample = age.build_sample()
        if sample is None:
            done = True
        else:
            done = False
        s = np.array(sample)
        epsilon = start_epsilon * (1. - float(n_epi) / (num_episodes - 1))
        while not done:
            action, prob_action_list = ac.choose_action(s)
            if np.random.rand() < epsilon:
                action = np.random.randint(age.NUM_ACTIONS)
            # action = ac.choose_action(s)
            s_prime, reward, done = age.noholdstep(action, prob_action_list[action])
            s_prime = np.array(s_prime)
            if done:
                break
            ac.learn(s, action, reward, s_prime, done)
            s = s_prime
            score += reward
        if n_epi % print_interval == 0 and n_epi != 0:
            print("episode :{} score : {:.4f} buy :{} sell : {} hold : {}".format(n_epi, score / print_interval,age.num_buy / print_interval, age.num_sell / print_interval, age.num_hold / print_interval))
            score = 0.0

    network_name = stock_code + '_ac_model_' + str(num_episodes)
    actorPATH = os.path.join(settings.BASE_DIR, 'models/{}_actor.h5'.format(network_name))
    criticPATH = os.path.join(settings.BASE_DIR, 'models/{}_critic.h5'.format(network_name))
    policyPATH = os.path.join(settings.BASE_DIR, 'models/{}_policy.h5'.format(network_name))
    ac.save_model(actor_path=actorPATH, critic_path=criticPATH, policy_path=policyPATH)

def test(stock_code, extraword):
    network_name = stock_code + '_ac_model_' + str(extraword)
    actorPATH = os.path.join(settings.BASE_DIR, 'models/{}_actor.h5'.format(network_name))
    criticPATH = os.path.join(settings.BASE_DIR, 'models/{}_critic.h5'.format(network_name))
    policyPATH = os.path.join(settings.BASE_DIR, 'models/{}_policy.h5'.format(network_name))
    initial_balance = 10000000
    start_time = None
    end_time = None
    reward_threshold = .05
    chart_data, training_data = data_manager.load_data(
        os.path.join(settings.BASE_DIR, 'data/{}/{}_data.txt'.format('a1', stock_code)), ver='a1')
    age = agent.Agent(environment=environment.Environment(chart_data, training_data), min_trading_unit=1,
                      max_trading_unit=max(int(initial_balance / chart_data.iloc[-1]['close']), 1),
                      delayed_reward_threshold=reward_threshold, importance=0.9)
    age.set_balance(initial_balance)
    num_features = age.STATE_DIM
    input_len = age.environment.get_training_data_shape()
    if input_len != 0:
        num_features += input_len  # traing_data만 상태임
    num_actions = age.NUM_ACTIONS
    ac = Actor_Critic(alpha=0.001, beta=0.005, gamma=0.9, n_actions=num_actions, input_dims=num_features)
    num_episodes = 10
    print_interval = 10
    start_epsilon = 0.5
    for n_epi in range(num_episodes):
        done = False
        age.environment.reset()
        age.reset()
        sample = age.build_sample()
        if sample is None:
            done = True
        else:
            done = False
        s = np.array(sample)
        epsilon = start_epsilon * (1. - float(n_epi) / (num_episodes - 1))
        while not done:
            action, prob_action_list = ac.choose_action(s)
            # action = ac.choose_action(s)
            s_prime, reward, done = age.noholdstep(action, prob_action_list[action])
            s_prime = np.array(s_prime)
            if done:
                print("episode : {} pv : {:.1f} buy :{} sell : {} hold : {} stocks : {}".format(n_epi, age.portfolio_value, age.num_buy, age.num_sell, age.num_hold, age.num_stocks))
                break
            #ac.learn(s, action, reward, s_prime, done)
            s = s_prime

if __name__ == '__main__':
    main('000660')
    #test('000660', '1000')
