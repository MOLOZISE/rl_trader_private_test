import sys
import environment
import agent
import data_manager
import settings
import os
import numpy as np
from ppo_keras import PPOIQN
import copy
import datetime
import random

# 케라스가 느린 이유?

import keras.backend.tensorflow_backend
import tensorflow as tf

from keras.backend import clear_session

# change f_data to five_data
def main(stock_code, model_name, target_date, actor_path=None, value_path=None, num_episodes=100):
    best_score = None
    initial_balance = 10000000 # threemin, fivemin
    chart_data, training_data = data_manager.load_data(
    #    os.path.join(settings.BASE_DIR, 'data/{}/{}_data.txt'.format('a1', stock_code)), ver='a1')
    #    os.path.join(settings.BASE_DIR, 'data/{}/{}_threemin_data.txt'.format('t1', stock_code)), ver='f1')
    #    os.path.join(settings.BASE_DIR, 'data/{}/{}/{}_{}_fivemin_data.txt'.format('f1', stock_code, stock_code, target_date)), ver='f1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code)), ver='u1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}_z.txt'.format(target_date, stock_code)), ver='z1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code)), ver='x1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code)), ver='y1')
        os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code)), ver='y2')
    age = agent.Agent(environment=environment.Environment(chart_data, training_data), min_trading_unit=1,
                      max_trading_unit=2, delayed_reward_threshold=.05, importance=0.9)
    age.set_balance(initial_balance)
    num_features = age.STATE_DIM
    input_len = age.environment.get_training_data_shape()
    if input_len != 0:
        num_features += input_len  # traing_data만 상태임
    num_actions = age.NUM_ACTIONS
    PPO = PPOIQN(num_state=num_features, num_action=num_actions, actor_path=actor_path, value_path=value_path)
    PPO.assign_policy_parameters()
    score = 0.0
    print_interval = 100
    start_epsilon = 0.5
    #num_episodes = 100
    age_num_buy = 0
    age_num_sell = 0
    age_num_hold = 0
    age_pv = 0.0
    for episodes in range(num_episodes): # 스타크래프트 2 1백만
        # 관측, 행동 리스트, 가치 예측, 보상 리스트 초기화
        observations = []
        actions_list = []
        v_preds = []
        rewards = []
        age.environment.reset()
        age.reset()
        state = age.build_sample()
        if state is None:
            done = True
        else:
            done = False
        while not done:
            state = np.stack([state]).astype(dtype=np.float32)
            # 상태에 대한 행동 및 가치 예측을 얻음 + 스칼라로 변환
            act, v_pred = PPO.get_action(state)  # 상태에 따른 정책을 얻음
            act, v_pred = act.item(), v_pred.item()  # 행동을 스칼라로, 가치를 스칼라로
            s_prime, reward, done = age.newstep(act)
            observations.append(state)
            actions_list.append(act)
            v_preds.append(v_pred)
            rewards.append(reward)
            if done:
                # 다음 예측은 예측의 2번째(1번째 인덱스 부터 마지막은 0
                v_preds_next = v_preds[1:] + [0]
                # gaes 계산 = PPO의 get_gaes
                gaes = PPO.get_gaes(rewards, v_preds, v_preds_next)
                # reshape
                observations = np.reshape(observations, [-1, num_features]) # 왜 이럼??
                #observations = np.array(observations).astype(dtype=np.float32)
                # action은 인트, 나머지는 플롯
                actions_list = np.array(actions_list).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)

                # 올드한 놈들을 최신 놈들로 바꿔줌

                PPO.assign_policy_parameters()

                # input = [상태들, 행동리스트, 보상리스트, 다음 예측 리스트, gaes]

                inp = [observations, actions_list, rewards, v_preds_next, gaes]
                # 왜 4임????
                for epoch in range(4):
                    # 왜 32임??? 0 ~ 관찰행전부까지의 랜덤 난수를 size 만큼 생성
                    sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)  # indices are in [low, high)
                    # 각 input에 대하여 take함
                    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                    # PPO.train을 각 상태, 행동, 보상, 다음 예측, gaes으로 실행
                    PPO.train(obs=sampled_inp[0],
                              actions=sampled_inp[1],
                              rewards=sampled_inp[2],
                              v_preds_next=sampled_inp[3],
                              gaes=sampled_inp[4])
                score = sum(rewards)
            state = s_prime
        age_num_buy += age.num_buy
        age_num_sell += age.num_sell
        age_num_hold += age.num_hold
        age_pv += age.portfolio_value
        if (episodes + 1) % print_interval == 0:
            print("# of episode :{} score : {:.4f} pv : {:.2f} buy :{:.2f} sell : {:.2f} hold : {:.2f}".format(episodes + 1 , score / print_interval, age_pv / print_interval, age_num_buy / print_interval, age_num_sell / print_interval, age_num_hold / print_interval))
            if best_score is None:
                best_score = score
            if score >= best_score:
                best_score = score
                # date로 할 때에는 stock_code 대신 target_date를 사용
                network_name_actor = target_date + '_ppo_model_' + model_name + '_actor'
                network_name_value = target_date + '_ppo_model_' + model_name + '_value'
                path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
                path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
                PPO.save_model(actor_path=path_actor, value_path=path_value)
            age_num_buy = 0
            age_num_sell = 0
            age_num_hold = 0
            age_pv = 0.0
            score = 0.0
    clear_session()
    # if keras.backend.tensorflow_backend._SESSION:
    #     tf.reset_default_graph()
    #     keras.backend.tensorflow_backend._SESSION.close()
    #     keras.backend.tensorflow_backend._SESSION = None

def relearn(stock_code1, stock_code2, model_name, new_model_name, target_date ,actor_path=None, value_path=None, num_episodes=100):
    #stock_code2는 before_target_date와 동일
    best_score = None
    initial_balance = 10000000 # threemin, fivemin
    chart_data, training_data = data_manager.load_data(
    #    os.path.join(settings.BASE_DIR, 'data/{}/{}_data.txt'.format('a1', stock_code1)), ver='a1')
    #    os.path.join(settings.BASE_DIR, 'data/{}/{}_threemin_data.txt'.format('t1', stock_code1)), ver='f1')
    #    os.path.join(settings.BASE_DIR, 'data/{}/{}/{}_{}_fivemin_data.txt'.format('f1', stock_code1, stock_code1, target_date)), ver='f1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code1)), ver='u1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}_z.txt'.format(target_date, stock_code1)), ver='z1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code1)), ver='x1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code1)), ver='y1')
        os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code1)), ver='y2')
    age = agent.Agent(environment=environment.Environment(chart_data, training_data), min_trading_unit=1,
                      max_trading_unit=2, delayed_reward_threshold=.05, importance=0.9)
    age.set_balance(initial_balance)
    num_features = age.STATE_DIM
    input_len = age.environment.get_training_data_shape()
    if input_len != 0:
        num_features += input_len  # traing_data만 상태임
    num_actions = age.NUM_ACTIONS
    network_name_actor = stock_code2 + '_ppo_model_' + model_name + '_actor'
    network_name_value = stock_code2 + '_ppo_model_' + model_name + '_value'
    path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
    path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
    PPO = PPOIQN(num_state=num_features, num_action=num_actions, actor_path=path_actor, value_path=path_value)
    PPO.assign_policy_parameters()
    score = 0.0
    print_interval = 100
    start_epsilon = 0.5
    #num_episodes = 100
    age_num_buy = 0
    age_num_sell = 0
    age_num_hold = 0
    age_pv = 0.0
    for episodes in range(num_episodes): # 스타크래프트 2 1백만
        # 관측, 행동 리스트, 가치 예측, 보상 리스트 초기화
        observations = []
        actions_list = []
        v_preds = []
        rewards = []
        age.environment.reset()
        age.reset()
        state = age.build_sample()
        if state is None:
            done = True
        else:
            done = False
        while not done:
            state = np.stack([state]).astype(dtype=np.float32)
            # 상태에 대한 행동 및 가치 예측을 얻음 + 스칼라로 변환
            act, v_pred = PPO.get_action(state)  # 상태에 따른 정책을 얻음
            act, v_pred = act.item(), v_pred.item()  # 행동을 스칼라로, 가치를 스칼라로
            s_prime, reward, done = age.newstep(act)
            observations.append(state)
            actions_list.append(act)
            v_preds.append(v_pred)
            rewards.append(reward)
            if done:
                # 다음 예측은 예측의 2번째(1번째 인덱스 부터 마지막은 0
                v_preds_next = v_preds[1:] + [0]
                # gaes 계산 = PPO의 get_gaes
                gaes = PPO.get_gaes(rewards, v_preds, v_preds_next)
                # reshape
                observations = np.reshape(observations, [-1, num_features]) # 왜 이럼??
                #observations = np.array(observations).astype(dtype=np.float32)
                # action은 인트, 나머지는 플롯
                actions_list = np.array(actions_list).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)

                # 올드한 놈들을 최신 놈들로 바꿔줌
                PPO.assign_policy_parameters()
                # input = [상태들, 행동리스트, 보상리스트, 다음 예측 리스트, gaes]
                inp = [observations, actions_list, rewards, v_preds_next, gaes]
                # 왜 4임????
                for epoch in range(4):
                    # 왜 32임??? 0 ~ 관찰행전부까지의 랜덤 난수를 size 만큼 생성
                    sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)  # indices are in [low, high)
                    # 각 input에 대하여 take함
                    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                    # PPO.train을 각 상태, 행동, 보상, 다음 예측, gaes으로 실행
                    PPO.train(obs=sampled_inp[0],
                              actions=sampled_inp[1],
                              rewards=sampled_inp[2],
                              v_preds_next=sampled_inp[3],
                              gaes=sampled_inp[4])
                score = sum(rewards)
            state = s_prime
        age_num_buy += age.num_buy
        age_num_sell += age.num_sell
        age_num_hold += age.num_hold
        age_pv += age.portfolio_value
        if (episodes + 1) % print_interval == 0:
            print("# of episode :{} score : {:.4f} pv : {:.2f} buy :{:.2f} sell : {:.2f} hold : {:.2f}".format(episodes + 1 , score / print_interval, age_pv / print_interval, age_num_buy / print_interval, age_num_sell / print_interval, age_num_hold / print_interval))
            if best_score is None:
                best_score = score
            if score >= best_score:
                best_score = score
                # date로 할 때에는 stock_code1 대신 target_date를 사용
                network_name_actor = target_date + '_ppo_model_' + new_model_name + '_actor'
                network_name_value = target_date + '_ppo_model_' + new_model_name + '_value'
                path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
                path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
                PPO.save_model(actor_path=path_actor, value_path=path_value)
            age_num_buy = 0
            age_num_sell = 0
            age_num_hold = 0
            age_pv = 0.0
            score = 0.0
    clear_session()
    # if keras.backend.tensorflow_backend._SESSION:
    #     tf.reset_default_graph()
    #     keras.backend.tensorflow_backend._SESSION.close()
    #     keras.backend.tensorflow_backend._SESSION = None

def test(stock_code1, stock_code2, model_name, target_date, actor_path=None, value_path=None):
    best_score = 0.
    initial_balance = 10000000
    chart_data, training_data = data_manager.load_data(
    #    os.path.join(settings.BASE_DIR, 'data/{}/{}_data.txt'.format('a1', stock_code1)), ver='a1')
    #    os.path.join(settings.BASE_DIR, 'data/{}/{}_threemin_data.txt'.format('t1', stock_code1)), ver='f1')
    #    os.path.join(settings.BASE_DIR, 'data/{}/{}/{}_{}_fivemin_data.txt'.format('f1', stock_code1, stock_code1, target_date)), ver='f1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code1)), ver='u1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}_z.txt'.format(target_date, stock_code1)), ver='z1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code1)), ver='x1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code1)), ver='y1')
        os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code1)), ver='y2')
    age = agent.Agent(environment=environment.Environment(chart_data, training_data), min_trading_unit=1,
                      max_trading_unit=2, delayed_reward_threshold=.05, importance=0.9)
    age.set_balance(initial_balance)
    num_features = age.STATE_DIM
    input_len = age.environment.get_training_data_shape()
    if input_len != 0:
        num_features += input_len  # traing_data만 상태임
    num_actions = age.NUM_ACTIONS
    network_name_actor = stock_code2 + '_ppo_model_' + model_name + '_actor'
    network_name_value = stock_code2 + '_ppo_model_' + model_name + '_value'
    path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
    path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
    PPO = PPOIQN(num_state=num_features, num_action=num_actions, actor_path=path_actor, value_path=path_value, testing=True)
    #PPO.assign_policy_parameters()
    score = 0.0
    total_score = 0.0
    total_profit = 0.0
    print_interval = 100
    start_epsilon = 0.5
    num_episodes = 10
    for episodes in range(num_episodes): # 스타크래프트 2 1백만
        # 관측, 행동 리스트, 가치 예측, 보상 리스트 초기화
        observations = []
        actions_list = []
        #v_preds = []
        rewards = []
        age.environment.reset()
        age.reset()
        state = age.build_sample()
        if state is None:
            done = True
        else:
            done = False
        while not done:
            state = np.stack([state]).astype(dtype=np.float32)
            # 상태에 대한 행동 및 가치 예측을 얻음 + 스칼라로 변환
            act = PPO.give_action(state)  # 상태에 따른 정책을 얻음
            act = act.item() #v_pred.item()  # 행동을 스칼라로, 가치를 스칼라로
            s_prime, reward, done = age.newstep(act)
            observations.append(state)
            actions_list.append(act)
            #v_preds.append(v_pred)
            rewards.append(reward)
            if done:
                score = sum(rewards)
            state = s_prime
        print("# of episode :{} score : {:.4f} pv : {:.4f} buy :{} sell : {} hold : {} stocks : {}".format(episodes + 1, score, age.portfolio_value, age.num_buy, age.num_sell, age.num_hold, age.num_stocks))
        total_score += score
        profit = age.portfolio_value - age.initial_balance
        total_profit += profit
        score = 0.0
    # after finish 10times of episodes, total score will be model's performance
    avg_score = total_score / num_episodes
    avg_profit = total_profit / num_episodes
    clear_session()
    # if keras.backend.tensorflow_backend._SESSION:
    #     tf.reset_default_graph()
    #     keras.backend.tensorflow_backend._SESSION.close()
    #     keras.backend.tensorflow_backend._SESSION = None
    return (avg_score, avg_profit)

def test_randmodel(stock_code1, stock_code2, model_name, target_date, actor_path=None, value_path=None):
    best_score = 0.
    initial_balance = 10000000
    chart_data, training_data = data_manager.load_data(
    #    os.path.join(settings.BASE_DIR, 'data/{}/{}_data.txt'.format('a1', stock_code1)), ver='a1')
    #    os.path.join(settings.BASE_DIR, 'data/{}/{}_threemin_data.txt'.format('t1', stock_code1)), ver='f1')
    #    os.path.join(settings.BASE_DIR, 'data/{}/{}/{}_{}_fivemin_data.txt'.format('f1', stock_code1, stock_code1, target_date)), ver='f1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code1)), ver='u1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}_z.txt'.format(target_date, stock_code1)), ver='z1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code1)), ver='x1')
    #    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code1)), ver='y1')
        os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code1)), ver='y2')
    age = agent.Agent(environment=environment.Environment(chart_data, training_data), min_trading_unit=1,
                      max_trading_unit=2, delayed_reward_threshold=.05, importance=0.9)
    age.set_balance(initial_balance)
    num_features = age.STATE_DIM
    input_len = age.environment.get_training_data_shape()
    if input_len != 0:
        num_features += input_len  # traing_data만 상태임
    num_actions = age.NUM_ACTIONS
    network_name_actor = model_name + '_actor'
    network_name_value = model_name + '_value'
    path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
    path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
    PPO = PPOIQN(num_state=num_features, num_action=num_actions, actor_path=path_actor, value_path=path_value, testing=True)
    #PPO.assign_policy_parameters()
    score = 0.0
    total_score = 0.0
    total_profit = 0.0
    print_interval = 100
    start_epsilon = 0.5
    num_episodes = 10
    for episodes in range(num_episodes): # 스타크래프트 2 1백만
        # 관측, 행동 리스트, 가치 예측, 보상 리스트 초기화
        observations = []
        actions_list = []
        #v_preds = []
        rewards = []
        age.environment.reset()
        age.reset()
        state = age.build_sample()
        if state is None:
            done = True
        else:
            done = False
        while not done:
            state = np.stack([state]).astype(dtype=np.float32)
            # 상태에 대한 행동 및 가치 예측을 얻음 + 스칼라로 변환
            act = PPO.give_action(state)  # 상태에 따른 정책을 얻음
            act = act.item() #v_pred.item()  # 행동을 스칼라로, 가치를 스칼라로
            s_prime, reward, done = age.newstep(act)
            observations.append(state)
            actions_list.append(act)
            #v_preds.append(v_pred)
            rewards.append(reward)
            if done:
                score = sum(rewards)
            state = s_prime
        print("# of episode :{} score : {:.4f} pv : {:.4f} buy :{} sell : {} hold : {} stocks : {}".format(episodes + 1, score, age.portfolio_value, age.num_buy, age.num_sell, age.num_hold, age.num_stocks))
        total_score += score
        profit = age.portfolio_value - age.initial_balance
        total_profit += profit
        score = 0.0
    # after finish 10times of episodes, total score will be model's performance
    avg_score = total_score / num_episodes
    avg_profit = total_profit / num_episodes
    clear_session()
    # if keras.backend.tensorflow_backend._SESSION:
    #     tf.reset_default_graph()
    #     keras.backend.tensorflow_backend._SESSION.close()
    #     keras.backend.tensorflow_backend._SESSION = None
    return (avg_score, avg_profit)

def autolearning(start_date="20200601", end_date="20200710", reverse=False):
    code_list = []
    learn_code_list = []
    file = open("files/beforedata/bigvolumedata.txt", "r", encoding="utf8")
    lines = file.readlines()
    for line in lines:
        code_list.append(line.strip())
    file.close()
    lenlist = len(code_list)
    if reverse:
        code_list = reversed(code_list)
    # 시간 for문 저장
    d1 = datetime.date(int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8]))
    d2 = datetime.date(int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8]))
    delta = d2 - d1
    # code_list에는 대상 종목 코드들이 들어가 있음
    # 하나씩 하기 위해서 0. beforedata에 원하는 종목 코드들(or 1개)를 적어둠 1. count = 만들 모델 번호 2. before_code 전 모델의 종목코드
    #  3. before_model_name 전 모델의 이름 4. tempcount 2이상
    count = 1
    tempcount = 1
    before_code = ""
    before_model_name = ""
    # 각 종목코드별로
    for code in code_list:
        if count < 10:
            count_str = "0" + str(count)
        else:
            count_str = str(count)
        if reverse:
            model_name = "r_five_" + count_str
        else:
            model_name = "five_" + count_str
        print("current model : %s" % model_name)
        print("current count : %s / %s" % (str(count), lenlist))
        # 시작 날짜 ~ 끝 날짜(포함)까지 for문
        for i in range(delta.days + 1):
            target_date = str(d1 + datetime.timedelta(days=i)).replace('-', '')
            print("current date : %s" % target_date)
            filepath = os.path.join(settings.BASE_DIR, 'data/{}/{}/{}_{}_fivemin_data.txt'.format('f1', code, code, target_date))
            if not os.path.isfile(filepath):
                continue
            if tempcount == 1:
                main(code, model_name, target_date)
            else:
                relearn(code, before_code, before_model_name, model_name, target_date)
            tempcount += 1
            before_code = code
            before_model_name = model_name
        count += 1

def autoverifying(start_date="20200704", end_date="20200710", reverse=False):
    code_list = []
    very_code_list = []
    best_score = 0.0
    best_model_name = ""
    model_score_dict = {}
    # code_list에는 배웠던 종목들을
    file = open("files/beforedata/bigvolumedata.txt", "r", encoding="utf8")
    lines = file.readlines()
    for line in lines:
        code_list.append(line.strip())
    file.close()
    # very_code_list에는 검증할 종목들을 전달
    # week1의 bigvolumedata에는 새로운(검증할) 종목 코드가 있음
    file = open("files/afterdata/bigvolumedata.txt", "r", encoding="utf8")
    lines = file.readlines()
    for line in lines:
        very_code_list.append(line.strip())
    file.close()
    #lenlist = len(code_list)
    verlist = len(very_code_list)
    if reverse:
        code_list = reversed(code_list)
    count = 1
    abs_code_list = copy.deepcopy(very_code_list)
    # 시간 for문 저장
    d1 = datetime.date(int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8]))
    d2 = datetime.date(int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8]))
    delta = d2 - d1
    # code_list에는 대상 종목 코드들이 들어가 있음
    # 하나씩 하기 위해서 0. beforedata에 원하는 종목 코드들(or 1개)를 적어둠 1. count = 만들 모델 번호 2. before_code 전 모델의 종목코드
    #  3. before_model_name 전 모델의 이름 4. tempcount 2이상
    count = 3
    for code in code_list:
        if count < 10:
            count_str = "0" + str(count)
        else:
            count_str = str(count)
        if reverse:
            model_name = "r_five_" + count_str
        else:
            model_name = "five_" + count_str
        # model_score = 0.0
        # model_profit = 0.0
        ####
        model_score_list = []
        model_profit_list = []
        ####
        semi_count = 1
        model_score_dict[model_name] = {}
        print("current model : %s" % model_name)
        semi_code_list = copy.deepcopy(abs_code_list)
        for testcode in semi_code_list:
            print("count : %s / %s codename : %s" % (semi_count, verlist, testcode))
            for i in range(delta.days + 1):
                target_date = str(d1 + datetime.timedelta(days=i)).replace('-', '')
                print("current date : %s" % target_date)
                filepath = os.path.join(settings.BASE_DIR,
                                        'data/{}/{}/{}_{}_fivemin_data.txt'.format('f1', code, code, target_date))
                if not os.path.isfile(filepath):
                    continue
                temp_score, temp_profit = test(testcode, code, model_name, target_date=target_date)
                # model_score += temp_score
                # model_profit += temp_profit
                # ####
                model_score_list.append(temp_score)
                model_profit_list.append(temp_profit)
                # model_score = 0.0
                # model_profit = 0.0
                # ####
            semi_count += 1
        ####
        model_score_dict[model_name].update({"점수 리스트": model_score_list})
        model_score_dict[model_name].update({"이득 리스트": model_profit_list})
        ####
        total_score = sum(model_score_list)
        total_profit = sum(model_profit_list)
        model_score_dict[model_name].update({"총 점수": total_score})
        model_score_dict[model_name].update({"총 이익": total_profit})
        count += 1
    veryfilepath = os.path.join(settings.BASE_DIR, 'files/verifydata/verifydetaildata.txt')
    if not os.path.isfile(veryfilepath):
        veryfile = open(veryfilepath, "w", encoding="utf8")
    else:
        veryfile = open(veryfilepath, "a", encoding="utf8")
    for mo_name in model_score_dict.keys():
        veryfile.write(mo_name + "\t")
        #print("모델 이름 : %s 총 점수 : %s 총 이익 : %s" % (mo_name, str(model_score_dict[mo_name]["총 점수"]), str(model_score_dict[mo_name]["총 이익"])))
        for i in range(len(model_score_dict[mo_name]["점수 리스트"])):
            veryfile.write(str(model_score_dict[mo_name]["점수 리스트"][i]) + "\t" + str(model_score_dict[mo_name]["이득 리스트"][i]) +"\t")
            #print("%d번째 결과 - 점수 : %f / 이득 : %f" % (i+1, model_score_dict[mo_name]["점수 리스트"][i], model_score_dict[mo_name]["이득 리스트"][i]))
        veryfile.write(str(model_score_dict[mo_name]["총 점수"]) + "\t" + str(model_score_dict[mo_name]["총 이익"]) + "\n")
    veryfile.close()

def autolearning_datever(start_date="20200720", end_date="20200807"):
    # 시간 for문 저장
    d1 = datetime.date(int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8]))
    d2 = datetime.date(int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8]))
    delta = d2 - d1
    # code_list에는 대상 종목 코드들이 들어가 있음
    # 하나씩 하기 위해서 0. beforedata에 원하는 종목 코드들(or 1개)를 적어둠 1. count = 만들 모델 번호 2. before_code 전 모델의 종목코드
    #  3. before_model_name 전 모델의 이름 4. tempcount 2이상
    num_episodes = 1000
    count = 1
    tempcount = 1
    before_date = ""
    before_model_name = ""
    print("before_model_name : %s" % before_model_name)
    # 각 종목코드별로
    for i in range(delta.days + 1):
        if count < 10:
            count_str = "0" + str(count)
        else:
            count_str = str(count)
        #model_name = "five_" + count_str
        #model_name = "one_chegeul" + count_str
        #model_name = "one_memechegeul" + count_str
        #model_name = "one_acc_chegeul" + count_str
        #model_name = "one_only_acc_chegeul" + count_str
        #model_name = "one_only_acc_chegeul_only_profit" + count_str
        #model_name = "one_only_acc_chegeul_only_profit_huge" + count_str
        #model_name = "profit_twenty" + count_str
        #model_name = "profit_fourty" + count_str
        #model_name = "full_highprofit_twenty_simul_three_30min" + count_str
        #model_name = "full_prouprate_eighty_simul_30min" + count_str
        model_name = "prouprate100_total_simul_20action_y2_5steps_2basic_1000episodes_" + count_str
        # simul 해보고 그담에 일반 버전
        target_date = str(d1 + datetime.timedelta(days=i)).replace('-', '')
        code_list = []
        #filepath = "files/selected/" + target_date + ".txt"
        ###
        #filepath = "files/" + target_date + "/bigvol.txt"
        if model_name == "profit_twenty_simul_baseandprofit10_three04" + count_str:
            filepath = "files/" + target_date + "/twenty.txt"
        elif model_name == "profit_fourty" + count_str:
            filepath = "files/" + target_date + "/fourty.txt"
        elif model_name == "full_prouprate_eighty_simul_30min" + count_str:
            filepath = "files/" + target_date + "/eighty.txt"
        elif model_name == "full_highprofit_twenty_simul_three_30min" + count_str:
            filepath = "files/" + target_date + "/highprofit.txt"
        elif model_name == "prouprate100_total_simul_20action_y2_5steps_2basic_1000episodes_" + count_str:
            filepath = "files/" + target_date + "_learning.txt"
        else:
            print("error")
            return
        ###
        # 없는 경우 넘어감
        if not os.path.isfile(filepath):
            continue
        f = open(filepath, "r", encoding="utf8")
        lines = f.readlines()
        for line in lines:
            code_list.append(line.strip())
        f.close()
        print("current date : %s" % target_date)
        pure_count = 1
        for code in code_list:
            print("current code : %s count : %s / %s" % (code, pure_count, len(code_list)))
            if tempcount == 1:
                main(code, model_name, target_date, num_episodes=num_episodes)
            else:
                relearn(code, before_date, before_model_name, model_name, target_date, num_episodes=num_episodes)
            tempcount += 1
            before_date = target_date
            before_model_name = model_name
            pure_count += 1
        count += 1

def autoverifying_datever(start_date="20200803", end_date="20200807"):
    # 20200803 ~ 20200807
    d1 = datetime.date(int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8]))
    d2 = datetime.date(int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8]))
    delta = d2 - d1
    # 20200720 ~ 20200731 이어서 하려면 count에 목표 모델의 번호, start_date에 목표 모델의 날짜 기입
    # random model 테스트 시
    # 10개에 대해서 하려면 test_start_date와 test_end_date가 10정도 나올 수 있도록 함(20200902 ~ 20200916이 10
    count = 1
    test_start_date = "20200902"
    test_end_date = "20200911"
    d3 = datetime.date(int(test_start_date[0:4]), int(test_start_date[4:6]), int(test_start_date[6:8]))
    d4 = datetime.date(int(test_end_date[0:4]), int(test_end_date[4:6]), int(test_end_date[6:8]))
    seta = d4 - d3
    #model_score_dict = {}
    # test할 모델 for문
    for i in range(seta.days + 1):
        if count < 10:
            count_str = "0" + str(count)
        else:
            count_str = str(count)
        #model_name = "highprofit_twenty_simul_" + count_str
        model_name = "full_prouprate_eighty_simul_30min" + count_str
        # target_date = 모델의 날짜 이름
        target_date = str(d3 + datetime.timedelta(days=i)).replace('-', '')
        # temp_filepath = "files/selected/" + target_date + ".txt"
        # 없는 경우 넘어감 (txt파일이 있다면 있는 것으로 간주)
        temp_filepath = "files/" + target_date + "/testvol.txt"
        # 이거 왜함???
        if not os.path.isfile(temp_filepath):
            continue
        print("current model : %s" % model_name)
        veryfilepath = os.path.join(settings.BASE_DIR, 'files/verifydata/verifydetaildata.txt')
        if not os.path.isfile(veryfilepath):
            veryfile = open(veryfilepath, "w", encoding="utf8")
        else:
            veryfile = open(veryfilepath, "a", encoding="utf8")
        veryfile.write(model_name + "\t")
        total_score_list = []
        total_profit_list = []
        # test할 대상 for문
        for j in range(delta.days + 1):
            model_score_list = []
            model_profit_list = []
            # test_target_date = 모델의 test 대상 날짜
            test_target_date = str(d1 + datetime.timedelta(days=j)).replace('-', '')
            print("current date : %s" % test_target_date)
            # filepath = "files/selected/" + test_target_date + ".txt"
            filepath = "files/" + test_target_date + "/testvol.txt"
            # 없는 경우 넘어감
            if not os.path.isfile(filepath):
                continue
            # 현재 날짜 적기
            veryfile.write(test_target_date + "\t")
            code_list = []
            f = open(filepath, "r", encoding="utf8")
            lines = f.readlines()
            for line in lines:
                code_list.append(line.strip())
            f.close()
            semicount = 1
            for code in code_list:
                # filepath = os.path.join(settings.BASE_DIR, 'data/{}/{}/{}_{}_fivemin_data.txt'.format('f1', code, code, test_target_date))
                filepath = os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(test_target_date, code))
                if not os.path.isfile(filepath):
                    continue
                print("current code : %s current status : %s / %s" % (code, semicount, len(code_list)))
                # test(사용할 종목 코드, 모델 이름 앞의 구별자(여기서는 날짜), 모델 이름, 테스트할 날짜)
                temp_score, temp_profit = test(code, target_date, model_name, target_date=test_target_date)
                model_score_list.append(temp_score)
                model_profit_list.append(temp_profit)
                semicount += 1
            ####
            #model_score_dict[model_name].update({"점수 리스트": model_score_list})
            #model_score_dict[model_name].update({"이득 리스트": model_profit_list})
            ####
            total_score = sum(model_score_list)
            total_profit = sum(model_profit_list)
            #model_score_dict[model_name].update({"총 점수": total_score})
            #model_score_dict[model_name].update({"총 이익": total_profit})
            total_score_list.append(total_score)
            total_profit_list.append(total_profit)
            # veryfilepath = os.path.join(settings.BASE_DIR, 'files/verifydata/verifydetaildata.txt')
            # if not os.path.isfile(veryfilepath):
            #     veryfile = open(veryfilepath, "w", encoding="utf8")
            # else:
            #     veryfile = open(veryfilepath, "a", encoding="utf8")
            #for mo_name in model_score_dict.keys():
            #veryfile.write(model_name + "\t")
                #print("모델 이름 : %s 총 점수 : %s 총 이익 : %s" % (mo_name, str(model_score_dict[mo_name]["총 점수"]), str(model_score_dict[mo_name]["총 이익"])))
                # for i in range(len(model_score_dict[mo_name]["점수 리스트"])):
                #     veryfile.write(str(model_score_dict[mo_name]["점수 리스트"][i]) + "\t" + str(model_score_dict[mo_name]["이득 리스트"][i]) +"\t")
                #     #print("%d번째 결과 - 점수 : %f / 이득 : %f" % (i+1, model_score_dict[mo_name]["점수 리스트"][i], model_score_dict[mo_name]["이득 리스트"][i]))
                # veryfile.write(str(model_score_dict[mo_name]["총 점수"]) + "\t" + str(model_score_dict[mo_name]["총 이익"]) + "\n")
            ####
            veryfile.write(str(total_score) + "\t" + str(total_profit) +"\t")
            ####
        # 모든 날짜에 해당하는 점수 출력 후
        count += 1
        veryfile.write("total_score" + "\t" + str(sum(total_score_list)) + "\t" + "total_profit" + "\t" + str(sum(total_profit_list))+ "\n")
        veryfile.close()

def autoverifying_test_datever(start_date="20200803", end_date="20200807"):
    # 20200803 ~ 20200807
    d1 = datetime.date(int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8]))
    d2 = datetime.date(int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8]))
    delta = d2 - d1
    # 20200720 ~ 20200731 이어서 하려면 count에 목표 모델의 번호, start_date에 목표 모델의 날짜 기입
    # test_start_date에는 시작할 모델 번호, test_end_date는 끝낼 모델 번호
    count = 20
    test_start_date = "20200731"
    test_end_date = "20200731"
    d3 = datetime.date(int(test_start_date[0:4]), int(test_start_date[4:6]), int(test_start_date[6:8]))
    d4 = datetime.date(int(test_end_date[0:4]), int(test_end_date[4:6]), int(test_end_date[6:8]))
    seta = d4 - d3
    model_score_dict = {}
    # test할 모델 for문
    for i in range(seta.days + 1):
        if count < 10:
            count_str = "0" + str(count)
        else:
            count_str = str(count)
        model_name = "five_" + count_str
        # target_date = 모델의 날짜 이름
        # target_date = str(d3 + datetime.timedelta(days=i)).replace('-', '')
        # temp_filepath = "files/selected/" + target_date + ".txt"
        # # 없는 경우 넘어감 (txt파일이 있다면 있는 것으로 간주)
        # if not os.path.isfile(temp_filepath):
        #     continue
        target_date = "008350"
        model_score_list = []
        model_profit_list = []
        model_score_dict[model_name] = {}
        print("current model : %s" % model_name)
        # test할 대상 for문
        for j in range(delta.days + 1):
            # test_target_date = 모델의 test 대상 날짜
            test_target_date = str(d1 + datetime.timedelta(days=j)).replace('-', '')
            print("current date : %s" % test_target_date)
            # 3번의 반복으로 월요일이 타겟 날짜일 때 저번주 금요일의 종목 코드를 사용할 수 있도록 조정
            code_list = []
            for k in range(3):
                test_code_date = str(d1 + datetime.timedelta(days=j - (k + 1))).replace('-', '')
                filepath = "files/selected/" + test_code_date + ".txt"
                # 없는 경우 넘어감
                if not os.path.isfile(filepath):
                    continue
                # 있다면 종료
                else:
                    break
            f = open(filepath, "r", encoding="utf8")
            lines = f.readlines()
            for line in lines:
                code_list.append(line.strip())
            f.close()
            semicount = 1
            for code in code_list:
                filepath = os.path.join(settings.BASE_DIR, 'data/{}/{}/{}_{}_fivemin_data.txt'.format('f1', code, code, test_target_date))
                if not os.path.isfile(filepath):
                    continue
                print("current code : %s current status : %s / %s" % (code, semicount, len(code_list)))
                # test(사용할 종목 코드, 모델 이름 앞의 구별자(여기서는 날짜), 모델 이름, 테스트할 날짜)
                temp_score, temp_profit = test(code, target_date, model_name, target_date=test_target_date)
                model_score_list.append(temp_score)
                model_profit_list.append(temp_profit)
                semicount += 1
        ####
        model_score_dict[model_name].update({"점수 리스트": model_score_list})
        model_score_dict[model_name].update({"이득 리스트": model_profit_list})
        ####
        total_score = sum(model_score_list)
        total_profit = sum(model_profit_list)
        model_score_dict[model_name].update({"총 점수": total_score})
        model_score_dict[model_name].update({"총 이익": total_profit})
        count += 1

    veryfilepath = os.path.join(settings.BASE_DIR, 'files/verifydata/verifydetaildata.txt')
    if not os.path.isfile(veryfilepath):
        veryfile = open(veryfilepath, "w", encoding="utf8")
    else:
        veryfile = open(veryfilepath, "a", encoding="utf8")
    for mo_name in model_score_dict.keys():
        veryfile.write(mo_name + "\t")
        #print("모델 이름 : %s 총 점수 : %s 총 이익 : %s" % (mo_name, str(model_score_dict[mo_name]["총 점수"]), str(model_score_dict[mo_name]["총 이익"])))
        for i in range(len(model_score_dict[mo_name]["점수 리스트"])):
            veryfile.write(str(model_score_dict[mo_name]["점수 리스트"][i]) + "\t" + str(model_score_dict[mo_name]["이득 리스트"][i]) +"\t")
            #print("%d번째 결과 - 점수 : %f / 이득 : %f" % (i+1, model_score_dict[mo_name]["점수 리스트"][i], model_score_dict[mo_name]["이득 리스트"][i]))
        veryfile.write(str(model_score_dict[mo_name]["총 점수"]) + "\t" + str(model_score_dict[mo_name]["총 이익"]) + "\n")
    veryfile.close()

def recoverlearning_datever(start_date="20200720", end_date="20200807"):
    # 시간 for문 저장
    d1 = datetime.date(int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8]))
    d2 = datetime.date(int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8]))
    delta = d2 - d1
    num_episodes = 1000
    # code_list에는 대상 종목 코드들이 들어가 있음
    # 복구할(정확히는 마저 이어나갈) 모델의 count와 date, model_name입력
    count = 8
    before_date = "20201028"
    before_model_name = "prouprate100_total_simul_20action_y2_5steps_2basic_1000episodes_08"
    # 각 종목코드별로
    print("before_model_name : %s" % before_model_name)
    for i in range(delta.days + 1):
        if count < 10:
            count_str = "0" + str(count)
        else:
            count_str = str(count)
        #model_name = "full_highprofit_twenty_simul_base_three" + count_str
        model_name = "prouprate100_total_simul_20action_y2_5steps_2basic_1000episodes_" + count_str
        target_date = str(d1 + datetime.timedelta(days=i)).replace('-', '')
        code_list = []
        #filepath = "files/selected/" + target_date + ".txt"
        ###
        #filepath = "files/" + target_date + "/bigvol.txt"
        if model_name == "profit_twenty" + count_str:
            filepath = "files/" + target_date + "/twenty.txt"
        elif model_name == "profit_fourty" + count_str:
            filepath = "files/" + target_date + "/fourty.txt"
        elif model_name == "profit_eighty_simul_30min" + count_str:
            filepath = "files/" + target_date + "/eighty.txt"
        elif model_name == "full_highprofit_twenty_simul_base_three" + count_str:
            filepath = "files/" + target_date + "/highprofit.txt"
        elif model_name == "prouprate100_total_simul_20action_y2_5steps_2basic_1000episodes_" + count_str:
            filepath = "files/" + target_date + "_learning.txt"
        else:
            print("error")
            return
        # 이어나갈 코드 리스트의 주소(직접 구해놓기)
        # 쭉 나갈 수 있도록 만듦(어차피 안쓰겠지만 아무튼)
        if i == 0:
          filepath = "tempcodelist.txt"
        ###
        # 없는 경우 넘어감
        if not os.path.isfile(filepath):
            continue
        f = open(filepath, "r", encoding="utf8")
        lines = f.readlines()
        for line in lines:
            code_list.append(line.strip())
        f.close()
        print("current date : %s" % target_date)
        pure_count = 1
        for code in code_list:
            print("current code : %s count : %s / %s" % (code, pure_count, len(code_list)))
            relearn(code, before_date, before_model_name, model_name, target_date, num_episodes=num_episodes)
            before_date = target_date
            before_model_name = model_name
            pure_count += 1
        count += 1

def print_current_time():
    now_time = datetime.datetime.now()
    now_hour = now_time.hour
    now_minute = now_time.minute
    now_second = now_time.second
    print("current time : %s:%s:%s" % (str(now_hour), str(now_minute), str(now_second)))

def autolearning_random(start_date="20200720", end_date="20200807"):
    # 시간 for문 저장
    d1 = datetime.date(int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8]))
    d2 = datetime.date(int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8]))
    delta = d2 - d1
    date_list = []
    code_dict = {}
    # code_list에는 대상 종목 코드들이 들어가 있음
    # 하나씩 하기 위해서 0. beforedata에 원하는 종목 코드들(or 1개)를 적어둠 1. count = 만들 모델 번호 2. before_code 전 모델의 종목코드
    #  3. before_model_name 전 모델의 이름 4. tempcount 2이상
    #num_learning + relearncount = 100000
    # relearncount = 이어할 모델 뒤 count
    num_learning = 100000
    relearncount = 0
    before_model_name = ""
    print("before_model_name : %s" % before_model_name)
    # 각 종목코드별로 각 거래대금 상위 100의 데이터 전부 사용 -> 어차피 검사 시에는 다음주 데이터 사용
    for i in range(delta.days + 1):
        #current_date = "20200902"
        current_date = str(d1 + datetime.timedelta(days=i)).replace('-', '')
        filepath = "files/" + current_date + ".txt"
        if not os.path.isfile(filepath):
            continue
        code_list = []
        date_list.append(current_date)
        f = open(filepath, "r", encoding="utf8")
        lines = f.readlines()
        for line in lines:
            code_list.append(line.strip())
        f.close()
        code_dict[current_date] = {}
        code_dict[current_date].update({"code_list": code_list})
    # stock_code2는 before_target_date와 동일
    random_dateint = random.randint(0, (len(date_list) - 1))
    # target_date에 "20200902"를 기대함(다른 날짜도 가능하고)
    target_date = date_list[random_dateint]
    # 특정 날짜의 eighty.txt의 코드들의 길이 만큼을 기대함
    random_codeint = random.randint(0, (len(code_dict[target_date]["code_list"]) - 1))
    stock_code = code_dict[target_date]["code_list"][random_codeint]
    ################################# 테스트
    # 초기설정
    # 1번 환경, 에이전트 설정
    # 2번 PPO 생성(이 때, 에이전트의 행동 반경 확인
    # 3번 생성한 PPO로 learning
    initial_balance = 10000000  # threemin, fivemin
    chart_data, training_data = data_manager.load_data(
        os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code)), ver='y2')
    age = agent.Agent(environment=environment.Environment(chart_data, training_data), min_trading_unit=1,
                      max_trading_unit=2, delayed_reward_threshold=.05, importance=0.9)
    age.set_balance(initial_balance)
    num_features = age.STATE_DIM
    input_len = age.environment.get_training_data_shape()
    if input_len != 0:
        num_features += input_len  # traing_data만 상태임
    num_actions = age.NUM_ACTIONS
    network_name_actor = before_model_name + '_actor'
    network_name_value = before_model_name + '_value'
    path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
    path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
    if not os.path.isfile(path_actor):
        path_actor = None
    if not os.path.isfile(path_value):
        path_value = None
    PPO = PPOIQN(num_state=num_features, num_action=num_actions, actor_path=path_actor, value_path=path_value)
    PPO.assign_policy_parameters()
    score = 0.0
    print_interval = 100
    save_interval = 1000
    num_repeat = 1
    age_num_buy = 0
    age_num_sell = 0
    age_num_hold = 0
    age_pv = 0.0
    age_score = 0.0
    for episodes in range(num_learning):  # 스타크래프트 2 1백만
        random_dateint = random.randint(0, (len(date_list) - 1))
        # target_date에 "20200902"를 기대함(다른 날짜도 가능하고)
        target_date = date_list[random_dateint]
        # 특정 날짜의 eighty.txt의 코드들의 길이 만큼을 기대함
        random_codeint = random.randint(0, (len(code_dict[target_date]["code_list"]) - 1))
        stock_code = code_dict[target_date]["code_list"][random_codeint]
        chart_data, training_data = data_manager.load_data(
            os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(target_date, stock_code)), ver='y2')
        age.set_environment(environment=environment.Environment(chart_data, training_data))
        for z in range(num_repeat):
            # 관측, 행동 리스트, 가치 예측, 보상 리스트 초기화
            observations = []
            actions_list = []
            v_preds = []
            rewards = []
            age.environment.reset()
            age.reset()
            state = age.build_sample()
            if state is None:
                done = True
            else:
                done = False
            while not done:
                state = np.stack([state]).astype(dtype=np.float32)
                # 상태에 대한 행동 및 가치 예측을 얻음 + 스칼라로 변환
                act, v_pred = PPO.get_action(state)  # 상태에 따른 정책을 얻음
                act, v_pred = act.item(), v_pred.item()  # 행동을 스칼라로, 가치를 스칼라로
                s_prime, reward, done = age.newstep(act)
                observations.append(state)
                actions_list.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)
                if done:
                    # 다음 예측은 예측의 2번째(1번째 인덱스 부터 마지막은 0
                    v_preds_next = v_preds[1:] + [0]
                    # gaes 계산 = PPO의 get_gaes
                    gaes = PPO.get_gaes(rewards, v_preds, v_preds_next)
                    # reshape
                    observations = np.reshape(observations, [-1, num_features])  # 왜 이럼??
                    # observations = np.array(observations).astype(dtype=np.float32)
                    # action은 인트, 나머지는 플롯
                    actions_list = np.array(actions_list).astype(dtype=np.int32)
                    rewards = np.array(rewards).astype(dtype=np.float32)
                    v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                    gaes = np.array(gaes).astype(dtype=np.float32)

                    # 올드한 놈들을 최신 놈들로 바꿔줌
                    PPO.assign_policy_parameters()
                    # input = [상태들, 행동리스트, 보상리스트, 다음 예측 리스트, gaes]
                    inp = [observations, actions_list, rewards, v_preds_next, gaes]
                    # 왜 4임????
                    for epoch in range(4):
                        # 왜 32임??? 0 ~ 관찰행전부까지의 랜덤 난수를 size 만큼 생성
                        sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                           size=32)  # indices are in [low, high)
                        # 각 input에 대하여 take함
                        sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                        # PPO.train을 각 상태, 행동, 보상, 다음 예측, gaes으로 실행
                        PPO.train(obs=sampled_inp[0],
                                  actions=sampled_inp[1],
                                  rewards=sampled_inp[2],
                                  v_preds_next=sampled_inp[3],
                                  gaes=sampled_inp[4])
                    score = sum(rewards)
                state = s_prime
            age_num_buy += age.num_buy
            age_num_sell += age.num_sell
            age_num_hold += age.num_hold
            age_pv += age.portfolio_value
            age_score += score
        if (episodes + 1) % print_interval == 0:
            print("# of episode :{} score : {:.4f} pv : {:.2f} buy :{:.2f} sell : {:.2f} hold : {:.2f}".format(
                episodes + 1 + relearncount, age_score / (print_interval * num_repeat), age_pv / (print_interval * num_repeat), age_num_buy / (print_interval * num_repeat),
                age_num_sell / (print_interval * num_repeat), age_num_hold / (print_interval * num_repeat)))
            # date로 할 때에는 stock_code1 대신 target_date를 사용
            age_num_buy = 0
            age_num_sell = 0
            age_num_hold = 0
            age_pv = 0.0
            age_score = 0.0
        if (episodes + 1) % save_interval == 0:
            count = episodes + 1 + relearncount
            model_name = "full_random_real_profit_30min_deep18f" + str(count)
            network_name_actor = model_name + '_actor'
            network_name_value = model_name + '_value'
            path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
            path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
            PPO.save_model(actor_path=path_actor, value_path=path_value)
    clear_session()

def autoverifying_randomver(start_date="20200803", end_date="20200807"):
    # 20200803 ~ 20200807
    d1 = datetime.date(int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8]))
    d2 = datetime.date(int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8]))
    delta = d2 - d1
    # 20200720 ~ 20200731 이어서 하려면 count에 목표 모델의 번호, start_date에 목표 모델의 날짜 기입
    # random model 테스트 시
    # 10개에 대해서 하려면 test_start_date와 test_end_date가 10정도 나올 수 있도록 함(20200902 ~ 20200915이 10
    count = 23
    test_start_date = "20200902"
    test_end_date = "20200902"
    d3 = datetime.date(int(test_start_date[0:4]), int(test_start_date[4:6]), int(test_start_date[6:8]))
    d4 = datetime.date(int(test_end_date[0:4]), int(test_end_date[4:6]), int(test_end_date[6:8]))
    seta = d4 - d3
    #model_score_dict = {}
    # test할 모델 for문
    for i in range(seta.days + 1):
        counting = count * 1000
        model_name = "full_random_real_profit_30min_deep18f" + str(counting)
        #############################
        target_date = str(d3 + datetime.timedelta(days=i)).replace('-', '')
        # temp_filepath = "files/selected/" + target_date + ".txt"
        # 없는 경우 넘어감 (txt파일이 있다면 있는 것으로 간주)
        temp_filepath = "files/" + target_date + "/testvol.txt"
        # 이거 왜함???
        if not os.path.isfile(temp_filepath):
            continue
        print("current model : %s" % model_name)
        veryfilepath = os.path.join(settings.BASE_DIR, 'files/verifydata/verifydetaildata.txt')
        if not os.path.isfile(veryfilepath):
            veryfile = open(veryfilepath, "w", encoding="utf8")
        else:
            veryfile = open(veryfilepath, "a", encoding="utf8")
        veryfile.write(model_name + "\t")
        total_score_list = []
        total_profit_list = []
        # test할 대상 for문
        for j in range(delta.days + 1):
            model_score_list = []
            model_profit_list = []
            # test_target_date = 모델의 test 대상 날짜
            test_target_date = str(d1 + datetime.timedelta(days=j)).replace('-', '')
            print("current date : %s" % test_target_date)
            code_list = []
            # filepath = "files/selected/" + test_target_date + ".txt"
            filepath = "files/" + test_target_date + "/testvol.txt"
            # 없는 경우 넘어감
            if not os.path.isfile(filepath):
                continue
            f = open(filepath, "r", encoding="utf8")
            lines = f.readlines()
            for line in lines:
                code_list.append(line.strip())
            f.close()
            semicount = 1
            for code in code_list:
                # filepath = os.path.join(settings.BASE_DIR, 'data/{}/{}/{}_{}_fivemin_data.txt'.format('f1', code, code, test_target_date))
                filepath = os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(test_target_date, code))
                if not os.path.isfile(filepath):
                    continue
                print("current code : %s current status : %s / %s" % (code, semicount, len(code_list)))
                # test(사용할 종목 코드, 모델 이름 앞의 구별자(여기서는 날짜), 모델 이름, 테스트할 날짜)
                temp_score, temp_profit = test_randmodel(code, target_date, model_name, target_date=test_target_date)
                model_score_list.append(temp_score)
                model_profit_list.append(temp_profit)
                semicount += 1
            total_score = sum(model_score_list)
            total_profit = sum(model_profit_list)
            total_score_list.append(total_score)
            total_profit_list.append(total_profit)
            veryfile.write(str(total_score) + "\t" + str(total_profit) +"\t")
        count += 1
        veryfile.write("total_score" + "\t" + str(sum(total_score_list)) + "\t" + "total_profit" + "\t" + str(sum(total_profit_list))+ "\n")
        veryfile.close()

def day_main(stock_code, model_name, target_date, actor_path=None, value_path=None, num_episodes=100):
    best_score = None
    initial_balance = 10000000 # threemin, fivemin
    chart_data, training_data = data_manager.load_data(
        os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format("daysdata", stock_code)), ver='z3')
    age = agent.Agent(environment=environment.Environment(chart_data, training_data))
    age.set_balance(initial_balance)
    num_features = age.STATE_DIM
    input_len = age.environment.get_training_data_shape()
    if input_len != 0:
        num_features += input_len  # traing_data만 상태임
    num_actions = age.NUM_ACTIONS
    PPO = PPOIQN(num_state=num_features, num_action=num_actions, actor_path=actor_path, value_path=value_path)
    PPO.assign_policy_parameters()
    score = 0.0
    print_interval = num_episodes
    age_num_buy = 0
    age_num_sell = 0
    age_num_hold = 0
    age_pv = 0.0
    for episodes in range(num_episodes): # 스타크래프트 2 1백만
        # 관측, 행동 리스트, 가치 예측, 보상 리스트 초기화
        observations = []
        actions_list = []
        v_preds = []
        rewards = []
        age.environment.reset()
        age.reset()
        state = age.build_sample()
        if state is None:
            done = True
        else:
            done = False
        while not done:
            state = np.stack([state]).astype(dtype=np.float32)
            # 상태에 대한 행동 및 가치 예측을 얻음 + 스칼라로 변환
            act, v_pred = PPO.get_action(state)  # 상태에 따른 정책을 얻음
            act, v_pred = act.item(), v_pred.item()  # 행동을 스칼라로, 가치를 스칼라로
            s_prime, reward, done = age.newstep(act)
            #s_prime, reward, done = age.simulstep(act)
            observations.append(state)
            actions_list.append(act)
            v_preds.append(v_pred)
            rewards.append(reward)
            if done:
                # 다음 예측은 예측의 2번째(1번째 인덱스 부터 마지막은 0
                v_preds_next = v_preds[1:] + [0]
                # gaes 계산 = PPO의 get_gaes
                gaes = PPO.get_gaes(rewards, v_preds, v_preds_next)
                # reshape
                observations = np.reshape(observations, [-1, num_features]) # 왜 이럼??
                #observations = np.array(observations).astype(dtype=np.float32)
                # action은 인트, 나머지는 플롯
                actions_list = np.array(actions_list).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)

                # 올드한 놈들을 최신 놈들로 바꿔줌

                PPO.assign_policy_parameters()

                # input = [상태들, 행동리스트, 보상리스트, 다음 예측 리스트, gaes]

                inp = [observations, actions_list, rewards, v_preds_next, gaes]
                # 왜 4임????
                for epoch in range(4):
                    # 왜 32임??? 0 ~ 관찰행전부까지의 랜덤 난수를 size 만큼 생성
                    sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)  # indices are in [low, high)
                    # 각 input에 대하여 take함
                    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                    # PPO.train을 각 상태, 행동, 보상, 다음 예측, gaes으로 실행
                    PPO.train(obs=sampled_inp[0],
                              actions=sampled_inp[1],
                              rewards=sampled_inp[2],
                              v_preds_next=sampled_inp[3],
                              gaes=sampled_inp[4])
            state = s_prime
        #### 점수 수정
        if len(rewards) == 0:
            score = 0
        else:
            score += (sum(rewards) / len(rewards))
        age_num_buy += age.num_buy
        age_num_sell += age.num_sell
        age_num_hold += age.num_hold
        age_pv += age.portfolio_value
        if (episodes + 1) % print_interval == 0:
            print("# of episode :{} score : {:.4f} pv : {:.2f} buy :{:.2f} sell : {:.2f} hold : {:.2f}".format(episodes + 1 , score / print_interval, age_pv / print_interval, age_num_buy / print_interval, age_num_sell / print_interval, age_num_hold / print_interval))
            if best_score is None:
                best_score = score
            if score >= best_score:
                best_score = score
                # date로 할 때에는 stock_code 대신 target_date를 사용
                network_name_actor = target_date + '_ppo_model_' + model_name + '_actor'
                network_name_value = target_date + '_ppo_model_' + model_name + '_value'
                path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
                path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
                PPO.save_model(actor_path=path_actor, value_path=path_value)
            age_num_buy = 0
            age_num_sell = 0
            age_num_hold = 0
            age_pv = 0.0
            score = 0.0
    clear_session()
    # if keras.backend.tensorflow_backend._SESSION:
    #     tf.reset_default_graph()
    #     keras.backend.tensorflow_backend._SESSION.close()
    #     keras.backend.tensorflow_backend._SESSION = None

def day_relearn(stock_code1, stock_code2, model_name, new_model_name, target_date ,actor_path=None, value_path=None, num_episodes=100):
    #stock_code2는 before_target_date와 동일
    best_score = None
    initial_balance = 10000000 # threemin, fivemin
    chart_data, training_data = data_manager.load_data(
        os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format("daysdata", stock_code1)), ver='z3')
    age = agent.Agent(environment=environment.Environment(chart_data, training_data), min_trading_unit=1,
                      max_trading_unit=2)
    age.set_balance(initial_balance)
    num_features = age.STATE_DIM
    input_len = age.environment.get_training_data_shape()
    if input_len != 0:
        num_features += input_len  # traing_data만 상태임
    num_actions = age.NUM_ACTIONS
    network_name_actor = stock_code2 + '_ppo_model_' + model_name + '_actor'
    network_name_value = stock_code2 + '_ppo_model_' + model_name + '_value'
    path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
    path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
    PPO = PPOIQN(num_state=num_features, num_action=num_actions, actor_path=path_actor, value_path=path_value)
    PPO.assign_policy_parameters()
    score = 0.0
    print_interval = num_episodes
    age_num_buy = 0
    age_num_sell = 0
    age_num_hold = 0
    age_pv = 0.0
    for episodes in range(num_episodes): # 스타크래프트 2 1백만
        # 관측, 행동 리스트, 가치 예측, 보상 리스트 초기화
        observations = []
        actions_list = []
        v_preds = []
        rewards = []
        age.environment.reset()
        age.reset()
        state = age.build_sample()
        if state is None:
            done = True
        else:
            done = False
        while not done:
            state = np.stack([state]).astype(dtype=np.float32)
            # 상태에 대한 행동 및 가치 예측을 얻음 + 스칼라로 변환
            act, v_pred = PPO.get_action(state)  # 상태에 따른 정책을 얻음
            act, v_pred = act.item(), v_pred.item()  # 행동을 스칼라로, 가치를 스칼라로
            s_prime, reward, done = age.newstep(act)
            #s_prime, reward, done = age.simulstep(act)
            observations.append(state)
            actions_list.append(act)
            v_preds.append(v_pred)
            rewards.append(reward)
            if done:
                # 다음 예측은 예측의 2번째(1번째 인덱스 부터 마지막은 0
                v_preds_next = v_preds[1:] + [0]
                # gaes 계산 = PPO의 get_gaes
                gaes = PPO.get_gaes(rewards, v_preds, v_preds_next)
                # reshape
                observations = np.reshape(observations, [-1, num_features]) # 왜 이럼??
                #observations = np.array(observations).astype(dtype=np.float32)
                # action은 인트, 나머지는 플롯
                actions_list = np.array(actions_list).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)

                # 올드한 놈들을 최신 놈들로 바꿔줌
                PPO.assign_policy_parameters()
                # input = [상태들, 행동리스트, 보상리스트, 다음 예측 리스트, gaes]
                inp = [observations, actions_list, rewards, v_preds_next, gaes]
                # 왜 4임????
                for epoch in range(4):
                    # 왜 32임??? 0 ~ 관찰행전부까지의 랜덤 난수를 size 만큼 생성
                    sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)  # indices are in [low, high)
                    # 각 input에 대하여 take함
                    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                    # PPO.train을 각 상태, 행동, 보상, 다음 예측, gaes으로 실행
                    PPO.train(obs=sampled_inp[0],
                              actions=sampled_inp[1],
                              rewards=sampled_inp[2],
                              v_preds_next=sampled_inp[3],
                              gaes=sampled_inp[4])
            state = s_prime
        ##### 점수 수정
        if len(rewards) == 0:
            score = 0
        else:
            score += (sum(rewards) / len(rewards))
        age_num_buy += age.num_buy
        age_num_sell += age.num_sell
        age_num_hold += age.num_hold
        age_pv += age.portfolio_value
        if (episodes + 1) % print_interval == 0:
            print("# of episode :{} score : {:.4f} pv : {:.2f} buy :{:.2f} sell : {:.2f} hold : {:.2f}".format(episodes + 1 , score / print_interval, age_pv / print_interval, age_num_buy / print_interval, age_num_sell / print_interval, age_num_hold / print_interval))
            if best_score is None:
                best_score = score
            if score >= best_score:
                best_score = score
                # date로 할 때에는 stock_code1 대신 target_date를 사용
                network_name_actor = target_date + '_ppo_model_' + new_model_name + '_actor'
                network_name_value = target_date + '_ppo_model_' + new_model_name + '_value'
                path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
                path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
                PPO.save_model(actor_path=path_actor, value_path=path_value)
            age_num_buy = 0
            age_num_sell = 0
            age_num_hold = 0
            age_pv = 0.0
            score = 0.0
    clear_session()

def day_test(stock_code1, stock_code2, model_name, target_date, actor_path=None, value_path=None, is_random=False):
    best_score = 0.
    initial_balance = 10000000
    chart_data, training_data = data_manager.load_data(
    os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format("testdaysdata", stock_code1)), ver='z3')
    age = agent.Agent(environment=environment.Environment(chart_data, training_data), min_trading_unit=1,
                      max_trading_unit=2)
    age.set_balance(initial_balance)
    num_features = age.STATE_DIM
    input_len = age.environment.get_training_data_shape()
    if input_len != 0:
        num_features += input_len  # traing_data만 상태임
    num_actions = age.NUM_ACTIONS
    if is_random:
        network_name_actor = model_name + "_actor"
        network_name_value = model_name + "_value"
    else:
        network_name_actor = stock_code2 + '_ppo_model_' + model_name + '_actor'
        network_name_value = stock_code2 + '_ppo_model_' + model_name + '_value'

    path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
    path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
    PPO = PPOIQN(num_state=num_features, num_action=num_actions, actor_path=path_actor, value_path=path_value, testing=True)
    #PPO.assign_policy_parameters()
    score = 0.0
    total_score = 0.0
    total_profit = 0.0
    num_episodes = 10
    for episodes in range(num_episodes): # 스타크래프트 2 1백만
        # 관측, 행동 리스트, 가치 예측, 보상 리스트 초기화
        observations = []
        actions_list = []
        #v_preds = []
        rewards = []
        age.environment.reset()
        age.reset()
        state = age.build_sample()
        if state is None:
            done = True
        else:
            done = False
        while not done:
            state = np.stack([state]).astype(dtype=np.float32)
            # 상태에 대한 행동 및 가치 예측을 얻음 + 스칼라로 변환
            act = PPO.give_action(state)  # 상태에 따른 정책을 얻음
            act = act.item() #v_pred.item()  # 행동을 스칼라로, 가치를 스칼라로
            s_prime, reward, done = age.newstep(act)
            #s_prime, reward, done = age.simulstep(act)
            observations.append(state)
            actions_list.append(act)
            #v_preds.append(v_pred)
            rewards.append(reward)
            if done:
                if len(rewards) == 0:
                    score = 0
                else:
                    score = (sum(rewards) / len(rewards))
            state = s_prime
        print("# of episode :{} score : {:.4f} pv : {:.4f} buy :{} sell : {} hold : {} stocks : {}".format(episodes + 1, score, age.portfolio_value, age.num_buy, age.num_sell, age.num_hold, age.num_stocks))
        total_score += score
        profit = age.portfolio_value - age.initial_balance
        total_profit += profit
        score = 0.0
    # after finish 10times of episodes, total score will be model's performance
    avg_score = total_score / num_episodes
    avg_profit = total_profit / num_episodes
    clear_session()
    return (avg_score, avg_profit)

def autolearning_daysver(start_date="20200720", end_date="20200807"):
    # 시간 for문 저장
    num_episodes = 1000
    code_list = []
    tempcount = 1
    filepath = "files/eight_ten_three_month_big.txt"
    if not os.path.isfile(filepath):
        return
    f = open(filepath, "r", encoding="utf8")
    lines = f.readlines()
    for line in lines:
        code_list.append(line.strip())
    f.close()
    pure_count = 0
    target_date = "8to10"
    count_str = "01"
    before_date = ""
    before_model_name = ""
    for code in code_list:
        if pure_count % 100 == 0:
            if (pure_count/100) < 10:
                count_str = "0" + str((pure_count/100 + 1))
            else:
                count_str = str((pure_count/100 + 1))
        model_name = "daymemoryzeroonechangeotheradjust_total_simul_extreme_z3_1steps_plus2basic_1000episodes_" + count_str
        print("current code : %s count : %s / %s" % (code, pure_count + 1, len(code_list)))
        if tempcount == 1:
            day_main(code, model_name, target_date, num_episodes=num_episodes)
        else:
            day_relearn(code, before_date, before_model_name, model_name, target_date, num_episodes=num_episodes)
        tempcount += 1
        before_date = target_date
        before_model_name = model_name
        # pure_count로 제어 가능 20, 40, 60, 80 등...
        pure_count += 1

def recoverlearning_daysver(start_date="20200720", end_date="20200807"):
    # 시간 for문 저장
    num_episodes = 1000
    code_list = []
    tempcount = 2
    filepath = "tempcodelist.txt"
    if not os.path.isfile(filepath):
        return
    f = open(filepath, "r", encoding="utf8")
    lines = f.readlines()
    for line in lines:
        code_list.append(line.strip())
    f.close()
    pure_count = 211
    target_date = "8to10"
    count_str = "02"
    before_date = "8to10"
    before_model_name = "daymemoryzeroonechangeotheradjust_total_simul_extreme_z3_1steps_2basic_1000episodes_02"
    for code in code_list:
        if (pure_count/100) < 10:
            count_str = "0" + str(int(pure_count/100 + 1))
        else:
            count_str = str(int(pure_count/100 + 1))
        model_name = "daymemoryzeroonechangeotheradjust_total_simul_extreme_z3_1steps_2basic_1000episodes_" + count_str
        print("current code : %s count : %s / %s" % (code, pure_count + 1, len(code_list)))
        if not os.path.isfile('files/{}/{}.txt'.format("daysdata", code)):
            continue
        if tempcount == 1:
            day_main(code, model_name, target_date, num_episodes=num_episodes)
        else:
            day_relearn(code, before_date, before_model_name, model_name, target_date, num_episodes=num_episodes)
        tempcount += 1
        before_date = target_date
        before_model_name = model_name
        # pure_count로 제어 가능 20, 40, 60, 80 등...
        pure_count += 1

def autoverifying_daysever(start_date="20200803", end_date="20200807", target_date="", is_random=False):
    # test할 모델 for문
    # target_date = "8to10"
    # count_str = "03"
    # model_name = "daymemoryzeroonechangeotheradjust_total_simul_extreme_z3_1steps_2basic_1000episodes_" + count_str
    target_date = ""
    # count_str = "10000"
    # model_name = "prouprate100_random_simul_extreme_z3_1steps_2basic_1episodes_" + count_str
    for i in range(10):
        count_str = str(i+1) + "0000"
        model_name = "prouprate100_random_simul_extreme_z3_1steps_2basic_1episodes_" + count_str
        # target_date = 모델의 날짜 이름
        # temp_filepath = "files/selected/" + target_date + ".txt"
        # 없는 경우 넘어감 (txt파일이 있다면 있는 것으로 간주)
        #temp_filepath = "files/" + target_date + "/testvol.txt"
        print("current model : %s" % model_name)
        veryfilepath = os.path.join(settings.BASE_DIR, 'files/verifydata/verifydetaildata.txt')
        if not os.path.isfile(veryfilepath):
            veryfile = open(veryfilepath, "w", encoding="utf8")
        else:
            veryfile = open(veryfilepath, "a", encoding="utf8")
        veryfile.write(model_name + "\t")
        total_score_list = []
        total_profit_list = []
        # test할 대상 for문
        model_score_list = []
        model_profit_list = []
        # test_target_date = 모델의 test 대상 날짜
        test_target_date = ""
        print("current model_date : %s" % test_target_date)
        # filepath = "files/selected/" + test_target_date + ".txt"
        #filepath = "files/" + test_target_date + "/testvol.txt"
        # 전체 데이터로 테스트(100개)
        filepath = "files/eight_ten_three_month_big.txt"
        # 없는 경우 넘어감
        # 현재 날짜 적기
        veryfile.write(test_target_date + "\t")
        code_list = []
        f = open(filepath, "r", encoding="utf8")
        lines = f.readlines()
        for line in lines:
            code_list.append(line.strip())
        f.close()
        semicount = 1
        for code in code_list:
            # filepath = os.path.join(settings.BASE_DIR, 'data/{}/{}/{}_{}_fivemin_data.txt'.format('f1', code, code, test_target_date))
            #filepath = os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format(test_target_date, code))
            filepath = os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format("testdaysdata", code))
            if not os.path.isfile(filepath):
                continue
            print("current code : %s current status : %s / %s" % (code, semicount, len(code_list)))
            # test(사용할 종목 코드, 모델 이름 앞의 구별자(여기서는 날짜), 모델 이름, 테스트할 날짜)
            if is_random:
                temp_score, temp_profit = day_test(code, target_date, model_name, target_date=test_target_date, is_random=True)
            else:
                temp_score, temp_profit = day_test(code, target_date, model_name, target_date=test_target_date)
            model_score_list.append(temp_score)
            model_profit_list.append(temp_profit)
            # 20, 40, 60, 80 끊기
            semicount += 1
            ####
            # model_score_dict[model_name].update({"점수 리스트": model_score_list})
            # model_score_dict[model_name].update({"이득 리스트": model_profit_list})
            ####
            total_score = sum(model_score_list)
            total_profit = sum(model_profit_list)
            # model_score_dict[model_name].update({"총 점수": total_score})
            # model_score_dict[model_name].update({"총 이익": total_profit})
            total_score_list.append(total_score)
            total_profit_list.append(total_profit)
            # veryfilepath = os.path.join(settings.BASE_DIR, 'files/verifydata/verifydetaildata.txt')
            # if not os.path.isfile(veryfilepath):
            #     veryfile = open(veryfilepath, "w", encoding="utf8")
            # else:
            #     veryfile = open(veryfilepath, "a", encoding="utf8")
            # for mo_name in model_score_dict.keys():
            # veryfile.write(model_name + "\t")
            # print("모델 이름 : %s 총 점수 : %s 총 이익 : %s" % (mo_name, str(model_score_dict[mo_name]["총 점수"]), str(model_score_dict[mo_name]["총 이익"])))
            # for i in range(len(model_score_dict[mo_name]["점수 리스트"])):
            #     veryfile.write(str(model_score_dict[mo_name]["점수 리스트"][i]) + "\t" + str(model_score_dict[mo_name]["이득 리스트"][i]) +"\t")
            #     #print("%d번째 결과 - 점수 : %f / 이득 : %f" % (i+1, model_score_dict[mo_name]["점수 리스트"][i], model_score_dict[mo_name]["이득 리스트"][i]))
            # veryfile.write(str(model_score_dict[mo_name]["총 점수"]) + "\t" + str(model_score_dict[mo_name]["총 이익"]) + "\n")
            ####
            veryfile.write(str(total_score) + "\t" + str(total_profit) + "\t")
            ####
        # 모든 날짜에 해당하는 점수 출력 후
        veryfile.write("total_score" + "\t" + str(sum(total_score_list)) + "\t" + "total_profit" + "\t" + str(
            sum(total_profit_list)) + "\n")
        veryfile.close()

def autolearning_daysver_random(start_date="20200720", end_date="20200807"):
    # 시간 for문 저장
    code_list = []
    filepath = "files/eight_ten_three_month_big.txt"
    if not os.path.isfile(filepath):
        return
    f = open(filepath, "r", encoding="utf8")
    lines = f.readlines()
    for line in lines:
        code_list.append(line.strip())
    f.close()
    # stock_code2는 before_target_date와 동일
    # 특정 날짜의 eighty.txt의 코드들의 길이 만큼을 기대함
    num_learning = 100000
    relearncount = 0
    before_model_name = ""
    print("before_model_name : %s" % before_model_name)
    ################################# 테스트
    # 초기설정
    # 1번 환경, 에이전트 설정
    # 2번 PPO 생성(이 때, 에이전트의 행동 반경 확인
    # 3번 생성한 PPO로 learning
    random_codeint = random.randint(0, (len(code_list) - 1))
    stock_code = code_list[random_codeint]
    initial_balance = 10000000  # threemin, fivemin
    chart_data, training_data = data_manager.load_data(
        os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format("daysdata", stock_code)), ver='z3')
    age = agent.Agent(environment=environment.Environment(chart_data, training_data), min_trading_unit=1,
                              max_trading_unit=2)
    age.set_balance(initial_balance)
    num_features = age.STATE_DIM
    input_len = age.environment.get_training_data_shape()
    if input_len != 0:
        num_features += input_len  # traing_data만 상태임
    num_actions = age.NUM_ACTIONS
    network_name_actor = before_model_name + '_actor'
    network_name_value = before_model_name + '_value'
    path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
    path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
    if not os.path.isfile(path_actor):
        path_actor = None
    if not os.path.isfile(path_value):
        path_value = None
    PPO = PPOIQN(num_state=num_features, num_action=num_actions, actor_path=path_actor, value_path=path_value)
    PPO.assign_policy_parameters()
    score = 0.0
    print_interval = 1000
    save_interval = 10000
    num_repeat = 1
    age_num_buy = 0
    age_num_sell = 0
    age_num_hold = 0
    age_pv = 0.0
    age_score = 0.0
    for episodes in range(num_learning):  # 스타크래프트 2 1백만
        random_codeint = random.randint(0, (len(code_list)-1))
        stock_code = code_list[random_codeint]
        chart_data, training_data = data_manager.load_data(
            os.path.join(settings.BASE_DIR, 'files/{}/{}.txt'.format("daysdata", stock_code)), ver='z3')
        age.set_environment(environment=environment.Environment(chart_data, training_data))
        for z in range(num_repeat):
            # 관측, 행동 리스트, 가치 예측, 보상 리스트 초기화
            observations = []
            actions_list = []
            v_preds = []
            rewards = []
            age.environment.reset()
            age.reset()
            state = age.build_sample()
            if state is None:
                done = True
            else:
                done = False
            while not done:
                state = np.stack([state]).astype(dtype=np.float32)
                # 상태에 대한 행동 및 가치 예측을 얻음 + 스칼라로 변환
                act, v_pred = PPO.get_action(state)  # 상태에 따른 정책을 얻음
                act, v_pred = act.item(), v_pred.item()  # 행동을 스칼라로, 가치를 스칼라로
                s_prime, reward, done = age.newstep(act)
                #s_prime, reward, done = age.simulstep(act)
                observations.append(state)
                actions_list.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)
                if done:
                    # 다음 예측은 예측의 2번째(1번째 인덱스 부터 마지막은 0
                    v_preds_next = v_preds[1:] + [0]
                    # gaes 계산 = PPO의 get_gaes
                    gaes = PPO.get_gaes(rewards, v_preds, v_preds_next)
                    # reshape
                    observations = np.reshape(observations, [-1, num_features])  # 왜 이럼??
                    # observations = np.array(observations).astype(dtype=np.float32)
                    # action은 인트, 나머지는 플롯
                    actions_list = np.array(actions_list).astype(dtype=np.int32)
                    rewards = np.array(rewards).astype(dtype=np.float32)
                    v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                    gaes = np.array(gaes).astype(dtype=np.float32)

                    # 올드한 놈들을 최신 놈들로 바꿔줌
                    PPO.assign_policy_parameters()
                    # input = [상태들, 행동리스트, 보상리스트, 다음 예측 리스트, gaes]
                    inp = [observations, actions_list, rewards, v_preds_next, gaes]
                    # 왜 4임????
                    for epoch in range(4):
                        # 왜 32임??? 0 ~ 관찰행전부까지의 랜덤 난수를 size 만큼 생성
                        sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                           size=32)  # indices are in [low, high)
                        # 각 input에 대하여 take함
                        sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in
                                       inp]  # sample training data
                        # PPO.train을 각 상태, 행동, 보상, 다음 예측, gaes으로 실행
                        PPO.train(obs=sampled_inp[0],
                                  actions=sampled_inp[1],
                                  rewards=sampled_inp[2],
                                  v_preds_next=sampled_inp[3],
                                  gaes=sampled_inp[4])
                    score = (sum(rewards) / len(rewards))
                state = s_prime
            age_num_buy += age.num_buy
            age_num_sell += age.num_sell
            age_num_hold += age.num_hold
            age_pv += age.portfolio_value
            age_score += score
        if (episodes + 1) % print_interval == 0:
            print("# of episode :{} score : {:.4f} pv : {:.2f} buy :{:.2f} sell : {:.2f} hold : {:.2f}".format(
                episodes + 1 + relearncount, age_score / (print_interval * num_repeat),
                age_pv / (print_interval * num_repeat), age_num_buy / (print_interval * num_repeat),
                age_num_sell / (print_interval * num_repeat), age_num_hold / (print_interval * num_repeat)))
            # date로 할 때에는 stock_code1 대신 target_date를 사용
            age_num_buy = 0
            age_num_sell = 0
            age_num_hold = 0
            age_pv = 0.0
            age_score = 0.0
        if (episodes + 1) % save_interval == 0:
            count = episodes + 1 + relearncount
            model_name = "prouprate100_random_simul_extreme_z3_1steps_2basic_1episodes_" + str(count) # 3만까지함
            #model_name = "changeprofit100_random_real_extreme_z3_1steps_plus2basic_1episodes_" + str(count)
            #model_name = "changeprofit100_random_zero_extreme_z3_1steps_plus2basic_1episodes_" + str(count)
            network_name_actor = model_name + '_actor'
            network_name_value = model_name + '_value'
            path_actor = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_actor))
            path_value = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name_value))
            PPO.save_model(actor_path=path_actor, value_path=path_value)
    clear_session()

if __name__ == "__main__":
    # state 없애기, limit 조정, 랜덤 불러오기
    # 다음에는 profit 임계점 지정 (-3 3% 등)
    # 아직 0917 데이터 안 가져옴
    # simul_baseandprofit10 이어하기 or 100번 대신 1000번 학습해보기
    # random 유용한 것들만 모은 걸로??
    print_current_time()
    start_date = "20201028"
    end_date = "20201030"
    # 10월 20일 7번째 부터
    #autolearning_random(start_date=start_date, end_date=end_date)
    #autolearning_datever(start_date=start_date, end_date=end_date)
    recoverlearning_datever(start_date=start_date, end_date=end_date)
    #test_start_date = "20200914"
    #test_end_date = "20200922"
    #autoverifying_randomver(start_date=test_start_date, end_date=test_end_date)
    #autoverifying_datever(start_date=test_start_date, end_date=test_end_date)
    #autolearning_daysver_random()
    #autoverifying_daysever(is_random=True)
    # autocast_verify_datever(input_start_date=test_start_date, input_end_date=test_end_date)
    #score_start_date = "20200810"
    #score_end_date = "20200813"
    #autoverifying_test_datever(start_date=score_start_date, end_date=score_end_date)
    #datever으로 변형하기 위하여 main, relearn, test에서 이름부분에 날짜가 들어가도록 수정
    #start_date = "20200601"
    #end_date = "20200703"
    #autolearning(start_date=start_date, end_date=end_date)
    #autotesting()
    #autolearning(start_date=start_date, end_date=end_date, reverse=True)
    #autotesting(reverse=True)
    #test_start_date = "20200704"
    #test_end_date = "20200710"
    #autoverifying(start_date=test_start_date, end_date=test_end_date)
    #autoverifying(start_date=test_start_date, end_date=test_end_date, reverse=True)
    print_current_time()