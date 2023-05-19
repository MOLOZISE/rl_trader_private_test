import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import environment
import data_manager
import os
import numpy as np
import copy
import random
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0005  # Adam optimizer
gamma = 0.98  # discount factor
lmbda = 0.95  # GAE param
eps_clip = 0.2  # clip param
K_epoch = 3  #
T_horizon = 20

# stock trade fees
TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
TRADING_TAX = 0.0025  # 거래세 0.25%

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(15, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc_pi = nn.Linear(64, 2)
        self.fc_v = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = torch.tanh((self.fc1(x)))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

class PPO_lstm(nn.Module):
    def __init__(self):
        super(PPO_lstm, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(15, 64)
        self.lstm = nn.LSTM(64, 32)
        self.fc_pi = nn.Linear(32, 2)
        self.fc_v = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden

    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()

def train(save_path_txt):
    save_path = "models/" + save_path_txt
    # 환경 초기화
    train_date = "20210730"
    code = "001520"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # chart_data, training_data = data_manager.load_data(os.path.join(BASE_DIR, 'data/{}_min_data.txt'.format(code)), ver='ppo',
    #                                                    start_time="20210730091000", end_time="20210730150000")
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
    chart_data_list = []
    train_data_list = []
    filepath = "data/sample.txt"
    # 없는 경우 넘어감
    if not os.path.isfile(filepath):
        #error
        exit(0)
    f = open(filepath, "r", encoding="utf8")
    lines = f.readlines()
    for line in lines:
        code_list = line.strip().split("\t")
    f.close()
    for code in code_list:
        temp_chart_data, temp_training_data = data_manager.load_data(os.path.join(BASE_DIR, 'data/{}_min_data.txt'.format(code)), ver='ppo',
                                                       start_time="20210730093000", end_time="20210730150000")
        chart_data_list.append(copy.deepcopy(temp_chart_data))
        train_data_list.append(copy.deepcopy(temp_training_data))
    env = environment.Environment(chart_data=None, training_data=None)
    model = PPO()
    score = 0.0
    print_interval = 20
    #!!!!! episode change_interval
    change_interval = 100
    for n_epi in range(10000):
        # 초기 상태 세팅
        env.reset()
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
        if n_epi % change_interval == 0:
            temp_random = random.randrange(0, len(chart_data_list))
            env.set_chart_data(chart_data_list[temp_random])
            env.set_training_data(train_data_list[temp_random])
        #s = np.array(env.observe(), dtype="float32")
        s = np.array(env.observe(), dtype=np.float64)
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                # a = 0 buy a = 1 sell (가칭)
                # env에서 다음 상태, 현재 상태에서의 행동의 보상, 종료 여부(True, False), info? 안씀
                r = env.step(a) # t+1
                s_prime = np.array(env.get_train_ob(), dtype=np.float64) # t+1 cur
                if(env.get_next_price() is None):
                    done = True # t+2
                model.put_data((s, a, r, s_prime, prob[a].item(), done))
                s = s_prime
                score += r
                if done:
                    break
            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : \t{:.1f}".format(n_epi, score / print_interval))
            score = 0.0
        torch.save(model.state_dict(), save_path)

def test(load_path_txt):
    load_path = "models/" + load_path_txt
    # 환경 초기화
    train_date = "20210730"
    code = "001550"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
    chart_data_list = []
    train_data_list = []
    filepath = "data/sample.txt"
    # 없는 경우 넘어감
    if not os.path.isfile(filepath):
        # error
        exit(0)
    f = open(filepath, "r", encoding="utf8")
    lines = f.readlines()
    for line in lines:
        code_list = line.strip().split("\t")
    f.close()
    for code in code_list:
        temp_chart_data, temp_training_data = data_manager.load_data(
            os.path.join(BASE_DIR, 'data/{}_min_data.txt'.format(code)), ver='ppo',
            start_time="20210730091000", end_time="20210730150000")
        chart_data_list.append(copy.deepcopy(temp_chart_data))
        train_data_list.append(copy.deepcopy(temp_training_data))
    env = environment.Environment(chart_data=None, training_data=None)
    ### model 불러오기
    model = PPO()
    model.load_state_dict(torch.load(load_path))
    model.eval()
    score = 0.0
    print_interval = 50
    change_interval = 1
    for n_epi in range(10000):
        # 초기 상태 세팅
        env.reset()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
        if n_epi % change_interval == 0:
            temp_random = random.randrange(0, len(chart_data_list))
            env.set_chart_data(chart_data_list[temp_random])
            env.set_training_data(train_data_list[temp_random])
        #s = np.array(env.observe(), dtype="float32")
        s = np.array(env.observe(), dtype=np.float64)
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                # a = 0 buy a = 1 sell (가칭)
                # env에서 다음 상태, 현재 상태에서의 행동의 보상, 종료 여부(True, False), info? 안씀
                r = env.step(a) # t+1
                s_prime = np.array(env.get_train_ob(), dtype=np.float64) # t+1 cur
                if(env.get_next_price() is None):
                    done = True # t+2
                #model.put_data((s, a, r, s_prime, prob[a].item(), done))
                s = s_prime
                score += r
                if done:
                    break
            #model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : \t{:.1f}".format(n_epi, score / print_interval))
            score = 0.0

def train_simul(save_path_txt):
    code_list = []
    ### 신경망 로딩
    save_path = "models/" + save_path_txt
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
    chart_data_list = []
    train_data_list = []
    filepath = os.path.join(BASE_DIR, "ppo_learning/20210806/20210806_27_rotate_50.txt")
    # 없는 경우 넘어감
    if not os.path.isfile(filepath):
        # error
        exit(0)
    f = open(filepath, "r", encoding="utf8")
    lines = f.readlines()
    for line in lines:
        code_list.append(line.strip())
    f.close()
    print(len(code_list))
    for code in code_list:
        temp_chart_data, temp_training_data = data_manager.load_data(
            os.path.join(BASE_DIR, 'ppo_learning/20210806/{}_min_data.txt'.format(code)), ver='ppo',
            start_time="20210805091000", end_time="20210805150000")
        chart_data_list.append(copy.deepcopy(temp_chart_data))
        train_data_list.append(copy.deepcopy(temp_training_data))
    env = environment.Environment(chart_data=None, training_data=None)
    ### model 불러오기
    model = PPO()
    # model.load_state_dict(torch.load(load_path))
    # model.eval()
    score = 0.0
    print_interval = 50
    change_interval = 1
    #SIMUL_CHARGE = 0.0035
    SIMUL_CHARGE = 0
    buy_sell_count = 0
    buy_buy_count = 0
    sell_buy_count = 0
    sell_sell_count = 0
    for n_epi in range(10000):
        # 초기 상태 세팅
        env.reset()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
        if n_epi % change_interval == 0:
            temp_random = random.randrange(0, len(chart_data_list))
            env.set_chart_data(chart_data_list[temp_random])
            env.set_training_data(train_data_list[temp_random])
        # s = np.array(env.observe(), dtype="float32")
        s = np.array(env.observe(), dtype=np.float64)
        done = False
        prev_a = 1
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                # a = 0 buy a = 1 sell (가칭)
                # env에서 다음 상태, 현재 상태에서의 행동의 보상, 종료 여부(True, False), info? 안씀
                r = env.step(a)  # t+1
                # r은 현재 내 action을 고려한 점수 action 0 buy action 1 sell 일때, buy의 경우 수익률 sell의 경우 역수익률
                if (prev_a == 0): ## 전에 산 것인지
                    if(a == 0): #현재 산 것인지
                        buy_buy_count += 1
                        pass
                    else:
                        buy_sell_count += 1
                        r = r - (TRADING_TAX + TRADING_CHARGE + SIMUL_CHARGE) * 100
                else: ## 전에 판 것인지
                    if (a == 0):  # 현재 산 것인지
                        sell_buy_count += 1
                        r = r - (TRADING_CHARGE + SIMUL_CHARGE) * 100
                    else:
                        sell_sell_count += 1
                        r = 0
                prev_a = a
                s_prime = np.array(env.get_train_ob(), dtype=np.float64)  # t+1 cur
                if (env.get_next_price() is None):
                    done = True  # t+2
                model.put_data((s, a, r, s_prime, prob[a].item(), done))
                s = s_prime
                score += r
                if done:
                    break
            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : \t{:.1f}\t, buy buy count : \t{}\t, buy sell count : \t{}\t,"
                  " sell buy count : \t{}\t, sell sell count : \t{}\t"
                  .format(n_epi, score / print_interval, buy_buy_count / print_interval, buy_sell_count / print_interval,
                          sell_buy_count / print_interval, sell_sell_count / print_interval))
            score = 0.0
            buy_sell_count = 0
            buy_buy_count = 0
            sell_buy_count = 0
            sell_sell_count = 0
            torch.save(model.state_dict(), save_path)
    ### 환경 설정 -

def real_simul(load_path_txt):
    code_list = []
    ### 신경망 로딩
    load_path = "models/" + load_path_txt
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
    chart_data_list = []
    train_data_list = []
    filepath = os.path.join(BASE_DIR, "ppo_learning/20210806/20210806_27_rotate_50.txt")
    # 없는 경우 넘어감
    if not os.path.isfile(filepath):
        # error
        exit(0)
    f = open(filepath, "r", encoding="utf8")
    lines = f.readlines()
    for line in lines:
        code_list.append(line.strip())
    f.close()
    for code in code_list:
        temp_chart_data, temp_training_data = data_manager.load_data(
            os.path.join(BASE_DIR, 'ppo_learning/20210806/{}_min_data.txt'.format(code)), ver='ppo',
            start_time="20210806091000", end_time="20210806150000")
        chart_data_list.append(copy.deepcopy(temp_chart_data))
        train_data_list.append(copy.deepcopy(temp_training_data))
    env = environment.Environment(chart_data=None, training_data=None)

    ### model 불러오기
    model = PPO()
    model.load_state_dict(torch.load(load_path))
    model.eval()
    score = 0.0
    print_interval = 50
    change_interval = 1
    #SIMUL_CHARGE = 0.0035
    SIMUL_CHARGE = 0
    buy_sell_count = 0
    buy_buy_count = 0
    sell_buy_count = 0
    sell_sell_count = 0
    for n_epi in range(10000):
        # 초기 상태 세팅
        env.reset()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
        if n_epi % change_interval == 0:
            temp_random = random.randrange(0, len(chart_data_list))
            env.set_chart_data(chart_data_list[temp_random])
            env.set_training_data(train_data_list[temp_random])
        # s = np.array(env.observe(), dtype="float32")
        s = np.array(env.observe(), dtype=np.float64)
        done = False
        prev_a = 1
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                #probs = prob.tolist()
                #print("a1 : " + str(a1[0]))
                m = Categorical(prob)
                a = m.sample().item()
                #print("a : " + str(a))
                # if(str(a1[0]) != str(a)):
                #     print("error")
                # a = 0 buy a = 1 sell (가칭)
                # env에서 다음 상태, 현재 상태에서의 행동의 보상, 종료 여부(True, False), info? 안씀
                r = env.step(a)  # t+1
                # r은 현재 내 action을 고려한 점수 action 0 buy action 1 sell 일때, buy의 경우 수익률 sell의 경우 역수익률
                if (prev_a == 0):  ## 전에 산 것인지
                    if (a == 0):  # 현재 산 것인지
                        buy_buy_count += 1
                        pass
                    else:
                        buy_sell_count += 1
                        r = r - (TRADING_TAX + TRADING_CHARGE + SIMUL_CHARGE) * 100
                else:  ## 전에 판 것인지
                    if (a == 0):  # 현재 산 것인지
                        sell_buy_count += 1
                        r = r - (TRADING_CHARGE + SIMUL_CHARGE) * 100
                    else:
                        sell_sell_count += 1
                        r = 0
                prev_a = a
                s_prime = np.array(env.get_train_ob(), dtype=np.float64)  # t+1 cur
                if (env.get_next_price() is None):
                    done = True  # t+2
                # model.put_data((s, a, r, s_prime, prob[a].item(), done))
                s = s_prime
                score += r
                if done:
                    break
            # model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : \t{:.1f}\t, buy buy count : \t{}\t, buy sell count : \t{}\t,"
                  " sell buy count : \t{}\t, sell sell count : \t{}\t"
                  .format(n_epi, score / print_interval, buy_buy_count / print_interval,
                          buy_sell_count / print_interval,
                          sell_buy_count / print_interval, sell_sell_count / print_interval))
            score = 0.0
            buy_sell_count = 0
            buy_buy_count = 0
            sell_buy_count = 0
            sell_sell_count = 0
    ### 환경 설정 -

def print_model_weights(model_name):
    model = PPO()
    model.load_state_dict(torch.load(model_name))
    #####
    f = open("models/paramst.txt", 'w', encoding='utf8')
    f.close()
    f = open("models/paramst.txt", 'a', encoding='utf8')
    for param in model.parameters():
        params = param.tolist()
        for i in (params):
            if(isinstance(i, list)):
                for j in i:
                    f.write(str(j) + "\t")
            else:
                f.write(str(i) + "\t")
            f.write("\n")
    f.close()

##################

def lstm_train_simul(save_path_txt):
    code_list = []
    ### 신경망 로딩
    save_path = "models/" + save_path_txt
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
    chart_data_list = []
    train_data_list = []
    filepath = os.path.join(BASE_DIR, "ppo_learning/20210806/20210806_27_rotate_50.txt")
    # 없는 경우 넘어감
    if not os.path.isfile(filepath):
        # error
        exit(0)
    f = open(filepath, "r", encoding="utf8")
    lines = f.readlines()
    for line in lines:
        code_list.append(line.strip())
    f.close()
    print(len(code_list))
    for code in code_list:
        temp_chart_data, temp_training_data = data_manager.load_data(
            os.path.join(BASE_DIR, 'ppo_learning/20210806/{}_min_data.txt'.format(code)), ver='ppo',
            start_time="20210805091000", end_time="20210805150000")
        chart_data_list.append(copy.deepcopy(temp_chart_data))
        train_data_list.append(copy.deepcopy(temp_training_data))
    env = environment.Environment(chart_data=None, training_data=None)
    ### model 불러오기
    model = PPO_lstm()
    # model.load_state_dict(torch.load(load_path))
    # model.eval()
    score = 0.0
    print_interval = 50
    change_interval = 50
    #SIMUL_CHARGE = 0.0035
    SIMUL_CHARGE = 0
    buy_sell_count = 0
    buy_buy_count = 0
    sell_buy_count = 0
    sell_sell_count = 0
    for n_epi in range(10000):
        # 초기 상태 세팅
        ### 추가
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        ###
        env.reset()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
        if n_epi % change_interval == 0:
            temp_random = random.randrange(0, len(chart_data_list))
            env.set_chart_data(chart_data_list[temp_random])
            env.set_training_data(train_data_list[temp_random])
        # s = np.array(env.observe(), dtype="float32")
        s = np.array(env.observe(), dtype=np.float64)
        done = False
        prev_a = 1
        while not done:
            for t in range(T_horizon):
                ###
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
                ###
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                # a = 0 buy a = 1 sell (가칭)
                # env에서 다음 상태, 현재 상태에서의 행동의 보상, 종료 여부(True, False), info? 안씀
                r = env.step(a)  # t+1
                # r은 현재 내 action을 고려한 점수 action 0 buy action 1 sell 일때, buy의 경우 수익률 sell의 경우 역수익률
                if (prev_a == 0): ## 전에 산 것인지
                    if(a == 0): #현재 산 것인지
                        buy_buy_count += 1
                        pass
                    else:
                        buy_sell_count += 1
                        r = r - (TRADING_TAX + TRADING_CHARGE + SIMUL_CHARGE) * 100
                else: ## 전에 판 것인지
                    if (a == 0):  # 현재 산 것인지
                        sell_buy_count += 1
                        r = r - (TRADING_CHARGE + SIMUL_CHARGE) * 100
                    else:
                        sell_sell_count += 1
                        r = 0
                prev_a = a
                s_prime = np.array(env.get_train_ob(), dtype=np.float64)  # t+1 cur
                if (env.get_next_price() is None):
                    done = True  # t+2
                ###
                model.put_data((s, a, r, s_prime, prob[a].item(), h_in, h_out, done))
                ###
                s = s_prime
                score += r
                if done:
                    break
            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : \t{:.1f}\t, buy buy count : \t{}\t, buy sell count : \t{}\t,"
                  " sell buy count : \t{}\t, sell sell count : \t{}\t"
                  .format(n_epi, score / print_interval, buy_buy_count / print_interval, buy_sell_count / print_interval,
                          sell_buy_count / print_interval, sell_sell_count / print_interval))
            score = 0.0
            buy_sell_count = 0
            buy_buy_count = 0
            sell_buy_count = 0
            sell_sell_count = 0
            torch.save(model.state_dict(), save_path)
    ### 환경 설정 -

def lstm_real_simul(load_path_txt):
    code_list = []
    ### 신경망 로딩
    load_path = "models/" + load_path_txt
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
    chart_data_list = []
    train_data_list = []
    filepath = os.path.join(BASE_DIR, "ppo_learning/20210806/20210806_27_rotate_50.txt")
    # 없는 경우 넘어감
    if not os.path.isfile(filepath):
        # error
        exit(0)
    f = open(filepath, "r", encoding="utf8")
    lines = f.readlines()
    for line in lines:
        code_list.append(line.strip())
    f.close()
    for code in code_list:
        temp_chart_data, temp_training_data = data_manager.load_data(
            os.path.join(BASE_DIR, 'ppo_learning/20210806/{}_min_data.txt'.format(code)), ver='ppo',
            start_time="20210806091000", end_time="20210806150000")
        chart_data_list.append(copy.deepcopy(temp_chart_data))
        train_data_list.append(copy.deepcopy(temp_training_data))
    env = environment.Environment(chart_data=None, training_data=None)
    ### model 불러오기
    model = PPO()
    model.load_state_dict(torch.load(load_path))
    model.eval()
    score = 0.0
    print_interval = 50
    change_interval = 1
    #SIMUL_CHARGE = 0.0035
    SIMUL_CHARGE = 0
    buy_sell_count = 0
    buy_buy_count = 0
    sell_buy_count = 0
    sell_sell_count = 0
    for n_epi in range(10000):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        # 초기 상태 세팅
        env.reset()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
        if n_epi % change_interval == 0:
            temp_random = random.randrange(0, len(chart_data_list))
            env.set_chart_data(chart_data_list[temp_random])
            env.set_training_data(train_data_list[temp_random])
        # s = np.array(env.observe(), dtype="float32")
        s = np.array(env.observe(), dtype=np.float64)
        done = False
        prev_a = 1
        while not done:
            for t in range(T_horizon):
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
                prob = prob.view(-1)
                #probs = prob.tolist()
                #print("a1 : " + str(a1[0]))
                m = Categorical(prob)
                a = m.sample().item()
                #print("a : " + str(a))
                # if(str(a1[0]) != str(a)):
                #     print("error")
                # a = 0 buy a = 1 sell (가칭)
                # env에서 다음 상태, 현재 상태에서의 행동의 보상, 종료 여부(True, False), info? 안씀
                r = env.step(a)  # t+1
                # r은 현재 내 action을 고려한 점수 action 0 buy action 1 sell 일때, buy의 경우 수익률 sell의 경우 역수익률
                if (prev_a == 0):  ## 전에 산 것인지
                    if (a == 0):  # 현재 산 것인지
                        buy_buy_count += 1
                        pass
                    else:
                        buy_sell_count += 1
                        r = r - (TRADING_TAX + TRADING_CHARGE + SIMUL_CHARGE) * 100
                else:  ## 전에 판 것인지
                    if (a == 0):  # 현재 산 것인지
                        sell_buy_count += 1
                        r = r - (TRADING_CHARGE + SIMUL_CHARGE) * 100
                    else:
                        sell_sell_count += 1
                        r = 0
                prev_a = a
                s_prime = np.array(env.get_train_ob(), dtype=np.float64)  # t+1 cur
                if (env.get_next_price() is None):
                    done = True  # t+2
                # model.put_data((s, a, r, s_prime, prob[a].item(), h_in, h_out, done))
                s = s_prime
                score += r
                if done:
                    break
            # model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : \t{:.1f}\t, buy buy count : \t{}\t, buy sell count : \t{}\t,"
                  " sell buy count : \t{}\t, sell sell count : \t{}\t"
                  .format(n_epi, score / print_interval, buy_buy_count / print_interval,
                          buy_sell_count / print_interval,
                          sell_buy_count / print_interval, sell_sell_count / print_interval))
            score = 0.0
            buy_sell_count = 0
            buy_buy_count = 0
            sell_buy_count = 0
            sell_sell_count = 0
    ### 환경 설정 -

##################

def reward_up_train_simul(save_path_txt):
    code_list = []
    ### 신경망 로딩
    save_path = "models/" + save_path_txt
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
    chart_data_list = []
    train_data_list = []
    filepath = os.path.join(BASE_DIR, "ppo_learning/20210806/20210806_27_rotate_50.txt")
    # 없는 경우 넘어감
    if not os.path.isfile(filepath):
        # error
        exit(0)
    f = open(filepath, "r", encoding="utf8")
    lines = f.readlines()
    for line in lines:
        code_list.append(line.strip())
    f.close()
    print(len(code_list))
    for code in code_list:
        temp_chart_data, temp_training_data = data_manager.load_data(
            os.path.join(BASE_DIR, 'ppo_learning/20210806/{}_min_data.txt'.format(code)), ver='ppo',
            start_time="20210805091000", end_time="20210805150000")
        chart_data_list.append(copy.deepcopy(temp_chart_data))
        train_data_list.append(copy.deepcopy(temp_training_data))
    env = environment.Environment(chart_data=None, training_data=None)
    ### model 불러오기
    model = PPO()
    # model.load_state_dict(torch.load(load_path))
    # model.eval()
    score = 0.0
    print_interval = 50
    change_interval = 1
    #SIMUL_CHARGE = 0.0035
    SIMUL_CHARGE = 0
    buy_sell_count = 0
    buy_buy_count = 0
    sell_buy_count = 0
    sell_sell_count = 0
    for n_epi in range(10000):
        # 초기 상태 세팅
        env.reset()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
        if n_epi % change_interval == 0:
            temp_random = random.randrange(0, len(chart_data_list))
            env.set_chart_data(chart_data_list[temp_random])
            env.set_training_data(train_data_list[temp_random])
        # s = np.array(env.observe(), dtype="float32")
        s = np.array(env.observe(), dtype=np.float64)
        done = False
        prev_a = 1
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                # a = 0 buy a = 1 sell (가칭)
                # env에서 다음 상태, 현재 상태에서의 행동의 보상, 종료 여부(True, False), info? 안씀
                r = env.up_step(a)  # t+1
                # r은 현재 내 action을 고려한 점수 action 0 buy action 1 sell 일때, buy의 경우 수익률 sell의 경우 역수익률
                if (prev_a == 0): ## 전에 산 것인지
                    if(a == 0): #현재 산 것인지
                        buy_buy_count += 1
                        pass
                    else:
                        buy_sell_count += 1
                        r = r - (TRADING_TAX + TRADING_CHARGE + SIMUL_CHARGE) * 100
                else: ## 전에 판 것인지
                    if (a == 0):  # 현재 산 것인지
                        sell_buy_count += 1
                        r = r - (TRADING_CHARGE + SIMUL_CHARGE) * 100
                    else:
                        sell_sell_count += 1
                        r = 0
                prev_a = a
                s_prime = np.array(env.get_train_ob(), dtype=np.float64)  # t+1 cur
                if (env.get_next_price() is None):
                    done = True  # t+2
                model.put_data((s, a, r, s_prime, prob[a].item(), done))
                s = s_prime
                score += r
                if done:
                    break
            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : \t{:.1f}\t, buy buy count : \t{}\t, buy sell count : \t{}\t,"
                  " sell buy count : \t{}\t, sell sell count : \t{}\t"
                  .format(n_epi, score / print_interval, buy_buy_count / print_interval, buy_sell_count / print_interval,
                          sell_buy_count / print_interval, sell_sell_count / print_interval))
            score = 0.0
            buy_sell_count = 0
            buy_buy_count = 0
            sell_buy_count = 0
            sell_sell_count = 0
            torch.save(model.state_dict(), save_path)
    ### 환경 설정 -

def reward_up_real_simul(load_path_txt):
    code_list = []
    ### 신경망 로딩
    load_path = "models/" + load_path_txt
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
    chart_data_list = []
    train_data_list = []
    filepath = os.path.join(BASE_DIR, "ppo_learning/20210806/20210806_27_rotate_50.txt")
    # 없는 경우 넘어감
    if not os.path.isfile(filepath):
        # error
        exit(0)
    f = open(filepath, "r", encoding="utf8")
    lines = f.readlines()
    for line in lines:
        code_list.append(line.strip())
    f.close()
    for code in code_list:
        temp_chart_data, temp_training_data = data_manager.load_data(
            os.path.join(BASE_DIR, 'ppo_learning/20210806/{}_min_data.txt'.format(code)), ver='ppo',
            start_time="20210806091000", end_time="20210806150000")
        chart_data_list.append(copy.deepcopy(temp_chart_data))
        train_data_list.append(copy.deepcopy(temp_training_data))
    env = environment.Environment(chart_data=None, training_data=None)

    ### model 불러오기
    model = PPO()
    model.load_state_dict(torch.load(load_path))
    model.eval()
    score = 0.0
    print_interval = 50
    change_interval = 1
    #SIMUL_CHARGE = 0.0035
    SIMUL_CHARGE = 0
    buy_sell_count = 0
    buy_buy_count = 0
    sell_buy_count = 0
    sell_sell_count = 0
    for n_epi in range(10000):
        # 초기 상태 세팅
        env.reset()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 에피소드마다 주식종목이 달라짐
        if n_epi % change_interval == 0:
            temp_random = random.randrange(0, len(chart_data_list))
            env.set_chart_data(chart_data_list[temp_random])
            env.set_training_data(train_data_list[temp_random])
        # s = np.array(env.observe(), dtype="float32")
        s = np.array(env.observe(), dtype=np.float64)
        done = False
        prev_a = 1
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                #probs = prob.tolist()
                #print("a1 : " + str(a1[0]))
                m = Categorical(prob)
                a = m.sample().item()
                #print("a : " + str(a))
                # if(str(a1[0]) != str(a)):
                #     print("error")
                # a = 0 buy a = 1 sell (가칭)
                # env에서 다음 상태, 현재 상태에서의 행동의 보상, 종료 여부(True, False), info? 안씀
                r = env.step(a)  # t+1
                # r은 현재 내 action을 고려한 점수 action 0 buy action 1 sell 일때, buy의 경우 수익률 sell의 경우 역수익률
                if (prev_a == 0):  ## 전에 산 것인지
                    if (a == 0):  # 현재 산 것인지
                        buy_buy_count += 1
                        pass
                    else:
                        buy_sell_count += 1
                        r = r - (TRADING_TAX + TRADING_CHARGE + SIMUL_CHARGE) * 100
                else:  ## 전에 판 것인지
                    if (a == 0):  # 현재 산 것인지
                        sell_buy_count += 1
                        r = r - (TRADING_CHARGE + SIMUL_CHARGE) * 100
                    else:
                        sell_sell_count += 1
                        r = 0
                prev_a = a
                s_prime = np.array(env.get_train_ob(), dtype=np.float64)  # t+1 cur
                if (env.get_next_price() is None):
                    done = True  # t+2
                # model.put_data((s, a, r, s_prime, prob[a].item(), done))
                s = s_prime
                score += r
                if done:
                    break
            # model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : \t{:.1f}\t, buy buy count : \t{}\t, buy sell count : \t{}\t,"
                  " sell buy count : \t{}\t, sell sell count : \t{}\t"
                  .format(n_epi, score / print_interval, buy_buy_count / print_interval,
                          buy_sell_count / print_interval,
                          sell_buy_count / print_interval, sell_sell_count / print_interval))
            score = 0.0
            buy_sell_count = 0
            buy_buy_count = 0
            sell_buy_count = 0
            sell_sell_count = 0
    ### 환경 설정 -

if __name__ == '__main__':
    #train("vir_ch100_sp1_rat100_osspall")
    #test("vir_ch100_sp1_rat100_osspall")
    #train_simul("test3")
    #train_simul("real_ch100_rat100_sp1_ppo4")
    #train("vir_ch100_sp3")
    #test("vir_ch100_sp3")
    # train_simul("real_2")
    # real_simul("real_2")
    #lstm_train_simul("real_lstm_0")
    #lstm_real_simul("real_lstm_0")
    reward_up_train_simul("up_test1")
    #reward_up_real_simul("up_")
    #reward_down_train_simul("up_")
    #reward_down_real_simul("up_")
    #reward_merge_real_simul("up", "down")
    #print_model_weights("models/real_ch100_rat100_sp1_ppo")