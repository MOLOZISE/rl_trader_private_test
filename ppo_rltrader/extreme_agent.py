class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 1  # 자신의 보유 비율 (0 or 1 / 0이면 0개 1이면 최대치)

    # 매매 수수료 및 세금
    # TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    # TRADING_TAX = 0.0025  # 거래세 0.25%
    # SIMUL_CHARGE = 0.0035
    SIMUL_CHARGE = 0
    TRADING_CHARGE = 0
    TRADING_TAX = 0
    # SIMUL_CHARGE = 0
    # # 행동
    ACTION_ZERO = 0  # 보유 비율 0으로 하기
    ACTION_ONE = 1  # 보유 비율 1로 하기
    # ACTION_HOLD = 2  # 홀딩
    # # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_ZERO, ACTION_ONE]  # , ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(
            self, environment, min_trading_unit=1, max_trading_unit=2):
        # Environment 객체
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.change_portfolio_value = 0  # 바로 직전의 PV
        self.judge_portfolio_value = 0
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
        self.profitloss = 0  # 현재 손익
        self.base_profitloss = 0  # 일정 손익률을 축적하는 손익률
        self.change_profitloss = 0  # 바로 전과 비교하는 손익률
        self.exploration_base = 0  # 탐험 행동 결정 기준
        self.judge_profitloss = 0 # 행동의 이점 기준 판단

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        # self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.change_portfolio_value = self.initial_balance
        self.judge_portfolio_value = self.initial_balance
        self.judge_profitloss = 0

        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())
        return self.ratio_hold

    def advanced_change_trading_unit(self, curr_price):
        self.max_trading_unit = max(int(self.portfolio_value / curr_price), 1)

    def accurate_get_states(self):
        self.ratio_hold = self.num_stocks / self.max_trading_unit
        return self.ratio_hold

    def accurate_change_trading_unit(self, curr_price):
        self.max_trading_unit = max(
            int(self.portfolio_value / (curr_price * (1 + self.TRADING_CHARGE + self.SIMUL_CHARGE))), 1)

    def set_environment(self, environment):
        self.environment = environment

    ################################################
    def build_sample(self):
        # 이미 observe에서 idx+1을 함 아마 둘이 같기 때문에 None을 반환하면 길이도 맞을 것임
        choose = self.environment.observe()
        if choose is not None:
            if len(self.environment.training_data) > self.environment.idx:
                sample = self.environment.training_data.iloc[self.environment.idx].tolist()
                sample.extend([self.get_states()]) # agent의 STATEDIM = 1
                return sample
        return None

    ################################################
    def newstep(self, action, prob_a=None):
        before_action = self.ratio_hold
        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 최대 매수 단위 변경
        self.advanced_change_trading_unit(curr_price)

        # action이 0일때, other은 1 ation이 1일때 other은 0
        maintain_stocks = action * self.max_trading_unit
        other_maintain_stocks = (1 - action) * self.max_trading_unit

        trading_unit = self.num_stocks - maintain_stocks
        # 다른 선택지 확인
        ###############################################################################################
        other_trading_unit = self.num_stocks - other_maintain_stocks
        other_balance = self.balance
        other_stocks = self.num_stocks

        if other_trading_unit > 0:  # other_trading_unit만큼 매도
            sell_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE + self.SIMUL_CHARGE)) * other_trading_unit
            other_balance += sell_amount
            other_stocks -= other_trading_unit
        elif other_trading_unit < 0:  # trading_unit만큼 매수
            other_trading_unit = -other_trading_unit
            invest_amount = curr_price * (1 + self.TRADING_CHARGE + self.SIMUL_CHARGE) * other_trading_unit
            other_balance -= invest_amount
            other_stocks += other_trading_unit
        ###############################################################################################

        if trading_unit > 0:  # trading_unit만큼 매도
            sell_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE + self.SIMUL_CHARGE)) * trading_unit
            self.balance += sell_amount
            self.num_stocks -= trading_unit
            self.num_sell += 1
        elif trading_unit < 0:  # trading_unit만큼 매수
            trading_unit = -trading_unit
            invest_amount = curr_price * (1 + self.TRADING_CHARGE + self.SIMUL_CHARGE) * trading_unit
            self.balance -= invest_amount
            self.num_stocks += trading_unit
            self.num_buy += 1
        else:
            self.num_hold += 1

        # 즉시 보상 초기화
        self.immediate_reward = 0
        next_price = self.environment.get_next_price()
        if next_price is None:
            next_price = curr_price
        uprate = (next_price - curr_price) / curr_price
        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + next_price * self.num_stocks
        self.profitloss = ((self.portfolio_value - self.initial_balance) / self.initial_balance)
        s_prime = self.build_sample()
        if s_prime is None:
            done = True
        else:
            done = False
        #######################
        other_portfolio_value = other_balance + next_price * other_stocks
        other_change_profitloss = ((other_portfolio_value - self.change_portfolio_value) / self.change_portfolio_value)
        ######################
        self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        self.change_profitloss = ((self.portfolio_value - self.change_portfolio_value) / self.change_portfolio_value)
        self.change_portfolio_value = self.portfolio_value  # 바로 전 단계의 portfolio_value를 가짐

        # 즉시 보상 - 수익률
        # self.immediate_reward = self.profitloss
        # 지연 보상 - 익절, 손절 기준
        # self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        # if self.base_profitloss > 0.02:
        #   self.base_portfolio_value = self.portfolio_value
        # if self.base_profitloss < -0.02: #or self.base_profitloss > 0.03:
        # #     # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
        # #     # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
        #    self.base_portfolio_value = self.portfolio_value
        #    done = True # 기준치 손익률 달성 시 종료
        # 보상을 최대화하는 것이 목적
        # 그러나 주식을 사지않고 계속 매도만 하는 경우가 발생 가능(예를 들면 계속 하락장인 경우, 이득을 볼 수 없기 때문에)
        # 그런데 그 상황에 최적화되고(즉, 하락하는 것에서 관망만 하는 것이 이득이라고 생각하게 됨)
        # 빠져나오지 못할 수 있기 때문에, 보상의 선정을 다르게 함
        # 보상 = 얻을 수 있는 최대의 이익과 현재의 이익이 가까울 수록 높음
        # 위험한 투자자의 손익률 : 등락률(next - current / current) * 최대 보유 수 = 정확히는 모든 돈이 주식으로 넘어간 경우
        # 보상 = 손익률 - 위투손 (보상은 거의 -로 나올 것) 근데 이거 (현재 보유 수 - 최대 보유 수) * 등락률 아님?
        # 결국 (float(action/10) - 1.0) * uprate
        # self.immediate_reward = float((action - 4.5)/10) * uprate * 100 * 0.1 + self.profitloss * 100 * 0.9
        # 엄청난 손해 후에 이득을 보는 행동을 하여도 변함없이 손실되었다는 이유만으로 점수를 낮게 준다면 발전할 수 없을 것이라 판단함
        # self.immediate_reward = self.base_profitloss
        self.immediate_reward = (self.change_profitloss - other_change_profitloss) * 10
        #self.immediate_reward = (self.change_profitloss - other_change_profitloss + self.profitloss) * 10
        #self.immediate_reward = (self.change_profitloss - other_change_profitloss - ((before_action) * self.TRADING_TAX + self.TRADING_CHARGE + self.SIMUL_CHARGE)) * 100
        # profitoss대신 base로 계산
        # self.base_portfolio_value = self.portfolio_value
        # self.immediate_reward = self.base_profitloss# + self.profitloss * 10
        reward = self.immediate_reward
        return s_prime, reward, done
    ################################################
    def simulstep(self, action):
        curr_price = self.environment.get_price()
        other_balance = self.balance
        other_stocks = self.num_stocks
        changedecision = False
        consecutivezero = False
        consecutiveone = False
        # action을 받으면
        # portfolio_value계산...
        if self.num_stocks > 0:
            before_action = 1
            max_trading_unit = self.num_stocks

            if action == 1:  # other_trading_unit만큼 매도
                sell_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE + self.SIMUL_CHARGE)) * max_trading_unit
                other_balance += sell_amount
                other_stocks -= max_trading_unit
                self.num_hold += 1
                consecutiveone = True
                changedecision = False

            if action == 0:
                sell_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE + self.SIMUL_CHARGE)) * max_trading_unit
                self.balance += sell_amount
                self.num_stocks -= max_trading_unit
                self.num_sell += 1
                changedecision = True

        else:
            before_action = 0
            max_trading_unit = max(int(self.portfolio_value / (curr_price * (1 + self.TRADING_CHARGE + self.SIMUL_CHARGE))), 1)

            if action == 0:  # trading_unit만큼 매수 else nothing
                invest_amount = curr_price * (1 + self.TRADING_CHARGE + self.SIMUL_CHARGE) * max_trading_unit
                other_balance -= invest_amount
                other_stocks += max_trading_unit
                self.num_hold += 1
                consecutivezero = True
                changedecision = False

            if action == 1:
                invest_amount = curr_price * (1 + self.TRADING_CHARGE + self.SIMUL_CHARGE) * max_trading_unit
                self.balance -= invest_amount
                self.num_stocks += max_trading_unit
                self.num_buy += 1
                changedecision = True

        if not (consecutivezero or consecutiveone):
            self.judge_profitloss = 0
        # 즉시 보상 초기화
        self.immediate_reward = 0
        next_price = self.environment.get_next_price()
        if next_price is None:
            next_price = curr_price
        #uprate = (next_price - curr_price) / curr_price
        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + next_price * self.num_stocks
        self.profitloss = ((self.portfolio_value - self.initial_balance) / self.initial_balance)
        s_prime = self.build_sample()
        if s_prime is None:
            done = True
        else:
            done = False
        #######################
        other_portfolio_value = other_balance + next_price * other_stocks
        other_profitloss = ((other_portfolio_value - self.initial_balance) / self.initial_balance)
        other_change_profitloss = ((other_portfolio_value - self.change_portfolio_value) / self.change_portfolio_value)
        other_base_profitloss = ((other_portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        ######################
        self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        self.change_profitloss = ((self.portfolio_value - self.change_portfolio_value) / self.change_portfolio_value)
        self.change_portfolio_value = self.portfolio_value  # 바로 전 단계의 portfolio_value를 가짐
        #self.judge_profitloss = ((self.judge_portfolio_value - self.initial_balance)/self.initial_balance)

        # 즉시 보상 - 수익률
        # self.immediate_reward = self.profitloss
        # 지연 보상 - 익절, 손절 기준
        # self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        # if self.base_profitloss > 0.03:
        #     self.base_portfolio_value = self.portfolio_value
        # if self.base_profitloss < -0.03:  # or self.base_profitloss > 0.03:
        #     #     # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
        #     #     # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
        #     self.base_portfolio_value = self.portfolio_value
        #     done = True  # 기준치 손익률 달성 시 종료
        # 보상을 최대화하는 것이 목적
        # 그러나 주식을 사지않고 계속 매도만 하는 경우가 발생 가능(예를 들면 계속 하락장인 경우, 이득을 볼 수 없기 때문에)
        # 그런데 그 상황에 최적화되고(즉, 하락하는 것에서 관망만 하는 것이 이득이라고 생각하게 됨)
        # 빠져나오지 못할 수 있기 때문에, 보상의 선정을 다르게 함
        # 보상 = 얻을 수 있는 최대의 이익과 현재의 이익이 가까울 수록 높음
        # 위험한 투자자의 손익률 : 등락률(next - current / current) * 최대 보유 수 = 정확히는 모든 돈이 주식으로 넘어간 경우
        # 보상 = 손익률 - 위투손 (보상은 거의 -로 나올 것) 근데 이거 (현재 보유 수 - 최대 보유 수) * 등락률 아님?
        # 결국 (float(action/10) - 1.0) * uprate
        # self.immediate_reward = float((action - 4.5)/10) * uprate * 100 * 0.1 + self.profitloss * 100 * 0.9
        # 엄청난 손해 후에 이득을 보는 행동을 하여도 변함없이 손실되었다는 이유만으로 점수를 낮게 준다면 발전할 수 없을 것이라 판단함
        # self.immediate_reward = self.base_profitloss
        #self.immediate_reward = (self.change_profitloss - other_change_profitloss) * 10
        # self.immediate_reward = (self.change_profitloss - other_change_profitloss + self.profitloss) * 10
        #self.immediate_reward = (self.change_profitloss - other_change_profitloss - ((before_action) * self.TRADING_TAX + self.TRADING_CHARGE + self.SIMUL_CHARGE)) * 1000
        #self.immediate_reward = (self.base_profitloss - other_base_profitloss - ((before_action) * self.TRADING_TAX + self.TRADING_CHARGE + self.SIMUL_CHARGE)) * 100
        #self.immediate_reward = (self.profitloss - other_profitloss - ((before_action) * self.TRADING_TAX + self.TRADING_CHARGE + self.SIMUL_CHARGE)) * 100
        #self.immediate_reward = (self.profitloss - other_profitloss) * 100
        # 일관된 0은 judge_profitloss를 0으로 만들고
        # 1이 1번이라도 있는 순간 반드시 changedecision이 일어나면서 judge_profitloss가 어떤 값으로 고정됨
        ###################################################################################################
        taxpenalty = (before_action) * self.TRADING_TAX + self.TRADING_CHARGE + self.SIMUL_CHARGE
        # if consecutivezero:
        #     # 0 -> 0
        #     # profitloss = 0 / self.judge에는 1로 했을 때의 이득 + 수수료벌점 완화해서
        #     # 다음에도 같으면 점차 늘어남 -> 보상은 taxpenalty 빼줌(그니까 초기에는 0 -> 0이 좋을 수도 있음)
        #     # 처음에는 반드시 0이므로, 0이 나올 때, other_profitloss가 누적이 되어야 함
        #     # 왜 안 샀어??? 페널티
        #     # 잘 안 샀어
        #     #self.judge_profitloss += (other_change_profitloss + taxpenalty)
        #     self.judge_profitloss = (other_change_profitloss)
        #     #self.judge_profitloss = other_base_profitloss
        #     #self.immediate_reward = (-1) * (self.judge_profitloss - taxpenalty) * 10
        #     self.immediate_reward = (-1) * (self.judge_profitloss) * 10
        # elif consecutiveone:
        #     # 1 -> 1
        #     # profitloss = 변동값 / other_profitloss 0 - 수수료(before_action)
        #     self.judge_profitloss += (self.change_profitloss)
        #     #self.judge_profitloss = (self.change_profitloss)
        #     #self.judge_profitloss = self.base_profitloss
        #     self.immediate_reward = (self.judge_profitloss) * 10
        # else:
        #     # 0 -> 1
        #     # profitloss = 변동값 - 수수료 / other_profitloss 0
        #     # 1 -> 0
        #     # profitloss = 0 - 수수료(before_action) / other_profitloss 변동값
        #     self.immediate_reward = (self.change_profitloss - other_change_profitloss) * 10
        #     #self.immediate_reward = (self.base_profitloss - other_base_profitloss) * 100
        ##################################################################################################

        # if changedecision:
        #     self.judge_portfolio_value = self.portfolio_value
        #     self.immediate_reward = (self.profitloss - other_profitloss) * 100
        # else:
        #     self.immediate_reward = ((self.profitloss - other_profitloss) + (2 * action - 1) * ((action) * self.profitloss + (1 - action) * other_profitloss - self.judge_profitloss)) * 100

        # if self.profitloss <= 0:
        #
        # elif other_profitloss <= 0:
        #
        # if other_profitloss <= 0:
        #     other_profitloss = 0
        self.immediate_reward = (self.profitloss - other_profitloss + taxpenalty) * 100
        # profitoss대신 base로 계산
        # self.base_portfolio_value = self.portfolio_value
        # self.immediate_reward = self.base_profitloss# + self.profitloss * 10
        # if self.immediate_reward > 0:
        #     self.immediate_reward = 0
        #self.immediate_reward = self.profitloss * 100
        reward = self.immediate_reward
        return s_prime, reward, done