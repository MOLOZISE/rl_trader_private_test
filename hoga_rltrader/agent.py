import numpy as np
import utils

def new_softmax(policylist):
    c = np.max(policylist)  # 최댓값
    exp_a = np.exp(policylist - c)  # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 0 # 주식 보유 비율, 포트폴리오 가치 비율 반영 X

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    TRADING_TAX = 0.0025  # 거래세 0.25%
    SIMUL_CHARGE = 0.0035

    # # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    # # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]#, ACTION_HOLD]
    # ACTION_ZERO = 0
    # ACTION_ONE = 1
    # ACTION_TWO = 2
    # ACTION_THREE = 3
    # ACTION_FOUR = 4
    # ACTION_FIVE = 5
    # ACTION_SIX = 6
    # ACTION_SEVEN = 7
    # ACTION_EIGHT = 8
    # ACTION_NINE = 9
    # ACTIONS = [ACTION_ZERO, ACTION_ONE, ACTION_TWO, ACTION_THREE, ACTION_FOUR, ACTION_FIVE, ACTION_SIX, ACTION_SEVEN, ACTION_EIGHT, ACTION_NINE]
    #NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수
    NUM_ACTIONS = len(ACTIONS)

    def __init__(
            self, environment, min_trading_unit=1, max_trading_unit=2,
            delayed_reward_threshold=.05, importance=0.9):
        # Environment 객체
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        # 지연보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
        self.profitloss = 0  # 현재 손익
        self.base_profitloss = 0  # 직전 지연 보상 이후 손익
        self.exploration_base = 0  # 탐험 행동 결정 기준
        self.importance = importance

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self):
        self.exploration_base = 0.5 + np.random.rand() / 2

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = (self.portfolio_value / self.base_portfolio_value)
        return self.ratio_hold, self.ratio_portfolio_value

    def decide_action(self, pred_value, pred_policy, epsilon):
        pred = pred_policy
        if pred is None:
            pred = pred_value
            pred = new_softmax(pred)
        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1
        if pred_policy is not None:  # 매수, 매도, 관망의 차이가 작으면 탐험 수행 약 5%
            if np.max(pred_policy) - np.min(pred_policy) < 0.05:
                epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY
            else:
                action = np.random.randint(1, 2)
        else:
            exploration = False
            action = np.random.choice(self.NUM_ACTIONS, 1, p=pred)[0]#np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (
                    1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(
            min(
                int(confidence * (self.max_trading_unit - self.min_trading_unit)),
                self.max_trading_unit - self.min_trading_unit
            ), 0)
        return self.min_trading_unit + added_trading

    def advanced_decide_trading_unit(self, curr_price):
        return max(int((self.initial_balance * 0.01) / curr_price), 1)

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD # 사거나 팔 수 없다면 관망...

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 최대 매수 단위 변경
        self.advanced_change_trading_unit(curr_price)

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                    self.balance - curr_price * (1 + self.TRADING_CHARGE) \
                    * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                                curr_price * (1 + self.TRADING_CHARGE))),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) \
                            * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가
        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                    1 - (self.TRADING_TAX + self.TRADING_CHARGE)) \
                            * trading_unit
            if invest_amount > 0:
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        next_price = self.environment.get_next_price()
        if next_price is None:
            next_price = curr_price
        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + next_price * self.num_stocks
        self.profitloss = (
                (self.portfolio_value - self.initial_balance) / self.initial_balance
        )

        # 즉시 보상 - 수익률
        self.immediate_reward = self.profitloss
        delayed_reward = 0
        # 지연 보상 - 익절, 손절 기준
        self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        if self.base_profitloss > self.delayed_reward_threshold or self.base_profitloss < -self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0

        return self.immediate_reward, delayed_reward

    def hoga_act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD  # 사거나 팔 수 없다면 관망...

        # 환경에서 현재 가격 얻기
        mado_price = self.environment.get_mado_price()
        masu_price = self.environment.get_masu_price()

        # 최대 매수 단위 변경


        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            self.advanced_change_trading_unit(mado_price)
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                    self.balance - mado_price * (1 + self.TRADING_CHARGE) \
                    * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                                mado_price * (1 + self.TRADING_CHARGE))),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = mado_price * (1 + self.TRADING_CHARGE) \
                            * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가
        # 매도
        elif action == Agent.ACTION_SELL:
            self.advanced_change_trading_unit(masu_price)
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = masu_price * (
                    1 - (self.TRADING_TAX + self.TRADING_CHARGE)) \
                            * trading_unit
            if invest_amount > 0:
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        # next_price = self.environment.get_next_price()
        # if next_price is None:
        #     next_price = curr_price
        next_price = self.environment.get_masu_price()
        if next_price is None:
            next_price = masu_price
        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + next_price * self.num_stocks
        self.profitloss = (
                (self.portfolio_value - self.initial_balance) / self.initial_balance
        )
        # 즉시 보상 - 수익률
        self.immediate_reward = self.profitloss
        delayed_reward = 0
        # 지연 보상 - 익절, 손절 기준
        self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        if self.base_profitloss > self.delayed_reward_threshold or self.base_profitloss < -self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0

        return self.immediate_reward, delayed_reward

    def advanced_change_trading_unit(self, curr_price):
        self.max_trading_unit = max(int(self.portfolio_value / curr_price), 1)

    def advanced_act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD # 사거나 팔 수 없다면 관망...

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 최대 매수 단위 변경
        self.advanced_change_trading_unit(curr_price)

        # 즉시 보상 초기화
        self.immediate_reward = 0
        #매수 or 매도 단위 판단
        trading_unit = self.advanced_decide_trading_unit(confidence)
        # 매수
        if action == Agent.ACTION_BUY:
            # # 매수할 단위를 판단
            # balance = (
            #         self.balance - curr_price * (1 + self.TRADING_CHARGE) \
            #         * trading_unit
            # )
            # # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            # if balance < 0:
            #     trading_unit = max(
            #         min(
            #             int(self.balance / (
            #                     curr_price * (1 + self.TRADING_CHARGE))),
            #             self.max_trading_unit
            #         ),
            #         self.min_trading_unit
            #     )
            # # 수수료를 적용하여 총 매수 금액 산정
            # invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            # if invest_amount > 0:
            #     self.balance -= invest_amount  # 보유 현금을 갱신
            #     self.num_stocks += trading_unit  # 보유 주식 수를 갱신
            #     self.num_buy += 1  # 매수 횟수 증가
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount
            self.num_stocks += trading_unit
            self.num_buy += 1
        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            # trading_unit = min(trading_unit, self.num_stocks)
            # # 매도
            # invest_amount = curr_price * (
            #         1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            # if invest_amount > 0:
            #     self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
            #     self.balance += invest_amount  # 보유 현금을 갱신
            #     self.num_sell += 1  # 매도 횟수 증가
            invest_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE))* trading_unit
            self.balance += invest_amount
            self.num_stocks -= trading_unit
            self.num_sell += 1
        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        next_price = self.environment.get_next_price()
        if next_price is None:
            next_price = curr_price

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + next_price * self.num_stocks
        # self.profitloss = (
        #         (self.portfolio_value - self.initial_balance) / self.initial_balance
        #
        # 즉시 보상 - 수익률
        # self.immediate_reward = self.profitloss
        delayed_reward = 0
        # 지연 보상 - 익절, 손절 기준

        # 2천만이 되었다면 max, min이 바뀔 것

        self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        # updown_rate = (next_price - curr_price) / curr_price
        # stock_rate = self.num_stocks / self.max_trading_unit
        self.immediate_reward = self.base_profitloss# ((self.max_trading_unit - self.num_stocks) / self.max_trading_unit) = 1 - stock_rate
        #즉시 보상 = 등락률 *
        if self.base_profitloss > self.delayed_reward_threshold or self.base_profitloss < -self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0
        self.base_portfolio_value = self.portfolio_value
        s_prime = self.build_sample()
        if s_prime is None:
            done = True
        else:
            done = False

        return s_prime, done #self.immediate_reward, delayed_reward

    def set_environment(self, environment):
        self.environment = environment

################################################
    def step(self, action):
        curr_price = self.environment.get_price()
        next_price = self.environment.get_next_price()
        if next_price is None:
            next_price = curr_price
        up_rate = (next_price - curr_price) / curr_price
        s_prime = self.build_sample()
        if s_prime is None:
            done = True
        else:
            done = False
        if action == self.ACTION_BUY:
            r = up_rate * 1 * 100 # 보상이 기껏해야 0.3%일 수도 있으니 3% -> 3 0.3% -> 0.3
            self.num_buy += 1
        elif action == self.ACTION_SELL:
            r = up_rate * -1 * 100
            self.num_sell += 1
        elif action == self.ACTION_HOLD:
            r = up_rate * -0.1 * 100 # 올랐을 때 안샀으므로 -0.5 내렸을 때 안샀으므로 0.5
            self.num_hold += 1
        else:
            r = None
            print("error")
        return s_prime, r, done
################################################
    def build_sample(self):
        # 이미 observe에서 idx+1을 함 아마 둘이 같기 때문에 None을 반환하면 길이도 맞을 것임
        choose = self.environment.observe()
        if choose is not None:
            if len(self.environment.training_data) > self.environment.idx:
                sample = self.environment.training_data.iloc[self.environment.idx].tolist()
                #sample.extend(self.get_states()) agent의 STATEDIM = 0
                return sample
        return None
################################################
    def realstep(self, action, prob_a):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD # 사거나 팔 수 없다면 관망...

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 최대 매수 단위 변경
        self.advanced_change_trading_unit(curr_price)

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(prob_a)
            balance = (
                    self.balance - curr_price * (1 + self.TRADING_CHARGE) \
                    * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                                curr_price * (1 + self.TRADING_CHARGE))),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) \
                            * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가
        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(prob_a)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                    1 - (self.TRADING_TAX + self.TRADING_CHARGE)) \
                            * trading_unit
            if invest_amount > 0:
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가
        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        next_price = self.environment.get_next_price()
        if next_price is None:
            next_price = curr_price
        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + next_price * self.num_stocks
        self.profitloss = ((self.portfolio_value - self.initial_balance) / self.initial_balance)

        # 즉시 보상 - 수익률
        #self.immediate_reward = self.profitloss
        delayed_reward = 0
        # 지연 보상 - 익절, 손절 기준
        self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        # if self.base_profitloss > self.delayed_reward_threshold or self.base_profitloss < -self.delayed_reward_threshold:
        #     # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
        #     # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
        #     self.base_portfolio_value = self.portfolio_value
        #     delayed_reward = self.immediate_reward
        # else:
        #     delayed_reward = 0
        self.immediate_reward = self.base_profitloss
        self.base_portfolio_value = self.portfolio_value
        s_prime = self.build_sample()
        if s_prime is None:
            done = True
        else:
            done = False
        return action, s_prime, self.immediate_reward, done
################################################
    def noholdstep(self, action, prob_a):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD # 사거나 팔 수 없다면 관망...

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 최대 매수 단위 변경
        self.advanced_change_trading_unit(curr_price)

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(prob_a)
            balance = (
                    self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                                curr_price * (1 + self.TRADING_CHARGE))),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) \
                            * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가
        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(prob_a)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                    1 - (self.TRADING_TAX + self.TRADING_CHARGE)) \
                            * trading_unit
            if invest_amount > 0:
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가
        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        next_price = self.environment.get_next_price()
        if next_price is None:
            next_price = curr_price
        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + next_price * self.num_stocks
        self.profitloss = ((self.portfolio_value - self.initial_balance) / self.initial_balance)

        # 즉시 보상 - 수익률
        self.immediate_reward = self.profitloss
        # 지연 보상 - 익절, 손절 기준
        self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        if self.base_profitloss > self.delayed_reward_threshold or self.base_profitloss < -self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        s_prime = self.build_sample()
        if s_prime is None:
            done = True
        else:
            done = False
        reward = (self.importance * self.immediate_reward) + ((1-self.importance) * self.base_profitloss)
        return s_prime, reward, done
################################################
    def onlypvstep(self, action, prob_a):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD # 사거나 팔 수 없다면 관망...

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 최대 매수 단위 변경
        self.advanced_change_trading_unit(curr_price)

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY: # 초기 자본금 기준으로 1%만 매수, 매도 관망
            # 매수할 단위를 판단
            trading_unit = self.advanced_decide_trading_unit(curr_price)
            balance = (
                    self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                                curr_price * (1 + self.TRADING_CHARGE))),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) \
                            * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가
        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.advanced_decide_trading_unit(curr_price)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                    1 - (self.TRADING_TAX + self.TRADING_CHARGE)) \
                            * trading_unit
            if invest_amount > 0:
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가
        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

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
        # 즉시 보상 - 수익률
        self.immediate_reward = self.profitloss
        # 지연 보상 - 익절, 손절 기준
        self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        if self.base_profitloss < -self.delayed_reward_threshold or self.base_profitloss > self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            # done = True # 기준치 손익률 달성 시 종료
        reward = self.immediate_reward - uprate
        return s_prime, reward, done
################################################
    def newstep(self, action, prob_a=None):
        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 최대 매수 단위 변경
        self.advanced_change_trading_unit(curr_price)

        max_trading_unit = max(int(self.portfolio_value / (curr_price * (1 + self.TRADING_CHARGE + self.SIMUL_CHARGE))),1)
        maintain_stocks = int((action / self.NUM_ACTIONS) * (max_trading_unit))

        trading_unit = self.num_stocks - maintain_stocks

        if trading_unit > 0: # trading_unit만큼 매도
            sell_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE + self.SIMUL_CHARGE)) * trading_unit
            self.balance += sell_amount
            self.num_stocks -= trading_unit
            self.num_sell += 1
        elif trading_unit < 0: #trading_unit만큼 매수
            trading_unit = -trading_unit
            invest_amount = curr_price * (1 + self.TRADING_CHARGE + self.SIMUL_CHARGE) * trading_unit
            self.balance -= invest_amount
            self.num_stocks += trading_unit
            self.num_buy += 1
        else:
            self.num_hold += 1

        # 즉시 보상 초기화
        self.immediate_reward = 0
        #
        # # 매수
        # if action == Agent.ACTION_BUY: # 초기 자본금 기준으로 1%만 매수, 매도 관망
        #     # 매수할 단위를 판단
        #     trading_unit = self.advanced_decide_trading_unit(curr_price)
        #     balance = (
        #             self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
        #     )
        #     # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
        #     if balance < 0:
        #         trading_unit = max(
        #             min(
        #                 int(self.balance / (
        #                         curr_price * (1 + self.TRADING_CHARGE))),
        #                 self.max_trading_unit
        #             ),
        #             self.min_trading_unit
        #         )
        #     # 수수료를 적용하여 총 매수 금액 산정
        #     invest_amount = curr_price * (1 + self.TRADING_CHARGE) \
        #                     * trading_unit
        #     if invest_amount > 0:
        #         self.balance -= invest_amount  # 보유 현금을 갱신
        #         self.num_stocks += trading_unit  # 보유 주식 수를 갱신
        #         self.num_buy += 1  # 매수 횟수 증가
        # # 매도
        # elif action == Agent.ACTION_SELL:
        #     # 매도할 단위를 판단
        #     trading_unit = self.advanced_decide_trading_unit(curr_price)
        #     # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
        #     trading_unit = min(trading_unit, self.num_stocks)
        #     # 매도
        #     invest_amount = curr_price * (
        #             1 - (self.TRADING_TAX + self.TRADING_CHARGE)) \
        #                     * trading_unit
        #     if invest_amount > 0:
        #         self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
        #         self.balance += invest_amount  # 보유 현금을 갱신
        #         self.num_sell += 1  # 매도 횟수 증가
        # # 홀딩
        # elif action == Agent.ACTION_HOLD:
        #     self.num_hold += 1  # 홀딩 횟수 증가
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
        self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        # 즉시 보상 - 수익률
        #self.immediate_reward = self.profitloss
        # 지연 보상 - 익절, 손절 기준
        #self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)
        # if self.base_profitloss < -0.03 or self.base_profitloss > 0.03:
        # #     # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
        # #     # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
        #    self.base_portfolio_value = self.portfolio_value
            #done = True # 기준치 손익률 달성 시 종료
        # 보상을 최대화하는 것이 목적
        # 그러나 주식을 사지않고 계속 매도만 하는 경우가 발생 가능(예를 들면 계속 하락장인 경우, 이득을 볼 수 없기 때문에)
        # 그런데 그 상황에 최적화되고(즉, 하락하는 것에서 관망만 하는 것이 이득이라고 생각하게 됨)
        # 빠져나오지 못할 수 있기 때문에, 보상의 선정을 다르게 함
        # 보상 = 얻을 수 있는 최대의 이익과 현재의 이익이 가까울 수록 높음
        # 위험한 투자자의 손익률 : 등락률(next - current / current) * 최대 보유 수 = 정확히는 모든 돈이 주식으로 넘어간 경우
        # 보상 = 손익률 - 위투손 (보상은 거의 -로 나올 것) 근데 이거 (현재 보유 수 - 최대 보유 수) * 등락률 아님?
        # 결국 (float(action/10) - 1.0) * uprate
        #self.immediate_reward = float((action - 4.5)/10) * uprate * 100 * 0.1 + self.profitloss * 100 * 0.9
        # 엄청난 손해 후에 이득을 보는 행동을 하여도 변함없이 손실되었다는 이유만으로 점수를 낮게 준다면 발전할 수 없을 것이라 판단함
        #self.immediate_reward = self.base_profitloss
        self.immediate_reward = (self.profitloss + uprate * (float(action / self.NUM_ACTIONS) - 1)) * 100
        #profitoss대신 base로 계산
        #self.base_portfolio_value = self.portfolio_value
        #self.immediate_reward = self.base_profitloss# + self.profitloss * 10
        reward = self.immediate_reward
        return s_prime, reward, done