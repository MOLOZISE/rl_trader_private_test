class Hoga_Environment:
    # 원래 f1에서의 PRICE_IDX = 4  # 종가의 위치
    ####
    # c1, u1에서의 종가의 위치
    MADO_IDX = 0
    MASU_IDX = 1
    ####
    STEP = 20 # 0초에 예측 시작 시 10초 후 결과 반영, 20초 후에 결과를 확인하여야 하므로
    ####

    def __init__(self, chart_data=None, training_data=None):
        self.chart_data = chart_data
        self.training_data = training_data
        self.observation = None
        self.idx = -1

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        if len(self.chart_data) > self.idx + self.STEP:
            self.idx += self.STEP
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    def get_mado_price(self):
        if self.observation is not None:
            if self.observation[self.MADO_IDX] == 0:
                return self.observation[self.MASU_IDX]
            else:
                return self.observation[self.MADO_IDX]
        return None

    def get_masu_price(self):
        if self.observation is not None:
            if self.observation[self.MASU_IDX] == 0:
                return self.observation[self.MADO_IDX]
            else:
                return self.observation[self.MASU_IDX]
        return None

    def get_next_price(self):
        if len(self.chart_data) > (self.idx + self.STEP):
            return self.chart_data.iloc[self.idx + self.STEP][self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data

    def set_training_data(self, training_data):
        self.training_data = training_data

    def get_training_data_shape(self):
        return self.training_data.shape[1]