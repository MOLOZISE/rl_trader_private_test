class Environment:
    PRICE_IDX = 4
    ####
    STEP = 1
    ####

    def __init__(self, chart_data=None, training_data=None):
        self.chart_data = chart_data
        self.training_data = training_data
        self.train_observation = None
        self.chart_observation = None
        self.idx = -1

    def reset(self):
        self.train_observation = None
        self.chart_observation = None
        self.idx = -1

    def observe(self):
        if len(self.training_data) > self.idx + self.STEP:
            self.idx += self.STEP
            self.train_observation = self.training_data.iloc[self.idx]
            self.chart_observation = self.chart_data.iloc[self.idx]
            return self.train_observation
        return None

    def get_price(self):
        if self.chart_observation is not None:
            return self.chart_observation[self.PRICE_IDX]
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

    def step(self, action):
        next_p = self.get_next_price()
        p = self.get_price()
        self.train_observation = self.observe()
        if(self.train_observation is None):
            return None
        if(action==0): # buy
            return float(((next_p - p) / p) * 100 * 1)
        elif(action==1): # sell
            return float(((next_p - p) / p) * 100 * -1)
        else:
            print("error")
            return None

    def up_step(self, action):
        next_p = self.get_next_price()
        p = self.get_price()
        self.train_observation = self.observe()
        if (self.train_observation is None):
            return None
        if (action == 0):  # buy
            if (float(((next_p - p) / p) * 100 * 1) >= 0):
                return 1
            else:
                return -1
        elif (action == 1):  # sell
            if (float(((next_p - p) / p) * 100 * 1) >= 0):
                return -1
            else:
                return 1
        else:
            print("error")
            return None

    def down_step(self, action):
        next_p = self.get_next_price()
        p = self.get_price()
        self.train_observation = self.observe()
        if (self.train_observation is None):
            return None
        if (action == 0):  # buy
            if (float(((next_p - p) / p) * 100 * 1) >= 0):
                return 1
            else:
                return -1
        elif (action == 1):  # sell
            if (float(((next_p - p) / p) * 100 * 1) >= 0):
                return -1
            else:
                return 1
        else:
            print("error")
            return None

    def get_train_ob(self):
        return self.train_observation