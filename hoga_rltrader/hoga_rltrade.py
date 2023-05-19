import os
import sys
import logging
import argparse
import json

import settings
import utils
import data_manager
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code', nargs='+')
    parser.add_argument('--ver', choices=['al', 'v1', 'v2', 'ossp_all', 'test_h'], default='a1')
    parser.add_argument('--rl_method',
                        choices=['dqn', 'pg', 'ac', 'a2c', 'a3c'], default='a3c')
    parser.add_argument('--net',
                        choices=['dnn', 'lstm', 'cnn'], default='lstm')
    parser.add_argument('--num_steps', type=int, default=5) # dnn제외 lstm, cnn을 사용할 경우에만 사용
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--start_epsilon', type=float, default=1)
    parser.add_argument('--balance', type=int, default=10000000)
    parser.add_argument('--num_epoches', type=int, default=100)
    parser.add_argument('--delayed_reward_threshold',
                        type=float, default=0.1) #1분봉에서는 다 거기서 거기이기 때문에 한 번의 거래에 지연 보상 발생 X 아예 거래세, 수수료 0.00265보다 조금만 높아도 지연 보상으로 지칭
    parser.add_argument('--backend',
                        choices=['tensorflow', 'plaidml'], default='tensorflow')
    parser.add_argument('--output_name', default=utils.get_time_str())
    parser.add_argument('--value_network_name')
    parser.add_argument('--policy_network_name')
    parser.add_argument('--reuse_models', action='store_true')
    parser.add_argument('--learning', action='store_true')
    ####################################################### 20210515
    parser.add_argument('--application_private', action='store_true')
    ####################################################### 20210515
    # 분봉의 경우
    #parser.add_argument('--date', default='20200626') # 투자를 할 일자를 선택 09:10 ~ 14:59 (15:19까지가 정상거래 -> 15:20 ~ 30분은 동시호가 진행이기에 무의미 -> 넉넉히 14:59까지로
    parser.add_argument('--start_date', default='20200626')
    parser.add_argument('--end_date', default='20200626')

    args = parser.parse_args()

    # Keras Backend 설정
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 output폴더에 output_name_강화학습기법_신경망구조의 이름 폴더 생성
    output_path = os.path.join(settings.BASE_DIR, 'output/{}_{}_{}'.format(args.output_name, args.rl_method, args.net))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    # 로그 기록 설정 위의 출력 경로에 log파일 생성
    file_handler = logging.FileHandler(filename=os.path.join(output_path, "{}.log".format(args.output_name)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s", handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 로그, Keras Backend 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from agent import Agent
    from hoga_learners import DQNLearner, PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''
    ######################### 삭제가능
    if args.value_network_name is not None:
        value_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(args.value_network_name))
    else:
        value_network_path = os.path.join(output_path, '{}_{}_value_{}.h5'.format(args.rl_method, args.net, args.output_name))
    if args.policy_network_name is not None:
        policy_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(args.policy_network_name))
    else:
        policy_network_path = os.path.join(output_path, '{}_{}_policy_{}.h5'.format(args.rl_method, args.net, args.output_name))

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []
    # 분봉의 경우
    # start_time = args.date + "090000"  # 9시 10분부터
    # end_time = args.date + "150000"  # 15시 까지
    start_date = args.start_date
    end_date = args.end_date
    # 1. start_time ~ end_time까지 20210323 ~ 20210515
    # 2. 20210323폴더의 20210323.txt 1줄 읽기
    # 3. 1줄의 첫번째 0900 -> 0900_{089150}.txt ~ 0900_{003490}.txt -> chart_data, training_data (반복)
    # 4. 0900 ~ 1500까지 10개 항목 반복 학습(하나의 신경망을 사용하도록)
    d1 = datetime.date(int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8]))
    d2 = datetime.date(int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:8]))
    delta = d2 - d1
    # code_list에는 대상 종목 코드들이 들어가 있음
    # 하나씩 하기 위해서 0. beforedata에 원하는 종목 코드들(or 1개)를 적어둠 1. count = 만들 모델 번호 2. before_code 전 모델의 종목코드
    #  3. before_model_name 전 모델의 이름 4. tempcount 2이상
    # 각 종목코드별로
    for i in range(delta.days + 1):
        target_date = str(d1 + datetime.timedelta(days=i)).replace('-', '')
        folderpath = "files/" + target_date
        if not os.path.isdir(folderpath):
            continue
        filepath = "files/" + target_date + "/" + target_date + ".txt"
        f = open(filepath, "r", encoding="utf8")
        lines = f.readlines()
        for line in lines: # line 0900 종목 코드 10개
            tabline = line.split('\t')
            tempdate = tabline[0]
            print(tempdate)
            codes = []
            for k in range(10):
                codes.append(tabline[k+1]) # 1 ~ 10
            for stock_code in codes:
                filepath_date = "files/" + target_date + "/" + tempdate + "_" + stock_code + ".txt"
                print(filepath_date)
                output_name = "test01"
                if (args.reuse_models):
                    value_network_path = os.path.join(settings.BASE_DIR,
                                                      'output/20210515183303_dqn_cnn_test/{}_{}_value_{}.h5'.format(
                                                          args.rl_method, args.net, output_name))
                    policy_network_path = os.path.join(settings.BASE_DIR,
                                                       'output/20210515183303_dqn_cnn_test/{}_{}_policy_{}.h5'.format(
                                                           args.rl_method, args.net, output_name))
                else:
                    value_network_path = os.path.join(output_path,
                                                      '{}_{}_value_{}.h5'.format(args.rl_method, args.net,
                                                                                 output_name))
                    policy_network_path = os.path.join(output_path,
                                                       '{}_{}_policy_{}.h5'.format(args.rl_method, args.net,
                                                                                 output_name))
                chart_data, training_data = data_manager.load_data(
                    os.path.join(settings.BASE_DIR, filepath_date),
                    ver=args.ver, start_time=None, end_time=None)

                min_trading_unit = 1  # max(int((args.balance)/100 / chart_data.iloc[-1]['close']), 1)

                if chart_data.iloc[0]['mado_price'] == 0:
                    max_trading_unit = max(int(args.balance / chart_data.iloc[0]['masu_price']), 1)
                else:
                    max_trading_unit = max(int(args.balance / chart_data.iloc[0]['mado_price']), 1)
                common_params = {'rl_method': args.rl_method,
                                 'delayed_reward_threshold': args.delayed_reward_threshold,
                                 'net': args.net, 'num_steps': args.num_steps, 'lr': args.lr,
                                 'output_path': output_path, 'reuse_models': args.reuse_models}

                # 강화학습 시작
                learner = None
                if args.rl_method != 'a3c':
                    common_params.update({'stock_code': stock_code,
                                          'chart_data': chart_data,
                                          'training_data': training_data,
                                          'min_trading_unit': min_trading_unit,
                                          'max_trading_unit': max_trading_unit})
                    if args.rl_method == 'dqn':
                        learner = DQNLearner(**{**common_params,
                                                'value_network_path': value_network_path})
                    elif args.rl_method == 'pg':
                        learner = PolicyGradientLearner(**{**common_params,
                                                           'policy_network_path': policy_network_path})
                    elif args.rl_method == 'ac':
                        learner = ActorCriticLearner(**{**common_params,
                                                        'value_network_path': value_network_path,
                                                        'policy_network_path': policy_network_path})
                    elif args.rl_method == 'a2c':
                        learner = A2CLearner(**{**common_params,
                                                'value_network_path': value_network_path,
                                                'policy_network_path': policy_network_path})
                    if learner is not None:
                        learner.run(balance=args.balance,
                                    num_epoches=args.num_epoches,
                                    discount_factor=args.discount_factor,
                                    start_epsilon=args.start_epsilon,
                                    learning=args.learning)
                        learner.save_models()
                else:
                    list_stock_code.append(stock_code)
                    list_chart_data.append(chart_data)
                    list_training_data.append(training_data)
                    list_min_trading_unit.append(min_trading_unit)
                    list_max_trading_unit.append(max_trading_unit)

            if args.rl_method == 'a3c':
                learner = A3CLearner(**{
                    **common_params,
                    'list_stock_code': list_stock_code,
                    'list_chart_data': list_chart_data,
                    'list_training_data': list_training_data,
                    'list_min_trading_unit': list_min_trading_unit,
                    'list_max_trading_unit': list_max_trading_unit,
                    'value_network_path': value_network_path,
                    'policy_network_path': policy_network_path})

                learner.run(balance=args.balance, num_epoches=args.num_epoches,
                            discount_factor=args.discount_factor,
                            start_epsilon=args.start_epsilon,
                            learning=args.learning)
                learner.save_models()