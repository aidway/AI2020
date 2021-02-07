%matplotlib inline
%config ZMQInteractiveShell.ast_node_interactivity='all'
%pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

pd.set_option('display.max_columns', None)
font_set = FontProperties(fname='/home/dm1/simhei.ttf')

from sklearn.preprocessing import MinMaxScaler

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam

from impala.dbapi import connect
from impala.util import as_pandas

import pyodbc



# 获取 交易量 训练数据
def data_scaler(df_train, df_predict):
    scaler = MinMaxScaler()
    df_train_scaler = df_train
    #df_train_scaler = df_train_scaler.drop(['classid'], axis=1)
    df_train_scaler = df_train_scaler.dropna(axis=0, how='any')
    df_train_scaler = df_train_scaler.astype('float64')
    
    df_train_scaler = scaler.fit_transform(df_train_scaler)
    df_train_scaler = np.array(df_train_scaler)
    
    
    df_predict_scaler = df_predict
    df_predict_scaler = df_predict_scaler.fillna(0)
    df_predict_scaler = df_predict_scaler.astype('float64')
    df_predict_scaler = scaler.transform(df_predict_scaler)
    df_predict_scaler = np.array(df_predict_scaler)
    
    print('df_train shape:', df_train_scaler.shape)
    print('df_predict shape:', df_predict_scaler.shape)
    return df_train_scaler, df_predict_scaler, scaler


def next_window(df, i, enc_len, dec_len):
    enc_data = df[i: i+enc_len]
    dec_data = df[i+enc_len : i+enc_len+dec_len, 0]
    return enc_data, dec_data


# 获取 encoder-decoder 数据
def get_enc_dec_data(dec_len, df_scaler_data):
    train_x = []
    train_y = []

    for i in range(df_scaler_data.shape[0] - dec_len):
        enc_data, dec_data = next_window(df_scaler_data, i, enc_len, dec_len)
        train_x.append(enc_data)
        train_y.append(dec_data)
    
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_y = train_y.reshape(train_y.shape + (1,))


    if debug_mode:
        print('train_x shape:', train_x.shape)
        print('train_y shape:', train_y.shape)
        
    return train_x, train_y



# 训练 seq2seq 模型
def train_seq2seq(train_x, train_y, n_input):
    ## define traing encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder_lstm = LSTM(n_units, return_sequences=True,return_state=True, dropout=dropout)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
 
    encoder_lstm_2 = LSTM(n_units, return_sequences=True,return_state=True,dropout=dropout)
    encoder_outputs_2, state_h_2, state_c_2 = encoder_lstm_2(encoder_outputs)
    encoder_states = [state_h_2, state_c_2]

#     encoder_lstm_3 = LSTM(n_units, return_state=True,dropout=dropout)
#     encoder_outputs_3, state_h_3, state_c_3 = encoder_lstm_3(encoder_outputs)
#     encoder_states = [state_h_3, state_c_3]


    ## define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout,activation='relu') 
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)



    decoder_dense = Dense(n_output)
    decoder_outputs = decoder_dense(decoder_outputs)

    ## define training model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#     model.summary()


    enc_input_data = train_x
    dec_target_data = train_y

    ###################################################################################################################################
    dec_input_data = np.zeros(dec_target_data.shape)
    dec_input_data[:, 1:, :] = dec_target_data[:, :-1, :]
    dec_input_data[:, 0, 0] = enc_input_data[:, -1, 0]

#     enc_input_data.shape
#     dec_input_data.shape
#     dec_target_data.shape

    ## compile model mean_squared_error
    model.compile(Adam(), loss='mean_absolute_error')
    history = model.fit([enc_input_data, dec_input_data],
                   dec_target_data,
                   batch_size = batch_size,
                   epochs = epochs,
                   validation_split = 0.2,
                   verbose=0)

    ## plot metrics
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    if debug_mode:
        plt.figure(figsize=(20,8))
        plt.plot(range(len(loss)), loss, label='train loss')
        plt.plot(range(len(val_loss)), val_loss, label='val loss')
        plt.legend()
    return decoder_lstm, decoder_inputs, decoder_dense,encoder_inputs, encoder_states



## 获取推断模型
def get_inference_model(encoder_inputs, encoder_states,decoder_lstm, decoder_inputs,decoder_dense):
    ## define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)


    ## define inference docoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    return encoder_model, decoder_model

    
def predict_once(enc_input_data, encoder_model, decoder_model,dec_len ):
    '''
    预测一个时间点的后5天
    '''
#     input_seq = enc_input_data[pred_idx:pred_idx+1, :, :]
    input_seq = enc_input_data
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1,1,1))
    target_seq[0,0,0] = input_seq[0, -1, 0]
    
    decoded_seq = np.zeros((1, dec_len, 1))

    for i in range(dec_len):
        output, h, c = decoder_model.predict([target_seq] + states_value)
        decoded_seq[0, i, 0] = output[0, 0, 0]
        
        states_value = [h, c]
        #target_seq = np.zeros((1,1,1))
        target_seq[0,0,0] = output[0,0,0]

    return decoded_seq.reshape(-1,1)



def predict(df_data,n_input ):
    '''
    将df_data作为训练集，df_data[-1]作为预测的输入，输出此时间点后5天的预测结果
    '''
    pred_value = np.zeros((5,5))

    # 枚举decoder的长度，从1到5，然后加权得到最终的预测结果
    for j in range(0,5):
        dec_len = j + 1
        df_train, df_predict, scaler = data_scaler(df_data, df_data[-1:])

        train_x, train_y = get_enc_dec_data(dec_len, df_train)

        decoder_lstm, decoder_inputs,decoder_dense,encoder_inputs, encoder_states = train_seq2seq(train_x, train_y, n_input)

        encoder_model, decoder_model = get_inference_model(encoder_inputs, encoder_states,decoder_lstm, decoder_inputs,decoder_dense)

        pred_y = predict_once(df_predict.reshape((1,)+df_predict.shape), encoder_model, decoder_model, dec_len)
        pred_y_init = np.round(scaler.inverse_transform(np.zeros((pred_y.shape[0],n_input))+pred_y)[:,0])
        
         
        pred_value[j, 0:len(pred_y_init)] = pred_y_init
    
    
    print(pred_value)
    
    pred_y = np.sum(pred_value * pred_value_weight, axis=0)

    return pred_y
    
    
def score(pred_y, true_y):
    score_weight = [0.5,0.15,0.15,0.1,0.1]
    return 1-np.sum(np.abs(pred_y - true_y)/true_y * score_weight)


def get_df_data(trade_date):
    con = connect(**INCEPTOR_CONFIG)
    cur= con.cursor()
    sql = "select classid,\
       commid,\
       trade_date,\
       vol_qty,\
       hold_qty,\
       one_cur_day_day_max_price_change,\
       pre_one_day_day_max_price_change,\
       pre_two_day_day_max_price_change,\
       pre_three_day_day_max_price_change,\
       pre_four_day_day_max_price_change,\
       pre_five_day_day_max_price_change,\
       /*pre_six_day_day_max_price_change,\
       pre_seven_day_day_max_price_change,\
       pre_eigth_day_day_max_price_change,\
       pre_nine_day_day_max_price_change,*/\
       one_cur_day_remain_last_trade_date_day,\
       pre_one_day_remain_last_trade_date_day,\
       pre_two_day_remain_last_trade_date_day,\
       pre_three_day_remain_last_trade_date_day,\
       pre_four_day_remain_last_trade_date_day,\
       pre_five_day_remain_last_trade_date_day,\
       pre_six_day_remain_last_trade_date_day,\
       pre_seven_day_remain_last_trade_date_day,\
       pre_eigth_day_remain_last_trade_date_day,\
       pre_nine_day_remain_last_trade_date_day,\
       one_cur_day_openprice_change_ratio,\
       one_cur_day_closeprice_change_ratio,\
       pre_one_day_openprice_change_ratio,\
       pre_two_day_openprice_change_ratio,\
       pre_three_day_openprice_change_ratio,\
       pre_four_day_openprice_change_ratio,\
       pre_five_day_openprice_change_ratio,\
       pre_six_day_openprice_change_ratio,\
       pre_seven_day_openprice_change_ratio,\
       pre_eigth_day_openprice_change_ratio,\
       pre_nine_day_openprice_change_ratio,\
       pre_one_day_closeprice_change_ratio,\
       pre_two_day_closeprice_change_ratio,\
       pre_three_day_closeprice_change_ratio,\
       pre_four_day_closeprice_change_ratio,\
       pre_five_day_closeprice_change_ratio,\
       pre_six_day_closeprice_change_ratio,\
       pre_seven_day_closeprice_change_ratio,\
       pre_eigth_day_closeprice_change_ratio,\
       pre_nine_day_closeprice_change_ratio,\
       one_cur_day_hold_to_vol_ratio,\
       pre_one_day_hold_to_vol_ratio,\
       pre_two_day_hold_to_vol_ratio,\
       pre_three_day_hold_to_vol_ratio,\
       pre_four_day_hold_to_vol_ratio,\
       pre_five_day_hold_to_vol_ratio,\
       pre_six_day_hold_to_vol_ratio,\
       pre_seven_day_hold_to_vol_ratio,\
       pre_eigth_day_hold_to_vol_ratio,\
       pre_nine_day_hold_to_vol_ratio,\
       one_cur_day_hold_change_ratio,\
       pre_one_day_hold_change_ratio,\
       pre_two_day_hold_change_ratio,\
       pre_three_day_hold_change_ratio,\
       pre_four_day_hold_change_ratio,\
       pre_five_day_hold_change_ratio,\
       pre_six_day_hold_change_ratio,\
       pre_seven_day_hold_change_ratio,\
       pre_eigth_day_hold_change_ratio,\
       pre_nine_day_hold_change_ratio,\
       one_cur_day_vol_change_ratio,\
       pre_one_day_vol_change_ratio,\
       pre_two_day_vol_change_ratio,\
       pre_three_day_vol_change_ratio,\
       pre_four_day_vol_change_ratio,\
       pre_five_day_vol_change_ratio,\
       pre_six_day_vol_change_ratio,\
       pre_seven_day_vol_change_ratio,\
       pre_eigth_day_vol_change_ratio,\
       pre_nine_day_vol_change_ratio,\
       one_cur_day_margin_rate_change_ratio,\
       pre_one_day_margin_rate_change_ratio,\
       pre_two_day_margin_rate_change_ratio,\
       pre_three_day_margin_rate_change_ratio,\
       pre_four_day_margin_rate_change_ratio,\
       pre_five_day_margin_rate_change_ratio,\
       pre_six_day_margin_rate_change_ratio,\
       pre_seven_day_margin_rate_change_ratio,\
       pre_eigth_day_margin_rate_change_ratio,\
       pre_nine_day_margin_rate_change_ratio,\
       one_cur_day_clearprice_change_ratio,\
       pre_one_day_clearprice_change_ratio,\
       pre_two_day_clearprice_change_ratio,\
       pre_three_day_clearprice_change_ratio,\
       pre_four_day_clearprice_change_ratio,\
       pre_five_day_clearprice_change_ratio,\
       pre_six_day_clearprice_change_ratio,\
       pre_seven_day_clearprice_change_ratio,\
       pre_eigth_day_clearprice_change_ratio,\
       pre_nine_day_clearprice_change_ratio,\
       one_cur_day_buy_to_open_liq_ratio,\
       pre_one_day_buy_to_open_liq_ratio,\
       pre_two_day_buy_to_open_liq_ratio,\
       pre_three_day_buy_to_open_liq_ratio,\
       pre_four_day_buy_to_open_liq_ratio,\
       pre_five_day_buy_to_open_liq_ratio,\
       pre_six_day_buy_to_open_liq_ratio,\
       pre_seven_day_buy_to_open_liq_ratio,\
       pre_eigth_day_buy_to_open_liq_ratio,\
       pre_nine_day_buy_to_open_liq_ratio,\
       one_cur_day_sell_to_open_liq_ratio,\
       pre_one_day_sell_to_open_liq_ratio,\
       pre_two_day_sell_to_open_liq_ratio,\
       pre_three_day_sell_to_open_liq_ratio,\
       pre_four_day_sell_to_open_liq_ratio,\
       pre_five_day_sell_to_open_liq_ratio,\
       pre_six_day_sell_to_open_liq_ratio,\
       pre_seven_day_sell_to_open_liq_ratio,\
       pre_eigth_day_sell_to_open_liq_ratio,\
       pre_nine_day_sell_to_open_liq_ratio,\
       one_cur_day_legal_buy_hold_ratio,\
       pre_one_day_legal_buy_hold_ratio,\
       pre_two_day_legal_buy_hold_ratio,\
       pre_three_day_legal_buy_hold_ratio,\
       pre_four_day_legal_buy_hold_ratio,\
       pre_five_day_legal_buy_hold_ratio,\
       pre_six_day_legal_buy_hold_ratio,\
       pre_seven_day_legal_buy_hold_ratio,\
       pre_eigth_day_legal_buy_hold_ratio,\
       pre_nine_day_legal_buy_hold_ratio,\
       one_cur_day_legal_sell_hold_ratio,\
       pre_one_day_legal_sell_hold_ratio,\
       pre_two_day_legal_sell_hold_ratio,\
       pre_three_day_legal_sell_hold_ratio,\
       pre_four_day_legal_sell_hold_ratio,\
       pre_five_day_legal_sell_hold_ratio,\
       pre_six_day_legal_sell_hold_ratio,\
       pre_seven_day_legal_sell_hold_ratio,\
       pre_eigth_day_legal_sell_hold_ratio,\
       pre_nine_day_legal_sell_hold_ratio,\
       one_cur_day_inter_contract_change_ratio,\
       pre_one_day_inter_contract_change_ratio,\
       pre_two_day_inter_contract_change_ratio,\
       pre_three_day_inter_contract_change_ratio,\
       pre_four_day_inter_contract_change_ratio,\
       pre_five_day_inter_contract_change_ratio,\
       pre_six_day_inter_contract_change_ratio,\
       pre_seven_day_inter_contract_change_ratio,\
       pre_eigth_day_inter_contract_change_ratio,\
       pre_nine_day_inter_contract_change_ratio,\
       pre_cur_hold_to_d0_hold_max,\
       pre_pre_one_hold_to_d0_hold_max,\
       pre_pre_two_hold_to_d0_hold_max,\
       pre_pre_three_hold_to_d0_hold_max,\
       pre_pre_four_hold_to_d0_hold_max,\
       pre_pre_five_hold_to_d0_hold_max,\
       pre_pre_six_hold_to_d0_hold_max,\
       pre_pre_seven_hold_to_d0_hold_max,\
       pre_pre_eight_hold_to_d0_hold_max,\
       pre_pre_nine_hold_to_d0_hold_max,\
       pre_cur_hold_to_d0_hold_min,\
       pre_pre_one_hold_to_d0_hold_min,\
       pre_pre_two_hold_to_d0_hold_min,\
       pre_pre_three_hold_to_d0_hold_min,\
       pre_pre_four_hold_to_d0_hold_min,\
       pre_pre_five_hold_to_d0_hold_min,\
       pre_pre_six_hold_to_d0_hold_min,\
       pre_pre_seven_hold_to_d0_hold_min,\
       pre_pre_eight_hold_to_d0_hold_min,\
       pre_pre_nine_hold_to_d0_hold_min,\
       pre_cur_vol_to_d0_vol_max,\
       pre_cur_vol_to_d0_vol_min,\
       pre_pre_one_vol_to_d0_vol_max,\
       pre_pre_one_vol_to_d0_vol_min,\
       pre_pre_two_vol_to_d0_vol_max,\
       pre_pre_two_vol_to_d0_vol_min,\
       pre_pre_three_vol_to_d0_vol_max,\
       pre_pre_three_vol_to_d0_vol_min,\
       pre_pre_four_vol_to_d0_vol_max,\
       pre_pre_four_vol_to_d0_vol_min,\
       pre_pre_five_vol_to_d0_vol_max,\
       pre_pre_five_vol_to_d0_vol_min,\
       pre_pre_six_vol_to_d0_vol_max,\
       pre_pre_six_vol_to_d0_vol_min,\
       pre_pre_seven_vol_to_d0_vol_max,\
       pre_pre_seven_vol_to_d0_vol_min,\
       pre_pre_eight_vol_to_d0_vol_max,\
       pre_pre_eight_vol_to_d0_vol_min,\
       pre_pre_nine_vol_to_d0_vol_max,\
       pre_pre_nine_vol_to_d0_vol_min,\
       one_cur_day_main_to_second_vol,\
       pre_one_day_main_to_second_vol,\
       pre_two_day_main_to_second_vol,\
       pre_three_day_main_to_second_vol,\
       pre_four_day_main_to_second_vol,\
       pre_five_day_main_to_second_vol,\
       pre_six_day_main_to_second_vol,\
       pre_seven_day_main_to_second_vol,\
       pre_eigth_day_main_to_second_vol,\
       pre_nine_day_main_to_second_vol,\
       pre_rednt_ten_day_main_to_second_vol,\
       pre_rednt_eleven_day_main_to_second_vol,\
       pre_rednt_twelve_day_main_to_second_vol,\
       pre_rednt_thriteen_day_main_to_second_vol,\
       pre_rednt_fourteen_day_main_to_second_vol,\
       one_cur_day_main_to_second_hold,\
       pre_one_day_main_to_second_hold,\
       pre_two_day_main_to_second_hold,\
       pre_three_day_main_to_second_hold,\
       pre_four_day_main_to_second_hold,\
       pre_five_day_main_to_second_hold,\
       pre_six_day_main_to_second_hold,\
       pre_seven_day_main_to_second_hold,\
       pre_eigth_day_main_to_second_hold,\
       pre_nine_day_main_to_second_hold,\
       pre_rednt_ten_day_main_to_second_hold,\
       pre_rednt_eleven_day_main_to_second_hold,\
       pre_rednt_twelve_day_main_to_second_hold,\
       pre_rednt_thriteen_day_main_to_second_hold,\
       pre_rednt_fourteen_day_main_to_second_hold,\
       one_cur_day_trade_today_change_ratio,\
       pre_one_day_trade_today_change_ratio,\
       pre_two_day_trade_today_change_ratio,\
       pre_three_day_trade_today_change_ratio,\
       pre_four_day_trade_today_change_ratio,\
       pre_five_day_trade_today_change_ratio,\
       pre_six_day_trade_today_change_ratio,\
       pre_seven_day_trade_today_change_ratio,\
       pre_eigth_day_trade_today_change_ratio,\
       pre_nine_day_trade_today_change_ratio,\
       pre_mean_five_vol,\
       pre_mean_five_hold,\
       cur_sum_vol,\
       cur_curttlopen,\
       one_cur_day_trade_rate_change_ratio,\
       pre_one_day_trade_rate_change_ratio,\
       pre_two_day_trade_rate_change_ratio,\
       pre_three_day_trade_rate_change_ratio,\
       pre_four_day_trade_rate_change_ratio,\
       pre_five_day_trade_rate_change_ratio,\
       pre_six_day_trade_rate_change_ratio,\
       pre_seven_day_trade_rate_change_ratio,\
       pre_eigth_day_trade_rate_change_ratio,\
       pre_nine_day_trade_rate_change_ratio,\
       type_0_hold,\
       type_0_vol,\
       type_0_hold_vol_ratio,\
       type_1_hold,\
       type_1_vol,\
       type_1_hold_vol_ratio,\
       type_n_hold,\
       type_n_vol,\
       type_n_hold_vol_ratio,\
       type_n_hold_ratio,\
       type_n_vol_ratio,\
       type_s_hold,\
       type_s_vol,\
       type_s_hold_vol_ratio,\
       type_s_hold_ratio,\
       type_s_vol_ratio,\
       type_l_hold,\
       type_l_vol,\
       type_l_hold_vol_ratio,\
       type_l_hold_ratio,\
       type_l_vol_ratio,\
       type_a_hold,\
       type_a_vol,\
       type_a_hold_vol_ratio,\
       type_a_hold_ratio,\
       type_a_vol_ratio \
  from jgfzjc_db.czce_seq2seq_features \
 where  commid_type='main' and trade_date <='" + trade_date + "' \
 order by trade_date asc \
;"
    cur.execute(sql)
    df_data = as_pandas(cur)
    cur.close()
    return df_data


# params
enc_len = 1
# dec_len = 5
n_units = 128
n_output = 1
dropout = 0.1
# 每个梯度批次的数量
batch_size = 2 ** 7
# 迭代轮次
epochs = 20
debug_mode=False
INCEPTOR_CONFIG = dict(host="10.10.180.249",port=10000,user='dm1',password='1234567',database='jgfzjc_db',auth_mechanism='PLAIN')


pred_value_weight = np.array([[0.6,0,0,0,0],
                         [0.2,0.6,0,0,0],
                         [0.1,0.2,0.7,0,0],
                         [0.05,0.1,0.2,0.8,0],
                         [0.05,0.1,0.1,0.2,1]])



## 预测所有的主力合约
def predict_all_commid():
    results = []
    commids = []
    classids = []
    trade_dates = []

    for i in df_data.classid.unique():
        print(i)
    
        tmp_data = df_data[df_data.classid == i]
    
        trade_date = tmp_data[-1:][['trade_date']]
        commid = tmp_data[-1:][['commid']]
    
        tmp_data = tmp_data.drop(['classid','commid','trade_date'], axis=1)
    
        if len(tmp_data) <= 20:
            print('class ', i ,' has enough data to train.')
            continue

        n_input = tmp_data.shape[1]
        
        pred_y = predict(tmp_data, n_input)
        
        results.append(pred_y)
        classids.append(i)
        commids.append(commid.commid.values)
        trade_dates.append(trade_date.trade_date.values)
        
    results = np.round(np.array(results))
    trade_dates = np.array(trade_dates)
    simple_name = {'AP':'苹果', 'CF':'棉花', 'CY':'棉纱', 'FG':'玻璃', 'MA':'甲醇', 'OI':'菜籽油', 'RM':'菜籽粕', 'SF':'硅铁', 'SM':'锰硅', 'SR':'白糖', 'TA':'PTA', 'WH':'强麦','ZC':'郑煤'}
    df_results = pd.DataFrame(columns=['trade_date','classid','commid','simple_name','after_1d_vol','after_2d_vol','after_3d_vol','after_4d_vol','after_5d_vol'])
    df_results['classid'] = classids
    df_results['commid'] = np.array(commids)
    df_results['simple_name'] = [simple_name[i] for i in classids]
    df_results['trade_date'] = np.array(trade_dates)
    df_results['after_1d_vol'] = results[:,0]
    df_results['after_2d_vol'] = results[:,1]
    df_results['after_3d_vol'] = results[:,2]
    df_results['after_4d_vol'] = results[:,3]
    df_results['after_5d_vol'] = results[:,4]
    return df_results


## 将结果写入Inceptor
def df_results_to_inceptor(df_results):
    '''
    将预测结果写入inceptor
    '''
    inceptor = pyodbc.connect("DSN=inceptor_db")

    cursor = inceptor.cursor()

    for i in range(len(df_results)):
        cursor.execute("insert into jgfzjc_db.czce_predict values(?,?,?,?,?,?,?,?,?,? )",
                   (df_results.values[i][0]+df_results.values[i][2],
                    df_results.values[i][0],
                    df_results.values[i][1],
                    df_results.values[i][2],
                    df_results.values[i][3],
                    df_results.values[i][4],
                    df_results.values[i][5],
                    df_results.values[i][6],
                    df_results.values[i][7],
                    df_results.values[i][8]
                   ))

    cursor.close()
    
	
import datetime

## 循环预测多天进行测试
date = datetime.datetime.strptime('2018-12-06', '%Y-%m-%d')
end = datetime.datetime.strptime('2018-12-10', '%Y-%m-%d')
while date <= end:
    print(date.strftime('%Y-%m-%d'))
    
    # 获取数据
    df_data = get_df_data(date.strftime('%Y-%m-%d'))
    print(df_data.shape)
    
    # 预测
    df_results = predict_all_commid()
    
    # 预测结果写入到inceptor
    df_results_to_inceptor(df_results)

    date = date + datetime.timedelta(1)
    
  
	
	
	
