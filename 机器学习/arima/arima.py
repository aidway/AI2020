from statsmodels.tsa.arima model import ARIMA
from arch.unitroot import ADF
import pmdarima as pmd
import numpy as пр
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox as lbtest
from arch import arch.model

import pyodbc
from impala.dbapi import connect
from impala.util import as_pandas

import configparser
import sys
import statsmodels.api as sm
import time
import os
abs file = _ file_



def LB_test(ts):
    '''
    作用：LB检验，检验数据ts是否为白噪声。对arima的残差进行检验，如果不是白噪声，就进行garch模型进行预测
    p > 0.05，是白噪声
    ts：残差序列
    '''
    lbvalue,p1_value=lbtest(ts,lags=None) 
    lbvalue,p2_value=lbtest(ts**2,lags=None) #lags = min((nobs//2 -2,40))  nobs :样本观测值数量 
    return p2_value


def QQ_plot(ts):    
    '''作用：QQ图，检验ts是否为白噪声'''
    plt.figure(figsize=(10,10))
    qq_ax=plt.subplot2grid((3,2),(2,0))
    sm.qqplot(ts,line='s',ax=qq_ax)
    
def draw_acf_pacf(ts,lags):
    '''作用：自相关图、偏自相关图
    lags:滞后期，一般用30'''
    f=plt.figure(facecolor='white')
    ax1=f.add_subplot(211)
    plot_acf(ts,ax=ax1,lags=lags)
    ax2=f.add_subplot(212)
    plot_pacf(ts,ax=ax2,lags=lags)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
def auto_arima_garch(ts,offset):
    '''
	建立ARIMA- GARCH模型:
	ts:要预测的数据

	ts commid:要预测的数据的合约

	offset: 保证要预测的数据全部大于的偏移量，手动计算，比如min(data)= -12 offset >12
	例如: ts = datal['vol'][:50]

	预测1期较好，后边判断使用的均为往后1期的预测值
	'''
    ts_log = np.log(ts + offset)
    model = pmd.auto_arima(ts_log, max_p =10 , max_q = 10,max_d = 1,seasonal=False,error_action ='ignore',suppress_warnings = True) 
    
    result_arima = model.fit(ts_log)
    r = result_arima.resid()
    forecast_ts1 = result_arima.predict(5)
    
    w = LB_test(r)
    forecast1 = np.exp(forecast_ts1) - offset
    return forecast1
    
def pred_results_to_inceptor(trade_date,commid,commid_type,pred_results,table_name):
    '''将预测结果写入inceptor'''
    db = pyodbc.connect(CONN)
    cursor = db.cursor()
    cursor.execute()
    cursor.close()
    
def get_train_data(trade_date,start_date):
    '''作用：根据输入的日期，获取特征数据'''
    con = connect(**INCEPTOR_CONFIG)
    cur = con.cursor()
    sql = ""
    cur.execute(sql)
    df_data = as_pandas(cur)
    cur.close()
    return df_data
    
    
def get_trade_date():
    '''作用：获取交易日期'''
    con = connect(**INCEPTOR_CONFIG)
    cur = con.cursor()
    sql = ""
    cur.execute(sql)
    df_data = as_pandas(cur)
    cur.close()
    return df_data.today.values[0],df_data.start_date.values[0]
    
if_name_ == "_main_":
	#加载参数
	parent_dir = os.path.dirname(os.path.abspath(abs_file))
	cf = configparser.ConfigParser()
	cf.read(parent_dir + '/config.ini')

	Driver = cf.get('inceptor', 'Driver') 
	CONN = "Driver= %s;Server =%s;Hive=%s;Host=%s,Port=%s;Database=%s;User =%s;Password=%s;Mech=%s" %(Driver, Server, Hive, Host, Port, Database, User, Password, Mech)
	INCEPTOR_CONFIG = dict(host=Host,port=Port,user=User,password=Password,database=Database,auth_mechanism='PLAIN')
	
	MIN_TRAIN_DATA_NUM = cf.getint('model', 'MIN_TRAIN_DATA_NUM')
	
	tb_trade_date = cf.get('table','tb_trade_date')

	#获取交易日期
	trade_date, start_date = get_trade_date()

	#获取数据
	df_train_data = get_train_data(trade_date, start_date)
	df_pred_data = get_train_data(trade_date, trade_date)

	commid_num = len(df_pred_data)
	#遍历每个待预测合约
	for i in range(commid_num):
		#预测数据
		df_pred = df_pred_data[i:i+1]
		df_pred = df_pred.fillna(0)
		commid = df_pred.commid.values[0]
		commid_type = df_pred.commid_type.values[0]
		trade_date = df_pred.trade_date.values[0]

		#训练数据
		df_train = df_train_data[df_train_data.commid == commid]
		df_train = df_train.filla(method='bfill')
		df_train = df_train.dropna(axis=0, how='any')

		if len(df_train) <= MIN_TRAIN_DATA_NUM:
			print('trade_date:', trade_date, ',commid:',commid,' has no enough data to train.')
			continue
		print(commid,df_train.shape)

		df_train.index = pd.bdate_range(start='2000-01-01',periods=len(df_train),freq = 'D')

		#偏移量，确保数据大于0
		offset = 50
		
		#交易量预测
		ts_vol = df_train['vol'][:]
		#单位根检验
		if ADF(ts_vol).nobs < 30:
			print(commid + trade_date + 'ADF INVALID')
		else:
			data_pred(ts_vol,offset,trade_date,commid,commid_type,table_name = 'mssad_db.ad_predict_5d_vol_arima')

	print('arima model finished.')