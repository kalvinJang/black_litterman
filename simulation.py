'''
@author : KY.Jang
'''

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

def cummdd(return_series: pd.Series):  #input은 %수치가 아님. 1% = 0.01로 바꿔서 넣어야함.
    return drawdown(return_series)['Drawdown'].cummin()

def monthly_rebalanced_daily_history(rt, rebalance_weight, start_date, end_date, slpgCost=0):
    if rt.shape[1]!=len(rebalance_weight):
        raise Exception('The number of asset is not equal to that of weight')
    if np.isnan(rt.iloc[0,:]).sum()>0:
        rt = rt.dropna()
        print('Some assets have no return data at the start_date point. Automatically adjusting the start_date that every series has full values')
    rt = rt.loc[start_date:end_date, :]
    weight = np.divide(rebalance_weight, sum(rebalance_weight))
    
    n_months = (relativedelta(pd.to_datetime(rt.index[-1]), pd.to_datetime(rt.index[0])).years*12 +
                relativedelta(pd.to_datetime(rt.index[-1]), pd.to_datetime(rt.index[0])).months)
    start = pd.date_range(start=rt.index[0], end=rt.index[-1], freq='BMS')  # rt.index[0]가 매월 3일 이후인 날이면 다음달부터 표시함
    end = pd.date_range(start=rt.index[0], end=rt.index[-1], freq='BM')
    temp = pd.DataFrame(weight).T
    temp.columns = rt.columns
    rep = 0
    while (start[0] > end[0]):
        month_ago = [x-relativedelta(days = rep+1) for x in start]
        month_ago = [x-relativedelta(months = 1) for x in month_ago]
        start = pd.date_range(start=month_ago[0], end=end[-1], freq='BMS')
        rep+=1
        if rep>30:
            raise Exception('index error')
    for i in range(n_months):
        temp2 = (1+rt.loc[start[i].date():end[i].date(), :]) * weight
        temp2.iloc[-1,:] = weight
        temp = pd.concat([temp, temp2])
    daily_weight = temp.shift(1).dropna(how='all')
    daily_weight = daily_weight.apply(lambda x: x/np.sum(x), axis=1)
    daily_weight = daily_weight.replace(np.nan, 0)
    daily_weight.index = pd.to_datetime(daily_weight.index)
    pf =  (daily_weight * (1+rt.loc[daily_weight.index[0]:daily_weight.index[-1],:])).sum(1)
    daily_history = pf.cumprod()/pf[0]*1000
    daily_history.name = 'portfolio index'
    pf.name = 'daily return(%)'
    
#     display(plt.plot(daily_history))
#     display(daily_history.iplot(kind='line'))
    temp_result = pd.concat([daily_history, (pf-1)*100], axis=1)
    
    ########## Rolling performance ###################
    
    y1_rolling_daily_return = ((1+temp_result['daily return(%)']/100).rolling(260).apply(np.prod, raw=True) - 1)*100
    y2_rolling_daily_return = ((1+temp_result['daily return(%)']/100).rolling(260*2).apply(np.prod, raw=True) - 1)*100
    y3_rolling_daily_return = ((1+temp_result['daily return(%)']/100).rolling(260*3).apply(np.prod, raw=True) - 1)*100
    y1_rolling_daily_return.name = 'y1_rolling_daily_return'
    y2_rolling_daily_return.name = 'y2_rolling_daily_return'
    y3_rolling_daily_return.name = 'y3_rolling_daily_return'
    
    annu_y1_r_daily_return = y1_rolling_daily_return.copy()
    annu_y2_r_daily_return = ((1+y2_rolling_daily_return/100)**(1/2)-1)*100
    annu_y3_r_daily_return = ((1+y3_rolling_daily_return/100)**(1/3)-1)*100
    annu_y1_r_daily_return.name = 'annu_y1_r_daily_return'
    annu_y2_r_daily_return.name = 'annu_y2_r_daily_return'
    annu_y3_r_daily_return.name = 'annu_y3_r_daily_return'
    
    annu_y1_r_daily_vol = (temp_result['daily return(%)']/100).rolling(260).std() * np.sqrt(260) *100
    annu_y2_r_daily_vol = (temp_result['daily return(%)']/100).rolling(260*2).std() * np.sqrt(260) *100
    annu_y3_r_daily_vol = (temp_result['daily return(%)']/100).rolling(260*3).std() * np.sqrt(260)*100
    annu_y1_r_daily_vol.name = 'annu_y1_r_daily_vol'
    annu_y2_r_daily_vol.name = 'annu_y2_r_daily_vol'
    annu_y3_r_daily_vol.name = 'annu_y3_r_daily_vol'
    
    mdd = cummdd(temp_result['daily return(%)']/100)*100
    mdd.name = 'mdd'
    
    rolling_return_result = pd.concat([y1_rolling_daily_return, y2_rolling_daily_return, y3_rolling_daily_return,
                                      annu_y1_r_daily_return, annu_y2_r_daily_return, annu_y3_r_daily_return,
                                      annu_y1_r_daily_vol, annu_y2_r_daily_vol, annu_y3_r_daily_vol,
                                      mdd], axis=1)
    
    return pd.concat([temp_result, rolling_return_result], axis=1)

def mdd_period(rt):
    '''
    rt : 기준이 0이고 %아닌 수익률
    '''
    xs = drawdown(rt)['Drawdown'] #xs : drawdown함수 결과값의 ['Drawdown'], drawdown series (it is negative value)
    mdd_point = np.argmin(xs) #MDD가 찍힌 시점
    mdd_restored = np.argmax(np.maximum.accumulate((xs[mdd_point:]-np.minimum.accumulate(xs[mdd_point:])))) #MDD가 복구된 시점이 MDD 몇일 후인가
    mdd_start = np.argmax(xs[:mdd_point+1][::-1])  #MDD 찍히기 며칠 전부터 MDD시작되었는가
    mdd=xs[mdd_point-mdd_start: mdd_point+mdd_restored+1]
    if mdd.index[-1]> datetime.datetime.now()-relativedelta(months=1):
        mdd.iloc[-1]='ing'
    return mdd_start, mdd_restored, mdd

def var(rt: pd.Series, alpha=5):
    screen_var = rt.dropna().rank().astype(int)==int(alpha/100*len(rt.dropna()))
    screen_cvar = rt.dropna().rank().astype(int)<=int(alpha/100*len(rt.dropna()))
    var = sorted(rt)[int(alpha/100*len(rt.dropna()))-1]*100
    cvar = ((screen_cvar)*rt).replace(0, np.nan).dropna().mean()*100
    return [var, cvar]

def record(rt, rebalance_weight, start_date, end_date, rebalancig_freq='M', significance_level=5,slpgCost=0):
    '''
    rt : daily return_series : pd.DataFrame, dim : (# of days, # of assets)
    rebalance_weight : asset alocation weight : list
    start_date, end_date : 'yyyy-mm' : str
    rebalancig_freq : one of ['Y', 'M', 'W', 'D'] ; 'D'로 할 경우 business day가 아니라 주말 포함 calendar day
    '''
    # performance of portfolio, assets for each year
    if rt.shape[1]!=len(rebalance_weight):
        raise Exception('The number of asset is not equal to that of weight')
    if np.isnan(rt.iloc[0,:]).sum()>0:
        rt = rt.dropna()
        print('Some assets have no return data at the start_date point. Automatically adjusting the start_date that every series has full values')
    rt_resample = (1+rt.loc[start_date:end_date, :]).resample(rebalancig_freq).prod()-1
    weight = np.divide(rebalance_weight, sum(rebalance_weight)).tolist()
    pf_rt = (rt_resample * weight).sum(1) * (1-slpgCost)  #rebalancig_freq rt 구해서 weight랑 곱하는 방식
    pf_rt.name='portfolio'
    all_in_one = pd.concat([pf_rt, rt_resample], axis=1)   #기준 0인 rebalancig_freq return
    result_table_yearly = (((1+all_in_one).resample('Y').prod()-1)*100).round(2)
    result_table_yearly.index = [str(x) for x in pd.to_datetime(result_table_yearly.index).year]
    
    #calculate cumulative return, annualized return, SD, R/SD, MDD and merge to result table
    cum_return = ((1+all_in_one).prod()-1)*100   #단위가 %,  cum_return값이 42면 n년 동안 142%가 되었다. 1.42배 됐다
#     num_of_days = (all_in_one.index[-1] - all_in_one.index[0]).days    #이건 business day가 아님
    num_of_days = rt.shape[0]
    annu_return = (((1+all_in_one).prod()**(260/num_of_days))-1)*100  #이미 단위가 %이므로  ann_return값이 1.41이면 매년 1.4%씩 성장
    if rebalancig_freq=='M':
        word = 'month'
        n = 12
    elif rebalancig_freq=='W':
        word = 'week'
        n =52
    elif rebalancig_freq=='D':
        word = 'calendar day'
        n = 260
    elif rebalancig_freq=='Y':
        word = 'year'
        n=1

    SD = np.std(all_in_one) * np.sqrt(n) * 100
    dSD = np.std((all_in_one<0)*all_in_one) * np.sqrt(n) * 100
    MDD=all_in_one.apply(cummdd).iloc[-1,:]*100
    record = pd.DataFrame({
        'cum_return(%)': cum_return,
        'annu_return(%)': annu_return,
        'SD(%)':SD,
        'MDD(%)': MDD,
        'R/SD':np.divide(annu_return, SD.tolist()),
        'R/MDD': -np.divide(annu_return, MDD),
        'Sortino' : np.divide(annu_return, dSD.tolist()),
        'MDD lasting time ({})'.format(word): all_in_one.apply(mdd_period).iloc[0],
        'MDD restoring time ({})'.format(word): all_in_one.apply(mdd_period).iloc[1],
        '{}% VaR'.format(significance_level): all_in_one.apply(var, alpha=significance_level).iloc[0],
        '{}% CVaR'.format(significance_level):all_in_one.apply(var, alpha=significance_level).iloc[1]
    })
    return pd.concat([record.T, result_table_yearly]), all_in_one, record

def rolling_performance(x, target_rate=7, n =5):
    '''
    x : first output of func(monthly_rebalanced_daily_history)
    '''
    result = pd.DataFrame({
        '1Y':[
            x.shape[0],
            x['y1_rolling_daily_return'].dropna().shape[0],
            x['y1_rolling_daily_return'].dropna().mean(),
            x['annu_y1_r_daily_return'].dropna().mean(),
            x['annu_y1_r_daily_vol'].dropna().mean(),
            x['annu_y1_r_daily_return'].dropna().mean()/x['annu_y1_r_daily_vol'].dropna().mean(),
            -x['annu_y1_r_daily_return'].dropna().mean()/x['mdd'].dropna().min(),
            x['annu_y1_r_daily_return'].dropna().mean()/(((x['daily return(%)'].dropna()<0) * (x['annu_y1_r_daily_return']/100).dropna()).std()*100),
            x['y1_rolling_daily_return'].dropna().max(),
            x['y1_rolling_daily_return'].dropna().min(),
            x['annu_y1_r_daily_return'].dropna().max(),

            ((x['annu_y1_r_daily_return'].dropna().rank() > (1-n/100)*x['annu_y1_r_daily_return'].dropna().shape[0])*x['annu_y1_r_daily_return'].dropna()).replace(0, np.nan).dropna().mean(),
            (((x['annu_y1_r_daily_return'].dropna().rank() < (1-n/100)*x['annu_y1_r_daily_return'].dropna().shape[0])&(x['annu_y1_r_daily_return'].dropna().rank() > n/100*x['annu_y1_r_daily_return'].dropna().shape[0]))*x['annu_y1_r_daily_return'].dropna()).replace(0, np.nan).dropna().mean(),
            ((x['annu_y1_r_daily_return'].dropna().rank() < n/100*x['annu_y1_r_daily_return'].dropna().shape[0])*x['annu_y1_r_daily_return'].dropna()).replace(0, np.nan).dropna().mean(),

            x['annu_y1_r_daily_return'].dropna().min(),
            (x['y1_rolling_daily_return'].dropna()<0).sum(), 
            100*(x['y1_rolling_daily_return'].dropna()<0).sum()/x['y1_rolling_daily_return'].dropna().shape[0],
            ((x['y1_rolling_daily_return'].dropna()<0) * x['y1_rolling_daily_return'].dropna()).mean(),
            ((x['annu_y1_r_daily_return'].dropna()<0) * x['annu_y1_r_daily_return'].dropna()).mean(),
            100*(x['y1_rolling_daily_return'].dropna()>target_rate).sum()/x['y1_rolling_daily_return'].dropna().shape[0]
             ],
        
        '2Y':[
            x.shape[0],
            x['y2_rolling_daily_return'].dropna().shape[0],
            x['y2_rolling_daily_return'].dropna().mean(), 
            x['annu_y2_r_daily_return'].dropna().mean(),
            x['annu_y2_r_daily_vol'].dropna().mean(),
            x['annu_y2_r_daily_return'].dropna().mean()/x['annu_y2_r_daily_vol'].dropna().mean(),
            -x['annu_y2_r_daily_return'].dropna().mean()/x['mdd'].dropna().min(),
            x['annu_y2_r_daily_return'].dropna().mean()/(((x['daily return(%)'].dropna()<0) * (x['annu_y2_r_daily_return']/100).dropna()).std()*100),
            x['y2_rolling_daily_return'].dropna().max(), 
            x['y2_rolling_daily_return'].dropna().min(),
            x['annu_y2_r_daily_return'].dropna().max(),
            
            ((x['annu_y2_r_daily_return'].dropna().rank() > (1-n/100)*x['annu_y2_r_daily_return'].dropna().shape[0])*x['annu_y2_r_daily_return'].dropna()).replace(0, np.nan).dropna().mean(),
            (((x['annu_y2_r_daily_return'].dropna().rank() < (1-n/100)*x['annu_y2_r_daily_return'].dropna().shape[0])&(x['annu_y2_r_daily_return'].dropna().rank() > n/100*x['annu_y2_r_daily_return'].dropna().shape[0]))*x['annu_y2_r_daily_return'].dropna()).replace(0, np.nan).dropna().mean(),
            ((x['annu_y2_r_daily_return'].dropna().rank() < n/100*x['annu_y2_r_daily_return'].dropna().shape[0])*x['annu_y2_r_daily_return'].dropna()).replace(0, np.nan).dropna().mean(),

            x['annu_y2_r_daily_return'].dropna().min(),
            (x['y2_rolling_daily_return'].dropna()<0).sum(), 
            100*(x['y2_rolling_daily_return'].dropna()<0).sum()/x['y2_rolling_daily_return'].dropna().shape[0],
            ((x['y2_rolling_daily_return'].dropna()<0) * x['y2_rolling_daily_return'].dropna()).mean(),
            ((x['annu_y2_r_daily_return'].dropna()<0) * x['annu_y2_r_daily_return'].dropna()).mean(),
            100*(x['y2_rolling_daily_return'].dropna()>target_rate).sum()/x['y2_rolling_daily_return'].dropna().shape[0]
             ],
        
        '3Y':[
            x.shape[0],
            x['y3_rolling_daily_return'].dropna().shape[0],
            x['y3_rolling_daily_return'].dropna().mean(), 
            x['annu_y3_r_daily_return'].dropna().mean(),
            x['annu_y3_r_daily_vol'].dropna().mean(),
            x['annu_y3_r_daily_return'].dropna().mean()/x['annu_y3_r_daily_vol'].dropna().mean(),
            -x['annu_y3_r_daily_return'].dropna().mean()/x['mdd'].dropna().min(),
            x['annu_y3_r_daily_return'].dropna().mean()/(((x['daily return(%)'].dropna()<0) * (x['annu_y3_r_daily_return']/100).dropna()).std()*100),
            x['y3_rolling_daily_return'].dropna().max(), 
            x['y3_rolling_daily_return'].dropna().min(),
            x['annu_y3_r_daily_return'].dropna().max(),
            
            ((x['annu_y3_r_daily_return'].dropna().rank() > (1-n/100)*x['annu_y3_r_daily_return'].dropna().shape[0])*x['annu_y3_r_daily_return'].dropna()).replace(0, np.nan).dropna().mean(),
            (((x['annu_y3_r_daily_return'].dropna().rank() < (1-n/100)*x['annu_y3_r_daily_return'].dropna().shape[0])&(x['annu_y3_r_daily_return'].dropna().rank() > n/100*x['annu_y3_r_daily_return'].dropna().shape[0]))*x['annu_y3_r_daily_return'].dropna()).replace(0, np.nan).dropna().mean(),
            ((x['annu_y3_r_daily_return'].dropna().rank() < n/100*x['annu_y3_r_daily_return'].dropna().shape[0])*x['annu_y3_r_daily_return'].dropna()).replace(0, np.nan).dropna().mean(),
            
            x['annu_y3_r_daily_return'].dropna().min(),
            (x['y3_rolling_daily_return'].dropna()<0).sum(), 
            100*(x['y3_rolling_daily_return'].dropna()<0).sum()/x['y3_rolling_daily_return'].dropna().shape[0],
            ((x['y3_rolling_daily_return'].dropna()<0) * x['y3_rolling_daily_return'].dropna()).mean(),
            ((x['annu_y3_r_daily_return'].dropna()<0) * x['annu_y3_r_daily_return'].dropna()).mean(),
            100*(x['y3_rolling_daily_return'].dropna()>target_rate).sum()/x['y3_rolling_daily_return'].dropna().shape[0]
             ]
    })
    result.index = ['총 관찰일수', '시행횟수', '평균 누적수익률(%)', '평균 연환산 수익률(%)', '평균 연환산 변동성(%)', 'R/SD', 'R/MDD',
                   'Sortino', '기간 최대수익률(%)', '기간 최저수익률(%)',  '연환산 최대수익률(%)',  '연환산 상위 {}% 평균수익률(%)'.format(n),
                   '연환산 {}% trimmed 평균수익률(%)'.format(n),'연환산 하위 {}% 평균수익률(%)'.format(n),
                    '연환산 최저수익률(%)', '손실횟수', '손실확률(%)',
                   '기간 평균손실(%)', '연환산 평균손실(%)', '목표수익률 상회 확률(%)']
    return result