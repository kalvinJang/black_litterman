'''
@author : KY.Jang
'''

import scipy.optimize as sco
import pandas as pd
import numpy as np
from numpy.linalg import inv
import datetime
from dateutil.relativedelta import relativedelta


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

def var(rt: pd.Series, alpha=5):
    screen_var = rt.rank()==int(alpha/100*len(rt.dropna()))
    screen_cvar = rt.rank()<=int(alpha/100*len(rt.dropna()))
    var = rt[screen_var].values[0]*100
    cvar = ((screen_cvar)*rt).replace(0, np.nan).dropna().mean()*100
    return [var, cvar]


def entire_disparity(df):
    '''
    df : dataframe that include two series for calculating disparity
    '''
    df_view = df.iloc[-100:,:]
    df_view['disparity'] = df_view.iloc[:,0]/df.iloc[:,1]
    z=(df_view['disparity']-df_view['disparity'].mean())/df_view['disparity'].std()
    if (z[-1]>0.5)&(z[-1]<=2.5):
        view='L'
    elif (z[-1]>=-2.5)&(z[-1]<-0.5):
        view='S'
    else:
        view='N'
    return view

def asset_disparity(df):
    df_view = df.iloc[-100:]
    z=(df_view-df_view.mean())/df_view.std()
    if (z[-1]>0.5)&(z[-1]<=2.5):
        view='L'
    elif (z[-1]>=-2.5)&(z[-1]<-0.5):
        view='S'
    else:
        view='N'
    return view

def cal_q_mean(df_r, df_d):
    '''
    df_r : 26week-rolling weekly returns
    df_d : daily returns
    '''
    mean_rt = []
    view_list = []
    for j in range(df_r.shape[1]):
        x = df_r.iloc[:,j]
        view = asset_disparity(df_d.iloc[-100:,j])  #자산별 전망은 최근 100일을 봐야하니 daily return
        view_list.append(view)
        df_rank = x.rank()
        if view=='L':  # mean값 집어넣을 때는 최근 n년간의 주간 수익률의 평균을 집어넣음
            mean_rt.append((((df_rank<=0.8*x.shape[0])&(df_rank>=0.6*x.shape[0]))*x).replace(0, np.nan).dropna().mean())
        elif view=='S':   
            mean_rt.append((((df_rank<=0.4*x.shape[0])&(df_rank>=0.2*x.shape[0]))*x).replace(0, np.nan).dropna().mean())
        else:
            mean_rt.append(x.mean())
    return mean_rt, view_list

def error_cov_matrix(C, tau, P, diag):
    if diag:
        matrix = np.diag(np.diag(P.dot(tau * C).dot(P.T)))
    else:
        matrix = ((tau*P)@C)@P.T   # 위는 대각원소만 뽑은거고 이렇게 하라는 게 정설인데 왜인지 값이 이상해진다..
#                                 off-diagonal을 0으로 안 만드는 게 결과값이 더 그럴듯함
    return matrix

def P_matrix(n):
    c=[]
    for i in range(n):
        a = [0]
        b = [0]*i+[1]+[0]*(n-i-1)
        c.append(b)
    return np.asarray(c)

def maxsharpe(weights, rt, cov):
    return ((np.matrix(weights).dot(cov)).dot(np.matrix(weights).T).item() / (rt * weights).sum())

def implied_rets(risk_aversion, sigma, w):
    implied_rets = risk_aversion * sigma.dot(w).squeeze()
    return implied_rets

def BL(asset, data, asset_weights, rolling=True, n=5, rolling_week=26, diag=False, rf=0, tau = 1, band=[0.1, 0.2, 0.2, 0.2, (0.01, 0.05)]):
    '''
    band에 float가 들어오면 band_width로 삼고, tuple이 들어오면 그걸 band로 삼음
    '''
    data.index = pd.to_datetime(data.index)
    
    rt = data.pct_change().dropna(how='all')
    rt_w = rt.resample('W-Fri').sum()  # 주간 수익률을 기준으로 모든 걸 처리. cov 게산 등등
    rt_r= rt_w.rolling(rolling_week).sum().iloc[-n*52:,:]    # 최근 26주의 rolling sum() => 6개월 수익률 : cal_q_mean()에 넣기위함
    rt_n = rt_w.iloc[-n*52:,:]
            
    excess_asset_returns = rt_n.subtract(rf, axis=0)
    cov = excess_asset_returns.cov()
    
    global_return = (excess_asset_returns* asset_weights).sum(1).mean()
    market_var = np.matmul(pd.DataFrame(asset_weights).T,
                           np.matmul(cov.values, asset_weights)).values[0]
    risk_aversion = global_return / market_var
    print(f'The risk aversion parameter is {risk_aversion:.2f}')
    
    implied_equilibrium_returns = implied_rets(risk_aversion, cov, asset_weights)
    
################################## View 포함 시키기 ###########################################
    if rolling:
        view_rt, view_list = cal_q_mean(rt_r, rt)
        Q = np.array([x/rolling_week for x in view_rt])
    else:
        view_rt, view_list = cal_q_mean(rt_w, rt)
        Q = np.array(view_rt)
    
    print('view for each asset : ', view_list)
    P = P_matrix(data.shape[1])
    omega = error_cov_matrix(cov, tau, P, diag)
    sigma_scaled = cov * tau
    BL_return_vector = inv(inv(sigma_scaled)+(P.T).dot(inv(omega)).dot(P)).dot(
        inv(sigma_scaled).dot(implied_equilibrium_returns)+(P.T).dot(inv(omega)).dot(Q))
    
    if rolling:
        returns_table = pd.concat([pd.Series([x/rolling_week for x in view_rt], index=rt.columns), pd.Series(implied_equilibrium_returns, index=rt.columns), pd.Series(BL_return_vector, index=rt.columns)], axis=1)
    else:
        returns_table = pd.concat([pd.Series(view_rt, index=rt.columns), pd.Series(implied_equilibrium_returns, index=rt.columns), pd.Series(BL_return_vector, index=rt.columns)], axis=1)

    returns_table.columns = ['View for each asset', 'Implied Returns', 'BL Return Vector']
    returns_table['Difference'] = returns_table['BL Return Vector'] - returns_table['Implied Returns']
    
    other_list = []
    cons = [{'type': 'eq', 'fun': lambda x:  np.sum(x) - 1},
            {'type':'ineq', 'fun': lambda x: x}] #cons는 제약조건 
    option = {'maxiter': 10000}     # maxiter는 최대반복횟수
    
    bnds = []  #자산군별 weight 제약조건
    for ind, width in enumerate(band):
        if type(width) == tuple:
            bnds.append(width)
        elif (type(width)==float)&(width<1):
            bnds.append((asset_weights[ind]*(1-width), asset_weights[ind]*(1+width)))
    print('applied band : ',bnds)
    opt = sco.minimize(maxsharpe, asset_weights, args=(returns_table['BL Return Vector'], cov), method='SLSQP', constraints=cons, bounds=bnds, options = option)
    # SLSQP 이외에는 weight합이 1이 되도록 하는 constraint가 적용 안 됨..
    print('최적화 성공 여부' , opt['success'])
    w_result = pd.DataFrame({'original weight':np.array(asset_weights).round(6),
                              'BL weight':opt['x'].round(6),
                              'difference': (opt['x']-asset_weights).round(6)}, index=returns_table.index).T
    w_result['가중치 합'] = w_result.sum(1)
    
    ######################## weight * expected return 의 샤프, 변동성 등 비교 ##########################
    init_rets = np.sum(returns_table['BL Return Vector']*asset_weights)*100*4
    init_vol = np.sqrt(np.array(asset_weights).T @ cov @ asset_weights)*100/np.sqrt(n*12)
    init_sharpe = (init_rets-rf)/init_vol
    other_list.append([init_rets, init_vol, init_sharpe])
    
    opt_rets = np.sum(returns_table['BL Return Vector']*opt['x'])*100*4 #자산 별 E(R)*w해서 최적해로 포트 만들었을 때의 return
    opt_vol = np.sqrt(opt['x'].T @ cov @ opt['x'])*100/np.sqrt(n*12) #optimizied volatility
    opt_sharpe = (opt_rets-rf)/opt_vol
    other_list.append([opt_rets, opt_vol, opt_sharpe])
    
    comp_rets = np.sum(returns_table['View for each asset']*asset_weights)*100*4
    comp_vol = np.sqrt(np.array(asset_weights).T @ cov @ asset_weights)*100/np.sqrt(n*12)
    comp_sharpe = (comp_rets-rf)/comp_vol
    other_list.append([comp_rets, comp_vol, comp_sharpe])
    
    comp_rets2 = np.sum(returns_table['View for each asset']*opt['x'])*100*4
    comp_vol2 = np.sqrt(opt['x'].T @ cov @ opt['x'])*100/np.sqrt(n*12)
    comp_sharpe2 = (comp_rets2-rf)/comp_vol
    other_list.append([comp_rets2, comp_vol2, comp_sharpe2])
    
    comp_rets4 = np.sum(returns_table['Implied Returns']*asset_weights)*100*4
    comp_vol4 = np.sqrt(np.array(asset_weights).T @ cov @ asset_weights)*100/np.sqrt(n*12)
    comp_sharpe4 = (comp_rets4-rf)/comp_vol4
    other_list.append([comp_rets4, comp_vol4, comp_sharpe4])
    
    comp_rets3 = np.sum(returns_table['Implied Returns']*opt['x'])*100*4
    comp_vol3 = np.sqrt(opt['x'].T @ cov @ opt['x'])*100/np.sqrt(n*12)
    comp_sharpe3 = (comp_rets3-rf)/comp_vol3
    other_list.append([comp_rets3, comp_vol3, comp_sharpe3])
    

    # other_result = pd.DataFrame(other_list, columns=['포트 기대수익률', 'vol', '샤프'], index=['initial weight*BL', 'BL weight*BL',
    #                                                                                   'init_weight*View', 'BL_weight*View',
    #                                                                                   'init_weight*implied', 'BL_weight*implied'])
    return returns_table.T.round(6), w_result.round(6), view_list