'''
@author : KY.Jang
'''

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from Weekly_report import execute3, pivot
from simulation import *
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from statsmodels.tsa.vector_ar.vecm import *

def table(df):
    cum_return = ((1+df).prod()-1)*100   #단위가 %,  cum_return값이 42면 n년 동안 142%가 되었다. 1.42배 됐다
    #     num_of_days = (all_in_one.index[-1] - all_in_one.index[0]).days    #이건 business day가 아님
    num_of_day = df.notna().sum(0)
    annu_return = (((1+df).prod()**(262/num_of_day))-1)*100  #이미 단위가 %이므로  ann_return값이 1.41이면 매년 1.4%씩 성장
    # num_of_months = df.shape[0]
    # annu_return = (((1+df).prod()**(12/num_of_months))-1)*100 
    word = 'day'
    n = 252
    significance_level=5

    SD = np.std(df) * np.sqrt(n) * 100
    dSD = np.std((df<0)*df) * np.sqrt(n) * 100
    MDD=df.apply(cummdd).iloc[-1,:]*100
    record = pd.DataFrame({
        'cum_return(%)': cum_return,
        'annu_return(%)': annu_return,
        'SD(%)':SD,
        'MDD(%)': MDD,
        'R/SD':np.divide(annu_return, SD.tolist()),
        'R/MDD': -np.divide(annu_return, MDD),
        'Sortino' : np.divide(annu_return, dSD.tolist()),
        'MDD lasting time ({})'.format(word): df.apply(mdd_period).iloc[0],
        'MDD restoring time ({})'.format(word): df.apply(mdd_period).iloc[1],
        '{}% VaR'.format(significance_level): df.apply(var, alpha=significance_level).iloc[0],
        '{}% CVaR'.format(significance_level):df.apply(var, alpha=significance_level).iloc[1]
    })

    # cum_return_display = (1+df).cumprod()
    # print(cum_return_display)
    return pd.DataFrame(record.T)


def picking_high_std(x, n=260, q=0.8):
    '''
    x: monthly_rebalanced_daily_history 함수 결과값
    '''
    port_ind_std = x['portfolio index'].rolling(n).std()   # 인덱스의 std  -> 이게 시작은 더 잘 파악하는데
    port_ret_std = (x['daily return(%)']/100).rolling(n).std() #인덱스의 차분의 std -> 이게 last시점은 더 이후임.
    warning_ind = []
    for i,j in enumerate(port_ind_std):
        if j>port_ind_std.quantile(q):
            warning_ind.append(port_ind_std.index[i])
    warning_ret = []
    for i,j in enumerate(port_ret_std):
        if j>port_ret_std.quantile(q):
            warning_ret.append(port_ret_std.index[i])
    warning=np.unique(warning_ind+warning_ret)
    
    # 2주동안 지속되면 카운트
    to_show_warning = sorted([x for x in set(warning).intersection(set(warning+relativedelta(weeks=2)))])
    test = sorted([(a.year, a.week) for a in to_show_warning])

    period = [test[0]]
    for i in range(len(test)-1):
        if ((test[i][1]==test[i+1][1]) & (test[i][0]==test[i+1][0]))or((test[i][1]+1==test[i+1][1]) & (test[i][0]==test[i+1][0])):
            pass
        else:
            if (test[i+1][1]==1)&(test[i][1]>=52):
                pass
            else:
                period.append(test[i]) ## std높은 시기의 마지막 주
                period.append(test[i+1])
    period.append(test[-1])
    
    start_week = [period[i] for i in range(len(period)) if i%2==0]
    end_week = [period[i] for i in range(len(period)) if i%2==1]
    if (2021, 53) in start_week:   #파이썬 오류. period가 길면 (a.year, a.week)에 뜬금없이 (2021, 53)값 들어감. 2021년에는 53주차가 없음.
        start_week.remove((2021, 53))
    if (2021, 53) in end_week:
        end_week.remove((2021, 53))
    start_day = [datetime.date.fromisocalendar(start_week[i][0],start_week[i][1], 1) for i in range(len(start_week))]
    end_day = [datetime.date.fromisocalendar(end_week[i][0],end_week[i][1], 5) for i in range(len(end_week))]
    time_set = list(zip(start_day, end_day))
    return time_set, to_show_warning

def picking_high_std2(x: pd.Series, n=260, q=0.8): ## 펀드 하나에 대해서 돌릴 때는 이거 사용
    '''
    x: daily return of only one pd.Series
    '''
    port_ind_std = x.rolling(n).std().dropna()
    warning_ind = []
    for i,j in enumerate(port_ind_std):
        if j>port_ind_std.quantile(q):
            warning_ind.append(port_ind_std.index[i])
    warning=np.unique(warning_ind)
    
    # 2주동안 지속되면 카운트
    to_show_warning = sorted([x for x in set(warning).intersection(set(warning+relativedelta(weeks=2)))])
    test = sorted([(a.year, a.week) for a in to_show_warning])[:-1]
    period = [test[0]]
    for i in range(len(test)-1):
        if ((test[i][1]==test[i+1][1]) & (test[i][0]==test[i+1][0]))or((test[i][1]+1==test[i+1][1]) & (test[i][0]==test[i+1][0])):
            pass
        else:
            if (test[i+1][1]==1)&(test[i][1]>=52):
                pass
            else:
                period.append(test[i]) ## std높은 시기의 마지막 주
                period.append(test[i+1])
    period.append(test[-1])
    
    start_week = [period[i] for i in range(len(period)) if i%2==0]
    end_week = [period[i] for i in range(len(period)) if i%2==1]
    if (2021, 53) in start_week:   #파이썬 오류. period가 길면 (a.year, a.week)에 뜬금없이 (2021, 53)값 들어감. 2021년에는 53주차가 없음.
        start_week.remove((2021, 53))
    if (2021, 53) in end_week:
        end_week.remove((2021, 53))
    start_day = [datetime.date.fromisocalendar(start_week[i][0],start_week[i][1], 1) for i in range(len(start_week))]
    end_day = [datetime.date.fromisocalendar(end_week[i][0],end_week[i][1], 5) for i in range(len(end_week))]
    time_set = list(zip(start_day, end_day))
    return time_set, to_show_warning

def table_summary_for_qt(rt, time_set):
    return_data = rt
    summary = pd.DataFrame()
    for_table = pd.DataFrame()
    for i in range(len(time_set)):
        des = (((1+return_data).resample('M').prod()-1).dropna()).loc[time_set[i][0]:time_set[i][1]].describe()
        des.name = str(time_set[i][0])+' - '+str(time_set[i][1])
        summary = pd.concat([summary, des], axis=1)
        temp_table = (return_data.dropna()).loc[time_set[i][0]:time_set[i][1]]
        temp_table.name = str(time_set[i][0])+' - '+str(time_set[i][1])
        for_table = pd.concat([for_table, temp_table], axis=1)
    return table(for_table), summary.T.iloc[:,1:]*1200


def mse(model, steps):
        """
        model = vecm_res
        """
        ma_coefs = model.ma_rep(steps)

        k = len(model.sigma_u)
        forc_covs = np.zeros((steps, k, k))

        prior = np.zeros((k, k))
        for h in range(steps):
            # Sigma(h) = Sigma(h-1) + Phi Sig_u Phi'
            phi = ma_coefs[h]
            var = phi @ model.sigma_u @ phi.T
            forc_covs[h] = prior = prior + var

        return forc_covs
    
def fevd(model, P=None, periods=None):  #model = vecm_res
        from statsmodels.tsa.vector_ar import output
        from statsmodels.tsa.vector_ar.output import VARSummary
        from statsmodels.compat.python import lrange
        from io import StringIO
        
        self_model = model
        self_neqs = model.neqs
        self_names = model.model.endog_names

        temp_irf = model.irf(periods=periods)
        self_orth_irfs = temp_irf.orth_irfs

        # cumulative impulse responses
        irfs = (self_orth_irfs[:periods] ** 2).cumsum(axis=0)

        rng = lrange(self_neqs)
        self_mse = mse(self_model, periods)[:, rng, rng]

        # lag x equation x component
        fevd = np.empty_like(irfs)

        for i in range(periods):
            fevd[i] = (irfs[i].T / self_mse[i]).T

        # switch to equation x lag x component
        self_decomp = fevd.swapaxes(0, 1)
        
        buf = StringIO()
        rng = lrange(periods)
        for i in range(self_neqs):
            ppm = output.pprint_matrix(self_decomp[i], rng, self_names)
            buf.write("FEVD for %s\n" % self_names[i])
            buf.write(ppm + "\n")
            
            globals()['fevd_{}'.format(self_names[i][:-6])] = pd.DataFrame(self_decomp[i], index=rng, columns=self_names)
        return globals()['fevd_portfolio']

#         print(buf.getvalue())

def standardize(df):
    return (df-df.mean())/df.std()

# rotation과 factor 최적 조합 산출 - 시간 오래 걸림. 요인 찾는 단계에서만 사용
def factor_analysis(df, n_factors=9, rotation='varimax'):
    button=True
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    fa.fit(df)
    eigen_values, vectors = fa.get_eigenvalues()
    
    sub_factors = (eigen_values>1).sum()
    print(sub_factors)
    while True:
        fa = FactorAnalyzer(n_factors=sub_factors, rotation=rotation)
        fa.fit(df)
        if fa.get_factor_variance()[2][-1]>0.7:
            for i in range(sub_factors):
                w = fa.get_factor_variance()[1][i]
                if w>=0.05:
                    pass
                else:
                    sub_factors = i
                    print('found', i)
                    break
        break
    while True:
        sub_factors+=1
        fa = FactorAnalyzer(n_factors=sub_factors, rotation=rotation)
        fa.fit(df)
        w = fa.get_factor_variance()[1][-1]
        if w>=0.05:
            print('one more...')
            continue
        else:
            sub_factors -= 1
            break
    print('At last,', sub_factors, rotation)
    fa = FactorAnalyzer(n_factors=sub_factors, rotation=rotation)
    fa.fit(df)
    return pd.DataFrame(fa.get_factor_variance(), index=['SS Loadings=factor variance', 'Proportion Var', 'Cumulative Var'])

