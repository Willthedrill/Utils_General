import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
import warnings
import seaborn as sns
from IPython.display import clear_output
from scipy.stats import ttest_ind,ttest_1samp
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import pearsonr,spearmanr
import matplotlib.ticker as pltticker
import plotly.express as px
warnings.filterwarnings("ignore")
pd.get_option("display.max_columns")
import lightgbm as lgb
print("LightGBM version: {}".format(lgb.__version__))
from tqdm import tqdm

# import dataframe_image as dfi
# pd.set_option('display.max_rows', 200)
# from scipy.signal import lfilter,savgol_filter
# import pickle
# from collections import defaultdict
# from datetime import timedelta
# from datetime import time,date,timedelta,datetime
# from matplotlib import style

# from fclib.common.utils import git_repo_path
# from fclib.models.lightgbm import predict
# from fclib.evaluation.evaluation_utils import MAPE
# from fclib.common.plot import plot_predictions_with_history
# from fclib.dataset.ojdata import download_ojdata, split_train_test
# from fclib.dataset.ojdata import FIRST_WEEK_START
# from fclib.feature_engineering.feature_utils import (
#     week_of_month,
#     df_from_cartesian_product,
#     combine_features,
# )


def see_modules(module):
    '''
    See the defined functions in a module that belongs to the module itself. 
    Args: 
        module: the module of interest
    Returns:
        pandas.DataFrame: dataframe of module function names, function, and its docstring.  
    '''

    from inspect import getmembers,isfunction
    list1=pd.DataFrame(getmembers(module,isfunction),columns=['function_name','function'])
    list1=list1.loc[list1['function'].apply(lambda x:True if x.__module__==module.__name__ else False)]
    list1['docstring']=list1['function'].apply(lambda x:x.__doc__)
    return list1.sort_values('function_name')



def plot_scatter(sub,summary=False,s=10,title='',xlabel='',ylabel='',pvalue=False,res_return=False,show_regression=True):
    '''
    Args: 
        sub: DataFrame of two columns required for scatter. The first column should contain independent variables and the second column should contain dependent variable. 
        

    Return: 
        
    '''

    if isinstance(sub,pd.Series):
        sub=pd.DataFrame(sub).reset_index()
    X=sub[sub.columns[0]]
    Y=sub[sub.columns[1]]
    plt.figure(figsize=(10,6))
    plt.scatter(X,Y,s=s)
    plt.xlabel(sub.columns[0])
    plt.ylabel(sub.columns[1])
    X=sm.add_constant(X)
    res=sm.OLS(Y,X).fit()
    X=sub[sub.columns[0]]
    #display(X)
    if show_regression:
        plt.plot(X,list(res.params)[0]+list(res.params)[1]*X,label='regression line',linestyle='dashed')
    if summary:
        display(res.summary())
    if title!='':
        plt.title(title)
    if xlabel!='':
        plt.xlabels(xlabel)
    if ylabel!='':
        plt.ylabels(ylabel)
    if pvalue==True:
        print(res.pvalues[1])
    plt.legend()
    plt.grid()
    plt.show()
    if res_return==True:
        return res
def plot_line(sub,target1='',target2='',targetlist=[],figsize=(10,6),title='',twin='',xlabel='',ylabel='',fontsize=10,linewidth=1,savefig=False,annotate='',annotate_pos=(0.35,-0.1),grid=True,ax='',num_xticks='',style='whitegrid'):
    import seaborn as sns
    sns.set_style(style)
    if isinstance(sub,pd.Series):
        sub=pd.DataFrame(sub)
    plt.rcParams.update({'font.size': fontsize})
    if ax=='':
        fig,ax1=plt.subplots(figsize=figsize)
    else:
        ax1=ax
    color='blue'
    if xlabel=='':
        ax1.set_xlabel('X-axis',fontsize=fontsize) 
    else:
        ax1.set_xlabel(xlabel,fontsize=fontsize)
    if ylabel=='':
        ax1.set_ylabel('Y-axis', color = color,fontsize=fontsize) 
    else:
        ax1.set_ylabel(ylabel,color=color,fontsize=fontsize)
    ax1.tick_params(axis ='y', labelcolor = color) 
    if targetlist!=[]:
        for i in targetlist:
            ax1.plot(sub[i],label=i)
    else:
        if target1!='':
            ax1.plot(sub.index,sub[target1],label=target1,linewidth=linewidth)
        if target2!='':
            ax1.plot(sub.index,sub[target2],label=target2,linewidth=linewidth)
        if (target1=='')&(target2==''):
            for i in sub.columns:
                if i!=twin:
                    ax1.plot(sub[i],label=i)
    if title!='':
        ax1.set_title(title)
    if annotate!='':
        ax1.annotate(annotate,xy = annotate_pos,xycoords='axes fraction',ha='right',va="center",fontsize=10)
    if grid==True:
        plt.grid()

    if twin!='':
        ax2=ax1.twinx()
        color='green'
        ax2.set_ylabel(twin,color=color,fontsize=fontsize)
        ax2.plot(sub.index,sub[twin],label=twin,color='red',linewidth=linewidth)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc=2)
        ax2.tick_params(axis='y',labelcolor=color)
    else:
        ax1.legend()
    if savefig!=False:
        plt.savefig(savefig+'.jpg')
    if num_xticks!='':
        loc = pltticker.MultipleLocator(base=num_xticks) # this locator puts ticks at regular intervals
        ax1.xaxis.set_major_locator(loc)
def get_ts_analysis(sub,**kwargs):
    plot_line(sub,**kwargs)
    if len(sub.columns)==2:
        regression(sub)
def regression(sub1,target1='',target2='',summary=True,quantile=1,figsize=(10,6.18),winsorize_window=0,quantile_summary=False,params=True,pvalue=True,s=5,mute=False):
    sub_all=sub1.copy().replace([np.inf,-np.inf],np.nan).dropna()
    sub=sub1.copy().replace([np.inf,-np.inf],np.nan).dropna()
    if sub_all.empty:
        return 0
    if (target1=='')&(target2==''):
        target1=sub_all.columns[0]
        target2=sub_all.columns[1]
    if winsorize_window!=0:
        sub_all[target1]=pd.Series(scipy.stats.mstats.winsorize(sub_all[target1],limits=[winsorize_window,winsorize_window]))
    quantile_width=1/quantile
    if mute!=True:
        fig,ax1=plt.subplots(figsize=figsize)
        plt.figure(figsize=(10,6.18))
    for i in range(quantile):
        upperquantile=(1+i)*quantile_width
        lowerquantile=i*quantile_width
        if quantile!=1:
            sub=sub_all.loc[(sub_all[target1]<=np.quantile(sub_all[target1],upperquantile))&(sub_all[target1]>=np.quantile(sub_all[target1],lowerquantile))]
        X=sub[target1]
        X=sm.add_constant(X)
        Y=sub[target2]
        res=sm.OLS(Y,X).fit()
        if quantile_summary:
            print(str(i)+"th quantile")
            if summary:
                display(res.summary())
        if mute!=True:
            if not isinstance(target1,list):
                ax1.scatter(sub[target1],sub[target2],s=s)
                ax1.plot(X[target1],list(res.params)[0]+list(res.params)[1]*X[target1],linestyle='dashed',label='regression line'+str(i))
                ax1.legend()
                #plt.xlim(X.min(),X.max())
            else:
                for i in range(len(target1)):
                    ax1.scatter(sub[target1[i]],sub[target2],s=s)
                    ax1.plot(X[target1[i]],list(res.params)[0]+list(res.params)[i+1]*X[target1[i]],linestyle='dashed',label='regression line'+target1[i])
                    ax1.legend()
    if summary:
        display(res.summary())
    if not mute:
        if params:
            print('params: '+str(list(res.params)[0])+'   '+str(list(res.params)[1]))
        if pvalue:
            print('p-value: '+str(list(res.pvalues)[0])+'   '+str(list(res.pvalues)[1]))
        ax1.set_xlabel(target1)
        ax1.set_ylabel(target2)
        plt.show()
    else:
        return res
    return res
def get_couplemajor(target):
    return get_major(target).copy(),get_major(dic[target]).copy()
def VWAP(sub,weights):
    return (sub*(weights/weights.sum())).sum()
def init_date(data):
    data['Dates']=data['Dates'].apply(lambda x:pd.to_datetime(x,format="%Y-%m-%d %H:%M:%S"))
    data['Year']=data['Dates'].apply(lambda x:x.year)
    data['Month']=data['Dates'].apply(lambda x:x.month)
    data['Day']=data['Dates'].apply(lambda x:x.day)
    return data
def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

def get_sheetnames(name):
    data_path=os.get_cwd()+'/Data/'
    try:
        xis=xlrd.open_workbook(data_path+name+'.xlsx',on_demand=True)
    except:
        xis=xlrd.open_workbook(data_path+name+'.xls',on_demand=True)
    return xis.sheet_names()

def add_ym(a):
    if a.index.name=='Date':
        a.reset_index(inplace=True)
        a['Year']=a['Date'].apply(lambda x:x.year)
        a['Month']=a['Date'].apply(lambda x:x.month)
    else:
        a['Year']=a['Date'].apply(lambda x:x.year)
        a['Month']=a['Date'].apply(lambda x:x.month)
    return a.set_index('Date') 
    return a.drop(columns={'Year','Date'})
def gen_port(df):
    ret=df.apply(lambda x:x/x.shift(1)-1).drop(columns={'US CPI SA','China CPI','Global Reits','Global inflation','Crypto','BTC Bloomberg Galaxy','Bitcoin'})
    ret=ret[1:]
    port1=ret
    port2=pd.DataFrame()
    port2['mean']=(port1.mean()*12)
    port2['std']=(port1.std()*(12**0.5))
    port2=port2.T
    return port1,port2
def concat_dic(dic,operation=''):
    total=pd.DataFrame()
    for i in dic.keys():
        if operation=='':
            total=pd.concat([total,dic[i].rename(i)],axis=1)
        else:
            total=pd.concat([total,operation(dic[i]).rename(i)],axis=1)
    return total.T
def write(alpha,name):
    alpha.to_excel(name)
def get_sheetnames(name):
    path=os.getcwd()
    xis=xlrd.open_workbook(path+'/Updated Data/'+name+'.xlsx',on_demand=True)
    return xis.sheet_names()
def get_sheet(name1,sheetname='Sheet1',index_col=''):
    path=os.getcwd()
    if index_col=='':
        try:
            xls=pd.ExcelFile(path+'/Data/'+name1+'.xlsx')
            sub=pd.read_excel(xls,sheetname)
        except:
            xls=pd.ExcelFile(path+'/Data/'+name1+'.xls')
            sub=pd.read_excel(xls,sheetname)
    else:
        try:
            xls=pd.ExcelFile(path+'/Data/'+name1+'.xlsx')
            sub=pd.read_excel(xls,sheetname,index_col=index_col)
        except:
            xls=pd.ExcelFile(path+'/Data/'+name1+'.xls')
            sub=pd.read_excel(xls,sheetname,index_col=index_col)
    return sub
def read_fromproject(name):
    return pd.read_excel('/Users/will/Desktop/Master/RA/Tasks/Task-21-05-20/Source Data/'+name)

def gen_corr(df):
    from scipy.stats import pearsonr
    def pearsonr_pval(x,y):
        return pearsonr(x,y)[0]
    return df.corr().round(5).astype(str)+df.corr(pearsonr_pval).applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t])).T




    
def gen_latex(display_adf_df):
    import latextable
    from tabulate import tabulate
    from texttable import Texttable
    table = Texttable()
    table.set_cols_align(["c"] * display_adf_df.shape[1])
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(display_adf_df.values)
    print(tabulate(display_adf_df, headers='keys', tablefmt='latex'))




#############################    result analysis  ###################################
def gen_profit(df,upperthres=0.1,lowerthres=0.1,how='quantile',rolling_window=20):
    '''
    columns required: pred, real,'Date'
    '''
    import scipy.stats
    if how=='quantile':
        df['position']=df.groupby('Date')['pred'].rank(pct=True)
        df['position']=df['position'].apply(lambda x:1 if x>1-upperthres else -1 if x<lowerthres else 0)
        df['position']=df.groupby('Date')['position'].apply(lambda x:x/np.sum(abs(x)))
    elif how=='normalized_prob':
        df['position']=df.groupby('Date')['pred'].apply(lambda x:(x-x.mean())/x.std())
        upperthres_prob,lowerthres_prob=scipy.stats.norm(0,1).ppf(1-upperthres),scipy.stats.norm(0,1).ppf(lowerthres)
        df['position']=df['position'].apply(lambda x:1 if x>upperthres_prob else -1 if x<lowerthres_prob else 0)
        df['position']=df.groupby('Date')['position'].apply(lambda x:x/abs(x).sum())
    elif how=='rolling_normalized_prob' :
        df['position']=df.groupby('Code')['pred'].rolling(rolling_window,min_periods=3).apply(lambda x:(x.iloc[-1]-x.mean())/x.std()).droplevel(0)
        upperthres_prob,lowerthres_prob=scipy.stats.norm(0,1).ppf(1-upperthres),scipy.stats.norm(0,1).ppf(lowerthres)
        df['position']=df['position'].apply(lambda x:1 if x>upperthres_prob else -1 if x<lowerthres_prob else 0)
        df['position']=df.groupby('Date')['position'].apply(lambda x:x/abs(x).sum())
    df['profit']=df['position']*df['real']
    return df
def add_transcost(x,transcost=0.0005):
    x['transcost']=abs(x['position'].diff())*transcost
    x['after_trans_profit']=x['profit']-x['transcost']

    return x
def eval_result(result,method='',bin_num=5):
    from sklearn.metrics import r2_score
    from scipy.stats import shapiro
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from statsmodels.stats.diagnostic import acorr_ljungbox
    result['pred_bin']=result.groupby('Date')['pred'].apply(lambda x:pd.qcut(x,q=bin_num,duplicates='drop',labels=False))
    resid=result.iloc[:,1]-result.iloc[:,0]
    corr=result.corr().iloc[0,1]
    rmse=sqrt(mean_squared_error(result.iloc[:,0],result.iloc[:,1]))
    mae=np.sum(abs(result.dropna().iloc[:,0]-result.dropna().iloc[:,1]))/len(result.dropna())
    adjusted_mae=mae/result.dropna().iloc[:,1].std()
    r2=r2_score(result.iloc[:,1],result.iloc[:,0])
    annual_profit=result['after_trans_profit'].cumsum().iloc[-1]/len(result['Date'].drop_duplicates())*255
    annual_profit_beforetrans=result['profit'].cumsum().iloc[-1]/len(result['Date'].drop_duplicates())*255
    sharpe=annual_profit/result['after_trans_profit'].replace(0,np.nan).dropna().std()/(255**0.5)
    ljung_1=acorr_ljungbox(resid,1,return_df=True).T.iloc[1,0]
    winrate=result['after_trans_profit'].replace(0,np.nan).dropna().apply(lambda x:1 if x>0 else 0).mean()
    ret_pertrade=result['after_trans_profit'].cumsum().iloc[-1]/result['after_trans_profit'].apply(lambda x:1 if abs(x)>0 else 0).replace(0,np.nan).dropna().count()
    shapiro_p=shapiro(resid.dropna()).pvalue
    # num_trading_days=len(result['Date'].drop_duplicates())/(result['Date'].max()-result['Date'].min()).days*255
    col_index=['Annualized Return_afterfee','Annualized Return_beforefee','Sharpe Ratio','winrate','return per trade','correlation','mae','adjusted mae','rmse','ljung-box test(1)-pvalue','Shapiro-Wilk Test pvalue','R-squared']
    eval_df=pd.DataFrame([annual_profit,annual_profit_beforetrans,sharpe,winrate,ret_pertrade,corr,mae,adjusted_mae,rmse,ljung_1,shapiro_p,r2],index=col_index,columns=['Parameter '+method])
    return eval_df
def result_analysis(result,transcost=0.0003,position_method='quantile',upperthres=0.1,lowerthres=0.1):
    '''
    columns required: pred, real,Date
    transcost is one way
    position_method: 'quantile', 'normalized_prob' 
    '''
    if position_method=='all':
        position_method=['quantile','normalized_prob','rolling_normalized_prob']
    if type(position_method)==list:
        eval_df_total=pd.DataFrame()
        for method in position_method:
            result=add_transcost(gen_profit(result,how=method,upperthres=upperthres,lowerthres=lowerthres),transcost=transcost)
            eval_df=eval_result(result,method)
            eval_df_total['Parameter '+method]=eval_df['Parameter '+method]
    else:
        result=add_transcost(gen_profit(result,how=position_method))
        eval_df_total=eval_result(result)
    return eval_df_total,result
def init_subplots(row=2,col=4,figsize=(40,40)):
    global fig,ax
    fig,ax=plt.subplots(row,col,figsize=figsize,squeeze=False)
    return fig,ax
def get_subplots_ax(ax,num):
    '''
    Get the axis to plot on for a plt subplots object given a specific ordinal number

    Args: 
        ax: axis to plot on
        num: the ordinal number of the graph to plot on, starting from 1

    Returns: 
        required axis

    '''
    try:
        row,col=ax.shape[0],ax.shape[1]
    except:
        row,col=ax.size/ax.shape[0],ax.shape[0]
    row_count=0
    if num>row*col:
        print('number exceeding maximum number')
        return 
    while num>=col:
        row_count+=1
        num-=col
    return ax[row_count,num]

def grouped_analysis(result,display=True,ax='',title='',num_xticks=20,fontsize=15,bin_num=5):
    if 'pred_bin' not in list(result.columns):
        result['pred_bin']=result.groupby('Date')['pred'].apply(lambda x:pd.qcut(x,q=bin_num,duplicates='drop',labels=False))
    grouped_result=result.groupby(['Date','pred_bin'])['real'].mean().unstack(1).cumsum()
    if display:
        plot_line(grouped_result,figsize=(20,12),ax=ax,title=title+' grouped cumulative return_no transaction cost',num_xticks=num_xticks,fontsize=fontsize) 
def grouped_analysis_withcost(result,display=True,ax='',title='',num_xticks=20,transcost=0.0005,relative=False,fontsize=15,bin_nuim=5):
    '''
    transcost: double side transcost, buy and sell
    '''
    if 'pred_bin' not in list(result.columns):
        result['pred_bin']=result.groupby('Date')['pred'].apply(lambda x:pd.qcut(x,q=bin_num,duplicates='drop',labels=False))
    if relative:
        grouped_result=(result.groupby(['Date','pred_bin'])['real'].mean().unstack(1)-transcost).cumsum()-result.groupby('Date')['real'].mean()
    else:
        grouped_result=(result.groupby(['Date','pred_bin'])['real'].mean().unstack(1)-transcost).cumsum()
    if display:
        plot_line(grouped_result,figsize=(20,12),ax=ax,title=title+' grouped cumulative return_with transaction cost',num_xticks=num_xticks,fontsize=fontsize) 
def plot_result_analysis(result,figsize=(20,12),title='',num_xticks=10):
    from ml_utils import bivariate_plot
    fig,ax=init_subplots(1,2,figsize=figsize)
    grouped_analysis(result,ax=get_subplots_ax(ax,0),title=title,num_xticks=num_xticks)
    bivariate_plot(result,'pred','real',ax=get_subplots_ax(ax,1),scatter=False,title=title)
    plt.show()



################################# Functions for pairs trading #######################################

def find_cointegrated_pairs(data,pvalue_thres):
    '''
    find cointegration for each pair of columns in the dataframe
    Args: 
        data: DataFrame of columns of data for inspection
        p_value_thres: float in (0,1) to decide the pvalue treshold for cointegration pair filtration
    '''
    from statsmodels.tsa.stattools import coint, adfuller
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    pvalue_list=[]
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[[keys[i],keys[j]]].dropna()[keys[i]]
            S2 = data[[keys[i],keys[j]]].dropna()[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < pvalue_thres:
                pairs.append((keys[i], keys[j]))
                pvalue_list.append(pvalue)
    return score_matrix, pvalue_matrix, pd.DataFrame([pairs,pvalue_list],index=['pairs','pvalue']).T.sort_values('pvalue',ascending=True)
def display_cointegrated_pairs(df,pvalue=0.02):
    '''
    Args: 
        df: DataFrame of columns of data for cointegration inspection
        p_value: float in (0,1) to decide the pvalue treshold for cointegration pair filtration

    Returns: 
        pairs: pairs with pvalue less then the required pvalue threshold
        p-value dataframe: dataframe of pvalue for each pair
        plt.plot: cointegration heatmap of pairs with cointegration pvalue lower then threshold
    '''
    scores, pvalues, pairs = find_cointegrated_pairs(df,pvalue)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(pvalues, xticklabels=list(df.columns), yticklabels=list(df.columns), cmap='RdYlGn_r' 
                    , mask = (pvalues >= pvalue),ax=ax)
    plt.show()
    return pairs,pd.DataFrame(pvalues,index=df.columns,columns=df.columns)
