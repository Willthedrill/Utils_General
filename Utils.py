import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
# print("LightGBM version: {}".format(lgb.__version__))
from tqdm import tqdm
from IPython import display
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

###################################     Plotting Functions     ###################################

def init_subplots(row=2,col=4,figsize=(40,40)):
    '''
    initiate subplots with specified rows, columns and figure size
    Arguments: 
        row(int): number of rows
        col(int): number of columns
        figsize: size of the required figure

    Returns: 
        fig, ax
    '''
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



def plot_scatter(sub,summary=False,s=10,title='',xlabel='',ylabel='',pvalue=False,res_return=False,show_regression=True,ax='',figsize=(10,6)):
    '''
    Args: 
        sub: DataFrame of two columns required for scatter. The first column should contain independent variables and the second column should contain dependent variable. 
        s: size of point
        ax: axis to plot in


    Return: 
        res(optional)
    '''
    if ax=='':
        fig,ax1=plt.subplots(figszie=figsize)
    else:
        ax1=ax

    if isinstance(sub,pd.Series):
        sub=pd.DataFrame(sub).reset_index()
    sub.dropna(inplace=True)
    X=sub[sub.columns[0]]
    Y=sub[sub.columns[1]]
    ax1.scatter(X,Y,s=s)
    ax1.set_xlabel(sub.columns[0])
    ax1.set_ylabel(sub.columns[1])
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
    ax1.legend()
    ax1.grid()
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

def autosave_fig(graphic_name):
    import os
    os.mkdirs('./graphics',exist_ok=True)
    plt.savefig('./graphics/'+graphic_name+'.png',bbox_inches='tight')


###################################     Regression Functions     ###################################

def regression(sub1,target1='',target2='',summary=True,quantile=1,figsize=(10,6.18),winsorize_window=0,quantile_summary=False,params=True,pvalue=True,s=5,mute=False):
    '''plot regression of twwo columns in a df
    Args: 
        sub1(df)
        target1(str/list): x variable
        target2(str/list): y variable
    Returns: 
        res: statsmodel regression summary object
    '''
    if (target1=='')&(target2==''):
        target1=sub_all.columns[0]
        target2=sub_all.columns[1]
    drop_na_list=[]
    if type(target1)==list:
        drop_na_list+=target1
    elif type(target1)==str:
        drop_na_list+=[target1]
    if type(target2)==list:
        drop_na_list+=target2
    elif type(target2)==str:
        drop_na_list+=[target2]
    sub_all=sub1.copy().replace([np.inf,-np.inf],np.nan).dropna()
    sub=sub1.copy().replace([np.inf,-np.inf],np.nan).dropna()
    if sub_all.empty:
        return 0
    
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

def log_regression(res,x_name='',y_name=''):
    '''given a statsmodel regression res object, record the key arguments into a pandas series'''
    param_df=pd.DataFrame(res.summary2().tables[1])
    param_df=param_df.iloc[1]
    return pd.Series(list(np.append([res.rsquared_adj,res.nobs],np.array(param_df.values)))+[x_name,y_name])

def regression_stat(target_df,x,y,prettify=True,standardize=''):
    '''display regression statistics of a dataframe
    Args: 
        standardize(list): list of columns to perform standardization
    '''
    if standardize!='':
        temp=target_df.copy()
        try:
            temp[standardize]=temp[standardize].apply(lambda x: rolling_standardize(x))
        except Exception as e:
            print("Error: "+e)
    else:
        temp=target_df
    reg=regression(temp,x,y,mute=True,summary=False)
    result=pd.read_html(reg.summary().tables[1].as_html(),header=0,index_col=0)[0]
    result=pd.concat([pd.DataFrame(result.iloc[0]).T,result.iloc[1:].sort_values('t')],axis=0)
    if prettify:
        return result.style.background_gradient(axis=0)
    else:
        return result

def regression_stat_multiple(df,study_variable_x,study_variable_y,scatter=False,prettify=True,*args, **kwargs):
    '''generate multiple bivariate plot figures
    Args: 
        study_variable_x(str/list): variables to be put on x axis in regression
        study_variable_y(str/list): variables to be put on y axis in regression
        prettify(bool): whether to return the dataframe or the decorated version of dataframe, default to be True
        kwargs(dic): parameters to be put in regression function
        
    Returns: 
        figures will be displayed with no return variable
    '''
    x_islist,y_islist=(type(study_variable_x)==list),(type(study_variable_y)==list)
    if not x_islist:
        study_variable_x=[study_variable_x]        
    if not y_islist:
        study_variable_y=[study_variable_y]
    df_total=pd.DataFrame()
    for sub_study_variable_x in study_variable_x:
        for sub_study_variable_y in study_variable_y:
            res=regression(df,sub_study_variable_x,sub_study_variable_y,mute=True,summary=False,**kwargs)
            df_total=pd.concat([df_total,log_regression(res,sub_study_variable_x,sub_study_variable_y)],ignore_index=True,axis=1)
    param_df=pd.DataFrame(res.summary2().tables[1])
    param_df=param_df.iloc[1]
    df_total.index=(['rsquared_adj','nobs']+list(param_df.index)+['x_variable','y_variable'])
    if prettify:
        return df_total.T.set_index(['x_variable','y_variable']).apply(lambda x:x.astype(float)).style.background_gradient()
    else:
        return df_total.T.set_index(['x_variable','y_variable']).apply(lambda x:x.astype(float))

####################################     DataFrame Operations     #####################################################

def winsorize_variable(df,target,limits=[0.01,0.01]):
    '''winsorize variable(s) in a dataframe by the specified limit boundaries'''
    if type(target)==list:
        df[target]=df[target].apply(lambda x:np.array(winsorize(x,limits=limits)))
    elif type(target)==str:
        df[target]=np.array(winsorize(df[target],limits=limits))
    return df

def try_drop(df,col):
    '''try to drop a column or a list of columns in df if it exists
    Args: 
        df
        col(str/list): columns to drop
    Returns: 
        df  
    '''
    if type(col)==str:
        col=[col]
    for sub_col in col:
        if sub_col in df.columns:
            try:
                df.drop(columns={sub_col},inplace=True)
            except Exception as e:
                print('Error: '+e)
    return df

def concat_dic(dic,operation=''):
    '''concat dictionary of pandas series into a dataframe
    Args: 
        dic(dict): dictionary of pandas series
        operation(func): operation to be applied on each series before integrating into dataframe
    Returns: 
        df
    '''
    total=pd.DataFrame()
    for i in dic.keys():
        if operation=='':
            total=pd.concat([total,dic[i].rename(i)],axis=1)
        else:
            total=pd.concat([total,operation(dic[i]).rename(i)],axis=1)
    return total.T


###################################     File I/O     ###################################

def get_sheetnames(name,data_path=''):
    '''get the sheet name of excel under datapath.
    Args: 
        data_path(str): if not specified, will be default to be set to the "Data" folder under the same working directory
    Returns: 
        list of sheet names
    '''
    data_path=os.get_cwd()+'/Data/'
    try:
        xis=xlrd.open_workbook(data_path+name+'.xlsx',on_demand=True)
    except:
        xis=xlrd.open_workbook(data_path+name+'.xls',on_demand=True)
    return xis.sheet_names()


###################################     Data Analysis     ###################################


def gen_corr(df):
    '''generate pearson correlation table of a dataframe with significance label "*"'''
    from scipy.stats import pearsonr
    def pearsonr_pval(x,y):
        return pearsonr(x,y)[0]
    return df.corr().round(5).astype(str)+df.corr(pearsonr_pval).applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t])).T
    
def gen_latex(display_adf_df):
    # import latextable
    from tabulate import tabulate
    from texttable import Texttable
    table = Texttable()
    table.set_cols_align(["c"] * display_adf_df.shape[1])
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(display_adf_df.values)
    print(tabulate(display_adf_df, headers='keys', tablefmt='latex'))




#############################    Result analysis  ###################################
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

def grouped_analysis(result,display=True,ax='',title='',num_xticks=20,fontsize=15,bin_num=5):
    if 'pred_bin' not in list(result.columns):
        result['pred_bin']=result.groupby('Date')['pred'].apply(lambda x:pd.qcut(x,q=bin_num,duplicates='drop',labels=False))
    grouped_result=result.groupby(['Date','pred_bin'])['real'].mean().unstack(1).cumsum()
    if display:
        plot_line(grouped_result,figsize=(20,12),ax=ax,title=title+' grouped cumulative return_no transaction cost',num_xticks=num_xticks,fontsize=fontsize) 

def grouped_analysis_withcost(result,display=True,ax='',title='',num_xticks=20,transcost=0.0005,relative=False,fontsize=15,bin_num=5):
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


###################################     Pairs Trading Functions     ###################################

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




###################################     EDA Functions   ###################################
def match_col(df1,df2,left_col,right_col,target_col,on=[],rename='',drop=False):
    '''replace left_col in df1 with values from target_col in df2 where it's corresponding right_col in the same row is the same as that in df1 left_col, with identicle values in the on columns
    Args: 
        left_col,right_col(str): columns with the same value to match from df1 and df2
        tagret_col(str): value to lookup from df2
        on(list): mutual columns that must contain same value for the match
        rename(str): rename the new target col
        drop(bool): whether to drop the original left_col in df1
    Returns:
        df1: dataframe with the merged column
    '''
    if rename=='':
        rename=target_col
    df1=try_drop(df1,rename)
    df1=pd.merge(df1,df2[on+[right_col,target_col]].rename(columns={right_col:left_col,target_col:rename}),on=on+[left_col],how='left')
    if drop:
        df1.drop(columns={left_col},inplace=True)
    return df1






def bivariate_plot_multiple(df,study_variable_x,study_variable_y,scatter=False,figsize=False,row='',col='',save='auto'):
    '''generate multiple bivariate plot figures
    Args: 
        study_variable_x(str/list): variables to be put on x axis in regression
        study_variable_y(str/list): variables to be put on y axis in regression

    Returns: 
        figure of multiple bivariate plot
    '''
    from ml_utils import bivariate_plot
    x_islist,y_islist=(type(study_variable_x)==list),(type(study_variable_y)==list)
    
    num=max(len(study_variable_x)*(type(study_variable_x)==list),1)*max(len(study_variable_y)*(type(study_variable_y)==list),1)
    if x_islist or y_islist:
        if (row=='') & (col==''):
            col=3
            import math
            row=math.ceil(num/col)
        else:
            row,col=1,1
        if figsize==False:
            figsize=(30,row*5+1)
    fig,axes=init_subplots(row,col,figsize=figsize)
    if not x_islist:
        study_variable_x=[study_variable_x]        
    if not y_islist:
        study_variable_y=[study_variable_y]
    count=0
    for sub_study_variable_x in study_variable_x:
        for sub_study_variable_y in study_variable_y:
            bivariate_plot(df,sub_study_variable_x,sub_study_variable_y,scatter=scatter,ax=get_subplots_ax(axes,count))
            count+=1
    plt.tight_layout(pad=0.3)
    if save!='':
        if save=='auto':
            graphic_name=''
            if x_islist:
                graphic_name=graphic_name+study_variable_x[0]
            else:
                graphic_name=graphic_name+study_variable_x
            graphic_name+='_'
            if y_islist:
                graphic_name=graphic_name+study_variable_y[0]
            else:
                graphic_anme=graphic_name+study_variable_y
            autosave_fig(graphic_name)
        else:
            plt.savefig(save)
    plt.show()

def rolling_normalize(s,cap=5):
    '''rolling normalize a pandas series. this rolling normalization calculates the percentile and use inverse cdf to get normalized value
    Args: 
        cap(float): maximum value for the series to take when it reaches maximum
    '''
    from scipy.stats import percentileofscore,norm
    s=s.rolling(len(s),min_periods=1).apply(lambda x:norm.ppf(percentileofscore(x.values,x.values[-1],kind='weak')/100))
    s=s.replace(np.inf,cap)
    return s

def rolling_standardize(s,rolling_length=False,method='normal'):
    '''rolling standardize a pandas series
    Args: 
        s(pd.Series)
        rolling_length(int): length of the rolling window
        method(str): 'normal' or 'minmax', indicating the method of standardization
    '''
    
    if (rolling_length==False) or (rolling_length==''):
        rolling_length=len(s)-1
    if method=='normal':
        return (s-s.rolling(rolling_length,min_periods=1).mean())/s.rolling(rolling_length,min_periods=1).std()
    elif method=='min_max':
        return (s-s.rolling(rolling_length,min_periods=1),min())/(s.rolling(rolling_length,min_periods=1),max()-s.rolling(rolling_length,min_periods=1).min())
    elif method=='normal_percetile':
        from scipy.stats import percentileofscore,norm
        return s.roilling(rolling_length,min_periods=1).apply(lambda x:norm.ppf(percentileofscore(x.dropna().values,x.iloc[-1]/100)))

###################################     EDA Functions   ###################################
class backtest:
    '''class for convenient backtesting
    Args: 
        target_df(df)
        signal_col(str/list): name/list of names of the columns as signals
        standardize_method(str): method for signal generation: "normal"/"min_max"/"none"
        position_method(str): method for position generation. "quantile"/"normal"
        threshold(float/tuple_like object): threshold for long/short posiution setting if position_method=='normal'
        quantile(int): number of quantile to use in quantile position generation method
        transaction_cost(float): two-way transaction cost to impose on each trade
        date_col(str): name of the date column, default 'trade_date'
        ret_col(str): name of the return column for each position, default 'return'
        hold_period(int): holding period of the position, e.g. if hold_period is 2, then each day only half of the position is invested so that the invested amounts are the same for each day
        max_position(list): max_position for long and short. e.g. [1,1]
        max_stocks(int): maximum stocks to trade per day. If number of stocks exceeds the limit, select only the top max_number highest signal stocks
        trading_strategy(str): (default 'full_even') how to trade the total AUM
            'full_even': everyday invest full AUM evenly across selected stocks
            'full_weighted':everyday invest full AUM across selected stocks weighted by signal(developer note: note the way to deal with positive and negative signal in the development of this function)
            'fixed_even': everyday invest partial AUM across selected stocks(requiring max_stocks to be specified). e.g. max_stocks=100, then everyday each stock can get maximum 1% of AUM
    '''
    def __init__(self,target_df,signal_col='signal',standardize_method='normal',position_method='quantile',quantile=10,threshold=0.05,transcost=0.03,date_col='trade_date',ret_col='return',hold_period=1,max_position=[1,0],trading_strategy='full_even',max_stocks='',rolling_adjust_fix_even=False):
        self.target_df=target_df.copy()
        self.signal_col=signal_col
        self.method=standardize_method
        self.position_method=position_method
        self.quantile=quantile
        self.threshold=threshold
        self.transcost=transcost
        self.date_col=date_col
        self.ret_col=ret_col
        self.hold_period=hold_period
        self.max_position=max_position
        self.act_time_col='act_pubtime'
        self.long_max_position,self.short_max_position=self.max_position[0]/self.hold_period,self.max_position[1]/self.hold_period
        self.trading_strategy=trading_strategy
        self.max_stocks=max_stocks
        if self.trading_strategy=='fixed_even':
            if (self.max_stocks==''):
                self.max_stocks=100
                print('trading strategy is fixed_even but max_stocks not specified, set to default value 100')
        self.rolling_adjust_fix_even=rolling_adjust_fix_even
        
        def generate_threshold_normal(self):
            '''generate upperthres and lower thres for the standardized signal according to the standardization method and threshold given'''
            if type(self.threshold)==float:
                if self.method=='normal':
                    from scipy.stats import norm
                    upperthres,lowerthres=norm.ppf(1-self.threshold),norm.ppf(self.threshold)
                elif self.method=='min_max':
                    upperthres,lowerthres=1-self.threshold,self.threshold   
            elif ((type(self.threshold)==list) or (type(self.threshold)==tuple)) and (len(self.threshold)==2):
                if self.method=='normal':
                    upperthres,lowerthres=norm.ppf(1-self.threshold[0]),norm.ppf(self.threshold[1])
                elif self.method=='min_max':
                    upperthres,lowerthres=1-self.threshold[0],self.threshold[1]
            return upperthres,lowerthres

        def generate_position(self,kwargs_rolling_standardize={}):
            #long_max_position and short_max_position are the maximum proportion of AUM that can be longed or shorted
            #e.g. max_position=[1,0.5], meaning that at any time thee maximum long position and maximum short position are 100% and 50% of AUM. If hold_period=2, for each day's new trade there can only be 50% AUM longed and 25% AUM shorted
            '''generate position for signals
            Args: 
                kwargs_rolling_standarzdize(dict): dictionary of parameters to be passed on to rolling standardize function applied on signals
            '''
            try: 
                self.target_df.sort_values(self.act_time_col,inplace=True)
            except:
                pass

            if type(self.signal_col)==str:
                col_name='position_signal'
                self.target_df=try_drop(self.target_df,col_name)
                # self.target_df[col_name]=rolling_standardize(self.target_df[self.signal_col,**kwargs_rolling_standardize])
                self.target_df[col_name]=rolling_standardize(self.target_df[self.signal_col])
                if self.position_method=='normal':
                    upperthres,lowerthres=self.generate_threshold_normal(self)
                    self.target_df['position']=self.target_df[col_name].apply(lambda x:x if x>upperthres else x if x<lowerthres else 0)
                    self.target_df['position']=self.target_df.groupby(self.date_col)['position'].apply(lambda x:x.apply(lambda y:y/abs(x.loc[x>0]).sum()*self.long_max_position if y>0 else y/abs(x.loc[x<0]).sum()*self.short_max_position if y<0 else 0))
                elif self.position_method=='quantile':
                    self.target_df['quantile']=0
                    for quantile in tqdm(range(1,self.quantile+1)):
                        position_col='position_quantile_'+str(quantile)
                        self.target_df['upper']=self.target_df[col_name].rolling(len(self.target_df),1).quantile(quantile/self.quantile)
                        self.target_df['lower']=self.target_df[col_name].rolling(len(self.target_df),1).quantile((quantile-1)/self.quantile)
                        self.target_df['quantile']=np.where((self.target_df[col_name]==self.target_df['upper'])*(self.target_df[col_name]>=self.target_df['lower']),np.array([quantile for i in range(len(self.target_df))]),self.target_df['quantile'].values)
                        self.target_df=try_drop(self.target_df,['upper','lower'])
                        self.target_df['quantile']=self.target_df['quantile'].replace(0,np.nan)
                        self.target_df[position_col]=np.where(self.target_df['quantile']==quantile,self.target_df[self.signal_col],0)
                        #limit for max_stocks traded
                        if self.trading_strategy=='full_even':
                            self.target_df[position_col]=np.where(self.target_df['qauntile']==quantile,1,0)
                            if self.max_stocks!='':
                                thres=self.target_df.groupby(self.date_col)[position_col].apply(lambda x:abs(x).nlargest(self.max_stocks).iloc[-1])
                                self.target_df=match_col(self.target_df,thres.rename('thres').reset_index(),left_col=self.date_col,right_col=self.date_col,target_col='thres')
                                self.target_df[position_col]=np.where(abs(self.target_df[position_col]>self.target_df['thres'],self.target_df[position_col].values,0))
                            self.target_df[position_col]=self.target_df.groupby(self.date_col)[position_col].apply(lambda x:x/x.sum()*self.long_max_position)
                        elif self.trading_strategy=='fixed_even':
                            thres=self.target_df.groupby(self.date_col)[position_col].apply(lambda x:abs(x).nlargest(self.max_stocks).iloc[-1])
                            self.target_df=match_col(self.target_df,thres.rename('thres').reset_index(),left_col=self.date_col,right_col=self.date_col,target_col='thres')
                            self.target_df[position_col]=np.where(abs(self.target_df[position_col])>self.target_df['thres'],self.long_max_position*self.hold_period/self.max_stocks,0)
                            #ADJUSTING OPEN POSITION METHOD
                            if self.rolling_adjust_fix_even:
                                target_array=self.target_df[position_col].values
                                for i in range(1,len(self.target_df)):
                                    prev_position=target_array[i-1]
                                    current_position=target_array[i-1]
                                    if (prev_position+current_position)>1:
                                        target_array[i]=1-prev_position
                                self.target_df[position_col]=target_array
                        self.target_df=try_drop(self.target_df,'thres')
                    #by default if position method is quantile then the default position corresponds to the position of quantile 10 or the maximum quantile
                    self.target_df['position']=self.target_df['position_quantile_'+str(self.quantile)]
            elif type(self.signal_col)==list:
                #further development needed
                for signal_col_name in self.signal_col:
                    col_name='position_signal_'+signal_col_name
                    self.target_df=try_drop(self.target_df,col_name)
                    self.target_df[col_name]=rolling_standardize(self.target_df[self.signal_col],**kwargs_rolling_standardize)
                    self.target_df['position_'+signal_col_name]=self.target_df[col_name].apply(lambda x:x if x>upperthres else x if x<lowerthres else 0)
                    self.target_df['position_'+signal_col_name]=self.target_df.groupby(self.date_col)['position_'+signal_col_name].apply(lambda x:x.apply(lambda y:y/abs(x.loc[x>0]).sum()*self.long_max_position if y>0 else y/abs(x.loc[x<0]).sum()*self.short_max_position if y<0 else 0))
                
        def calculate_profit(self,add_transcost=True,drop_position_col=True):
            '''calculate profit for the strategy
            Args: 
                add_transcost(bool): whether to add transaction cost to the profit
                drop_position_col(boll): whether to drop quantile position column after calculation of profit

            '''
            if type(self.signal_col)==str:
                if self.position_method=='normal':
                    self.target_df['profit']=self.target_df['position']*self.target_df[self.ret_col]
                    if add_transcost:
                        self.target_df['profit']-=abs(self.target_df['position'])*self.transcost
                
                elif self.position_method=='quantile':
                    for quantile in range(1,self.quantile+1):
                        try:
                            self.target_df['profit_quantile_'+str(quantile)]=self.target_df['position_quantile_'+str(quantile)]*self.target_df[self.ret_col]
                            if add_transcost:
                                self.target_df['profit_quantile_'+str(quantile)]-=self.target_df['position_quantile_'+str(quantile)]*self.transcost
                            if drop_position_col:
                                self.target_df=try_drop(self.target_df,'position_quantile_'+str(quantile))
                        except Exception as e:
                            print('Error: '+str(e))
                    self.target_df['profit']=self.target_df['profit_quantile_'+str(self.quantile)]
            elif type(self.signal_col)==list:
                for signal_col_name in self.signasl_col:
                    self.target_df['profit_'+signal_col_name]=self.target_df['position_'+signal_col_name]*self.target_df[self.ret_col]
                    if add_transcost:
                        self.target_df['profit_'+signal_col_name]-=abs(self.target_df['position_'+signal_col_name])*self.transcost
        
        def refresh(self,kwargs_generate_position={},kwargs_calculate_profit={}):
            self.generate_position(**kwargs_generate_position)
            self.calculate_profit(**kwargs_calculate_profit)
        
        def analyze_long_short_returns(self):
            if type(self.signal_col)==str:
                self.analyze_df=self.target_df[[self.date_col,self.ret_col,'position','profit']]
                # self.analyze_df['long_profit']=self.analyze_df['profit']-self.analyze_df['position']
            elif type(self.signal_col)==list:
                self.analyze_df=self.target_df[[self.date_col,self.ret_col]+['position_'+signal_col_name for signal_col_name in self.signal_col]+['profit_'+signal_col_name for signal_col_name in self.signal_col]]
        
        def analyze_capital_utilization_rate(self,target_col_list,return_df=True):
            '''analyze the capital utilization rate of specified list of position columns. if the specified position column is dropped when calculating profit, this function will refresh the dataframe without droppipng position column'''
            if type(target_col_list)==str:
                target_col_list=[target_col_list]
            for i in target_col_list:
                if i not in self.target_df.columns:
                    self.refresh(kwargs_calculate_profit={'drop_position_col':False})
                    break
            plt.figure(figsize=(10,6))
            plt.title('capital utilization rate distribution')
            temp=self.target_df.groupby(self.date_col)[target_col_list].sum().replace(0,np.nan)
            for i in temp.columns:
                temp[i].sort_values().reset_index(drop=True).plot(label=i)
            plt.legend()
            plt.show()
            display(temp.mean().reset_index())
            if return_df:
                return temp

        def analyze_yearly_return(self,return_col_list=[],plot_time=False,figsize=(10,6),title='cumulative return',start_year=2016):
            '''analyze the yearly return of a return column
            Args: 
                return _col_list(list)
                plot_time(bool): Whether to plot time in cumulative return plot. Setting it to false would make the graph look better when you have return series that are not continuous in time. 
            '''
            
            if return_col_list==[]:
                if self.position_method=='quantile':
                    return_col_list=['profit_quantile_'+str(i) for i in range(1,self.quantile+1)]
                else:
                    return_col_list=['profit']
            #plot cumulative return
            if type(return_col_list)==str:
                return_col_list=[return_col_list]
            plt.figure(figsize=figsize)
            plt.title(title)
            if not plot_time:
                series_cumret=self.target_df.loc[(self.target_df[self.date_col].dt.year>=start_year)].sort_values(self.date_col).replace(np.nan,0)[return_col_list].reset_index(drop=True)
            else:
                series_cumret=self.target_df.loc[(self.target_df[self.date_col].dt.year>=start_year)].replace(np.nan,0).set_index(self.date_col)[return_col_list]
            for i in series_cumret.columns:
                series_cumret[i].cumsum().plot(label=i)
            plt.legend()
            plt.show()
            print('yearly return details')
            yearly_earnings=self.target_df.loc[(self.target_df[self.date_col].dt.year>=start_year)].replace(np.nan,0).groupby( self.target_df.loc[(self.target_df[self.date_col].dt.year>=start_year)][self.date_col].st.year)[return_col_list].sum()
            display(yearly_earnings.style.background_gradient(axis=1))
            print('cumulative return detail')
            sharpe=self.target_df.replace(0,np.nan).loc[(self.target_df[self.date_col].dt.year>=start_year)].replace(0,np.nan).groupby(self.target_df.loc[(self.target_df[self.date_col].dt.year>=start_year)][self.date_col].dt.year)[return_col_list].apply(lambda x:x.mean()/(x.std())*(252**0.5))
            display(yearly_earnings.cumsum().iloc[-1].reset_index().set_index('index').T.style.background_gradient(axis=1))
            print('quantile return sharpe')
            display(pd.DataFrame(sharpe).style.background_gradient(axis=1))


###################################     Compress Dataset   ###################################
INT8_MIN    = np.iinfo(np.int8).min
INT8_MAX    = np.iinfo(np.int8).max
INT16_MIN   = np.iinfo(np.int16).min
INT16_MAX   = np.iinfo(np.int16).max
INT32_MIN   = np.iinfo(np.int32).min
INT32_MAX   = np.iinfo(np.int32).max

FLOAT16_MIN = np.finfo(np.float16).min
FLOAT16_MAX = np.finfo(np.float16).max
FLOAT32_MIN = np.finfo(np.float32).min
FLOAT32_MAX = np.finfo(np.float32).max

def memory_usage(data, detail=1):
    if detail:
        display(data.memory_usage())
    memory = data.memory_usage().sum() / (1024*1024)
    print("Memory usage : {0:.2f}MB".format(memory))
    return memory
                            
def compress_dataset(data):
    """
        Compress datatype as small as it can
        Parameters
        ----------
        path: pandas Dataframe

        Returns
        -------
            None
    """
    memory_before_compress = memory_usage(data, 0)
    print()
    length_interval      = 50
    length_float_decimal = 4

    print('='*length_interval)
    for col in tqdm(data.columns):
        col_dtype = data[col][:100].dtype

        if col_dtype != 'object':
            print("Name: {0:24s} Type: {1}".format(col, col_dtype))
            col_series = data[col]
            col_min = col_series.min()
            col_max = col_series.max()

            if col_dtype == 'float64':
                print(" variable min: {0:15s} max: {1:15s}".format(str(np.round(col_min, length_float_decimal)), str(np.round(col_max, length_float_decimal))))
                if (col_min > FLOAT16_MIN) and (col_max < FLOAT16_MAX):
                    data[col] = data[col].astype(np.float16)
                    print("  float16 min: {0:15s} max: {1:15s}".format(str(FLOAT16_MIN), str(FLOAT16_MAX)))
                    print("compress float64 --> float16")
                elif (col_min > FLOAT32_MIN) and (col_max < FLOAT32_MAX):
                    data[col] = data[col].astype(np.float32)
                    print("  float32 min: {0:15s} max: {1:15s}".format(str(FLOAT32_MIN), str(FLOAT32_MAX)))
                    print("compress float64 --> float32")
                else:
                    pass
                memory_after_compress = memory_usage(data, 0)
                print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))
                print('='*length_interval)

            if col_dtype == 'int64':
                print(" variable min: {0:15s} max: {1:15s}".format(str(col_min), str(col_max)))
                type_flag = 64
                if (col_min > INT8_MIN/2) and (col_max < INT8_MAX/2):
                    type_flag = 8
                    data[col] = data[col].astype(np.int8)
                    print("     int8 min: {0:15s} max: {1:15s}".format(str(INT8_MIN), str(INT8_MAX)))
                elif (col_min > INT16_MIN) and (col_max < INT16_MAX):
                    type_flag = 16
                    data[col] = data[col].astype(np.int16)
                    print("    int16 min: {0:15s} max: {1:15s}".format(str(INT16_MIN), str(INT16_MAX)))
                elif (col_min > INT32_MIN) and (col_max < INT32_MAX):
                    type_flag = 32
                    data[col] = data[col].astype(np.int32)
                    print("    int32 min: {0:15s} max: {1:15s}".format(str(INT32_MIN), str(INT32_MAX)))
                    type_flag = 1
                else:
                    pass
                memory_after_compress = memory_usage(data, 0)
                print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))
                if type_flag == 32:
                    print("compress (int64) ==> (int32)")
                elif type_flag == 16:
                    print("compress (int64) ==> (int16)")
                else:
                    print("compress (int64) ==> (int8)")
                print('='*length_interval)

    print()
    memory_after_compress = memory_usage(data, 0)
    print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))
        
