################################################################################
## IMPORT LIBRARIES
################################################################################
## Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime as dt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import boto3
#from dotenv import load_dotenv
################################################################################
## 1.0 LOAD DATA
################################################################################
#lpw_data = pd.read_csv('/content/drive/MyDrive/LPW/Chris LPW Focused Divisional structure and Metrics for 2021 draft April 2022.xlsx - dataset2.csv')
#load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=st.secrets["ACCESS_KEY"],
    aws_secret_access_key=st.secrets["SECRET_KEY"],
    region_name=st.secrets["AWS_REGION"]
)

bucket = "dashboatd-lp"
file_name = "Chris LPW Focused Divisional structure and Metrics for 2021 draft April 2022.xlsx - dataset2 .csv"

 
# 's3' is a key word. create connection to S3 using default config and all buckets within S3

obj = s3.get_object(Bucket= bucket, Key= file_name) 
# get object and file (key) from bucket

lpw_data = pd.read_csv(obj['Body']) # 'Body' is a key word

################################################################################
## 2.0 DATA CLEANING
################################################################################
## 1. Cancelled 2022
## Change 2022 to 1 and missing to 0

lpw_data.loc[lpw_data['Cancelled 2022'] == '2022', 'Cancelled 2022'] = 1
lpw_data.loc[lpw_data['Cancelled 2022'] == 'LPW', 'Cancelled 2022'] = 0
lpw_data.loc[lpw_data['Cancelled 2022'] == 'Reactivated 2022', 'Cancelled 2022'] = 0
lpw_data['Cancelled 2022'].fillna(0,inplace=True)

## 2. Account Manager

## Clean up account manager names
lpw_data['Account Manager'] = lpw_data['Account Manager'].str.strip()
lpw_data['Account Manager'] = lpw_data['Account Manager'].str.upper()
lpw_data.loc[lpw_data['Account Manager'].str.contains('CURTIS PHELAN', case=False), 'Account Manager'] = 'CURTIS PHELAN'
lpw_data.loc[lpw_data['Account Manager'].str.contains("CURTIS K PHELAN", case=False), 'Account Manager'] = 'CURTIS PHELAN'
lpw_data.loc[lpw_data['Account Manager'].str.contains("BJ O'Neal", case=False), 'Account Manager'] = 'BJ O’NEAL'
lpw_data.loc[lpw_data['Account Manager'].str.contains("DALE GBBONS", case=False), 'Account Manager'] = 'DALE GIBBONS'
lpw_data.loc[lpw_data['Account Manager'].str.contains("DANIEL M GIORDANO", case=False), 'Account Manager'] = 'DANIEL GIORDANO'
lpw_data.loc[lpw_data['Account Manager'].str.contains("DANIEL S. HASSLER|DANIEL SHANE HASSLER", case=False), 'Account Manager'] = 'DANIEL HASSLER'
lpw_data.loc[lpw_data['Account Manager'].str.contains("ERIC A. MERZ", case=False), 'Account Manager'] = 'ERIC MERZ'
lpw_data.loc[lpw_data['Account Manager'].str.contains("GERARTD CARPENTER", case=False), 'Account Manager'] = 'GERARD CARPENTER'
lpw_data.loc[lpw_data['Account Manager'].str.contains("SCOTT AND PAM LIPPITT|SCOTT D LIPPITT", case=False), 'Account Manager'] = 'PAM AND SCOTT LIPPITT'
lpw_data.loc[lpw_data['Account Manager'].str.contains("STEVEN LOVER", case=False), 'Account Manager'] = 'STEVE LOVER'

## Create coach name column with both first and last names
#lpw_data["coach_name"] = lpw_data["New Coach Name"].str.upper() + '' + lpw_data["Unnamed: 3"].str.upper() 
lpw_data["New Coach Name"] = lpw_data["New Coach Name"].str.strip()
lpw_data["Unnamed: 3"] = lpw_data["Unnamed: 3"].str.strip()
lpw_data["new_coach_name"] = lpw_data["New Coach Name"].str.upper() +  ' ' +lpw_data["Unnamed: 3"].str.upper()

## Drop new coach name and unnamed 3
lpw_data.drop(['New Coach Name','Unnamed: 3'],axis=1,inplace=True)

## 3. Start Date
## Replace Apriil with April
lpw_data['Start Date'] = lpw_data['Start Date'].str.replace('Aprril','April')
lpw_data['Start Date'] = lpw_data['Start Date'].str.replace('Ap 20','April 20')
lpw_data.loc[lpw_data['Start Date'].notnull(),'Start Date'] = lpw_data.loc[lpw_data['Start Date'].notnull(),'Start Date']+" "+ "2021"
lpw_data['Start Date'] = pd.to_datetime(lpw_data['Start Date'])

## 4. Date Cancelled
## Add 2021 to date cancelled
lpw_data['Date Cancelled'] = lpw_data['Date Cancelled'].replace('Reactivated 2022',np.nan)
lpw_data.loc[lpw_data['Date Cancelled'].notnull(),'Date Cancelled'] = lpw_data.loc[lpw_data['Date Cancelled'] .notnull(),'Date Cancelled']+" "+ "2021"
lpw_data['Date Cancelled'] = pd.to_datetime(lpw_data['Date Cancelled'])

## 5. Stopped
lpw_data['Stopped'].fillna(0,inplace=True)

lpw_data.loc[((lpw_data['Account Manager'] == 'JUAN VELASCO') & (lpw_data['new_coach_name'] == 'FELICIA JERVIS')),['Stopped']] = 1.0
lpw_data.loc[((lpw_data['Account Manager'] == 'MARK EILAND') & (lpw_data['new_coach_name'] == 'SCOTT SELFE')),['Stopped']] = 1.0
lpw_data.loc[((lpw_data['Account Manager'] == 'W. BURT LOCKARD') & (lpw_data['new_coach_name'] == 'CHRIS TAYLOR')),['Stopped']] = 1.0

lpw_data.loc[((lpw_data['Account Manager'] == 'HUGH TAFEL') & (lpw_data['new_coach_name'] == 'TODD BOYUM')),['Stopped']] = 0.0
lpw_data.loc[((lpw_data['Account Manager'] == 'REBECCA GOLD') & (lpw_data['new_coach_name'] == 'ANDY WEISS')),['Stopped']] = 0.0
lpw_data.loc[((lpw_data['Account Manager'] == 'SARA NAY') & (lpw_data['new_coach_name'] == 'DR. AMR MANSOUR')),['Stopped']] = 0.0
lpw_data.loc[((lpw_data['Account Manager'] == 'SCOTT WALKER') & (lpw_data['new_coach_name'] == 'CLAYTON NEUMEYER')),['Stopped']] = 0.0
lpw_data.loc[((lpw_data['Account Manager'] == 'STEVE FELD') & (lpw_data['new_coach_name'] == 'JEN BUMP')),['Stopped']] = 0.0
lpw_data.loc[((lpw_data['Account Manager'] == 'STEVEN KIRCH') & (lpw_data['new_coach_name'] == 'CHRISTOPHER HILGENDORF')),['Stopped']] = 0.0


## 6. Churn rate
## Drop churn rate
lpw_data.drop(['% of churn'],axis=1,inplace=True)

## 7. Country
##Fill missing data with unknown
lpw_data['Country'].fillna('Unknown',inplace=True)

## 8. Inquiry  Call 1
## Change NH to 0

lpw_data.loc[lpw_data['Inquiry  Call 1'] == 'NH','Inquiry  Call 1']= '0.0'
lpw_data.loc[lpw_data['Inquiry  Call 1'] == 0,'Inquiry  Call 1']= '0.0'
lpw_data.loc[lpw_data['Inquiry  Call 1'] == '85.0','Inquiry  Call 1'] = '0.0'

## 9. Onboaridng
lpw_data['Onboaridng'].fillna(0,inplace=True)

## 10. Walt Call 2
## Change NH to 0
lpw_data.loc[lpw_data['Walt Call 2'] == 'NH','Walt Call 2']= '0'
lpw_data.loc[lpw_data['Walt Call 2'] == '60','Walt Call 2']= '1'

## 11. walt Call 3
lpw_data.loc[lpw_data['walt Call 3'] == 'NH','walt Call 3']= '0'
lpw_data['walt Call 3'].fillna(0,inplace=True)
lpw_data.loc[lpw_data['walt Call 3'] == 33.0,'walt Call 3'] = 0

## 12. RP 1
lpw_data.loc[lpw_data['RP 1'] == 'NH','RP 1']= '0'
lpw_data['RP 1'].fillna('0',inplace=True)
lpw_data.loc[lpw_data['RP 1'] == '32','RP 1']= '0'

## 13. RP2
#lpw_data.loc[lpw_data['RP2'] == 'NH','RP2']= '0'
lpw_data['RP2'].fillna(0,inplace=True)
lpw_data.loc[lpw_data['RP2'] == 27.0,'RP2']= 0

## 14. RP 3
lpw_data['RP 3'].fillna(0,inplace=True)
lpw_data.loc[lpw_data['RP 3'] == 19.0,'RP 3']= 0

## 15. Alan LG 1 Group
lpw_data['Alan LG 1 Group'].fillna(0,inplace=True)
lpw_data.loc[lpw_data['Alan LG 1 Group'] == 40.0,'Alan LG 1 Group']= 0

## 16. LG 2
lpw_data['LG 2'].fillna(0,inplace=True)
lpw_data.loc[lpw_data['LG 2'] == 17.0,'LG 2']= 0

## 17. Completed Onboarded
lpw_data['Completed Onboarded'].fillna(0,inplace=True)

################################################################################
## 3.0 FEATURE ENGINEERING
################################################################################

## 1. Time on platform

lpw_data['time_on_platform'] = (lpw_data['Date Cancelled'] - lpw_data['Start Date']).dt.days




################################################################################
## STREAMLIT INITIALIZE
################################################################################
## Configure page
st.set_page_config(page_title="LPW ANALYTICS",page_icon=":bar_chart:",layout='wide')

## Set title
st.title('LPW 2021 ANALYTICS DASHBOARD')

st.header("SUMMARY STATISTICS")
################################################################################
## 4.0 DESCRIPTIVE STATS
################################################################################

## 1. Count of Account managers
account_managers_count = lpw_data['Account Manager'].nunique()
#account_managers_count
## 2.1 Count of new coaches
new_coaches_count = lpw_data['new_coach_name'].nunique()
#new_coaches_count
## 2.2 Count of new coaches by account manager
coaches_by_am = lpw_data.groupby('Account Manager')['new_coach_name'].nunique().reset_index().rename(columns={'new_coach_name':'new_coaches_count'})
coaches_by_am['percent'] = round(coaches_by_am['new_coaches_count'] / coaches_by_am['new_coaches_count'].sum(),4)*100

## 2.3 Top Account managers with most coaches
n=30

coach_count_topn = coaches_by_am.sort_values('percent',ascending=False).head(n)
coach_count_topn.sort_values('percent',ascending=True,inplace=True)
#px.bar(coach_count_topn,x="Percent",y='Account Manager',orientation="h",title="<b>Coach Distribution (top {}) by Account Manager 2021</b>".format(n),template="plotly_white",color_discrete_sequence=['Blue']*len(am_attrition_topn))

################################################################################
## KPI'S
################################################################################
## Row 1
kpi1, kpi2 = st.columns(2)

kpi1.metric(label="Account Managers Count",value=f"{account_managers_count:,}")

kpi2.metric(label="New Coaches Count",value=f"{new_coaches_count:,}")


##3.0 Stopped (Churn Rate)
stopped_df = lpw_data['Stopped'].value_counts().rename_axis('stopped').reset_index(name='count')
stopped_df.loc[stopped_df['stopped'] == 0.0,'stopped'] = 'Active'
stopped_df.loc[stopped_df['stopped'] == 1.0,'stopped'] = 'Cancelled'



#############
## PIE CHARTS
#############

pie1, pie2 = st.columns(2)

fig1 = px.bar(coach_count_topn,x="percent",y='Account Manager',text='new_coaches_count',orientation="h",title="<b>".format(n),template="plotly_white",color_discrete_sequence=['Blue']*len(coach_count_topn)) 
fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)),
                   title={
                       'text': "<b>Coach Count (top {}) by Account Manager 2021</b>".format(n),
                       'y':0.9,
                       'x':0.5,
                       'xanchor': 'center',
                       'yanchor': 'top'})          


fig2 = go.Figure(data=[go.Pie(labels=stopped_df['stopped'], values=stopped_df['count'], hole=.3)])
fig2.update_layout(
    title={
        'text': "<b>Churn Rate 2021</b>",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})


pie1.plotly_chart(fig1)
pie2.plotly_chart(fig2)

##3.1 Number of coaches cancelled in 2021 by am
coaches_cancelled_by_am = lpw_data.groupby('Account Manager')['Stopped'].sum().reset_index().rename(columns={'Stopped':'Cancelled_count'})

am_attrition = pd.merge(coaches_by_am[['Account Manager','new_coaches_count']],coaches_cancelled_by_am)
am_attrition['Attrition Percent'] = (round(am_attrition['Cancelled_count'] / am_attrition['new_coaches_count'],4))*100
am_attrition.sort_values('Attrition Percent',ascending=True,inplace=True)

n=30
am_attrition_topn = am_attrition.sort_values('Attrition Percent',ascending=False).head(n)
am_attrition_topn.sort_values('Attrition Percent',ascending=True,inplace=True)

fig_atr_by_ch = px.bar(am_attrition_topn,x="Attrition Percent",y='Account Manager',text='Cancelled_count',orientation="h",template="plotly_white",color_discrete_sequence=['Blue']*len(am_attrition_topn))
fig_atr_by_ch.update_layout(
    title={
        'text': '<b>Attrition percent (top {}) by Account Manager 2021</b>'.format(n),
        'y':1.0,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig_atr_by_ch.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)))

#####################################
## BAR CHART
#####################################
bar_chart1,barchart2 = st.columns(2)


bar_chart1.plotly_chart(fig_atr_by_ch)

## 5.1 How long do the attriting coaches last
attriting = lpw_data.loc[lpw_data['Stopped']==1]

attriting2 = attriting.loc[attriting['time_on_platform'] > 0]

min_days = attriting2['time_on_platform'].describe()['min']
percentile_25_days = attriting2['time_on_platform'].describe()['25%']
median_days = attriting2['time_on_platform'].describe()['50%']
mean_days = attriting2['time_on_platform'].describe()['mean']
percentile_75_days = attriting2['time_on_platform'].describe()['75%']
max_days = attriting2['time_on_platform'].describe()['max']
attriting_times = pd.DataFrame({'Statistic':['min','percentile_25','median','mean','percentile_75','max'],'Days':[min_days,percentile_25_days,median_days,mean_days,percentile_75_days,max_days]})


##############################################
## Days on platform df
#############################################
#st.write('Days on platform for Cancelled Coaches')
st.header("TIME STATISTICS")
dataset1,dataset2 = st.columns(2)
dataset1.markdown('**Days on platform for Cancelled Coaches 2021**')
dataset1.dataframe(attriting_times)


##6.1 Distribution by country
country_dist = lpw_data['Country'].value_counts().rename_axis('Country').reset_index(name='count')

country_dist2 = country_dist.loc[country_dist['Country'] != 'Unknown']
country_dist2.sort_values('count',ascending=True,inplace=True)
#px.bar(country_dist2,x="count",y='Country',orientation="h",title="<b>Attrition percentad(10) by Account Manager 2021</b>",template="plotly_white",color_discrete_sequence=['Blue']*len(country_dist2))


########################
#COUNTRY PIE CHART
#######################
country_bar_plot = px.bar(country_dist2,x="count",y='Country',orientation="h",template="plotly_white",color_discrete_sequence=['Blue']*len(country_dist2))
country_bar_plot.update_layout(
    title={
        'text': '<b>Coach Count by Country 2021</b>',
        'y':1.0,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

country_bar_plot.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)))

barchart2.plotly_chart(country_bar_plot)

################################################
## 7.1 Where do they drop off

onboarding = lpw_data[['new_coach_name','Inquiry  Call 1', 'Walt Call 2', 'walt Call 3', 'RP 1', 'RP2', 'RP 3', 'Alan LG 1 Group','LG 2']]
onboarding.set_index('new_coach_name', inplace=True)


calls_columns = ['Inquiry  Call 1', 'Walt Call 2', 'walt Call 3']
calls = onboarding[calls_columns]
calls[calls_columns] = calls[calls_columns] .apply(pd.to_numeric, errors='coerce', axis=1)
calls = calls.loc[calls['Inquiry  Call 1']==1]

#########################
#ONBOARDING CALLS HEATMAP
#########################
st.header("ONBOARDING STATISTICS")
st.subheader('CALLS STATISTICS')
onboarding_1, onboarding2 = st.columns(2)
heatmap1 = px.imshow(calls,text_auto=True,aspect="auto",color_continuous_scale='Blues')
heatmap1.update_xaxes(side="top")

onboarding_1.markdown('**Onboarding Calls Heatmap 2021**')
onboarding_1.plotly_chart(heatmap1)


inquiry_call_1_count = len(calls.loc[calls['Inquiry  Call 1']== 1.0])
walt_call_2_count = len(calls.loc[calls['Walt Call 2']== 1.0])
walt_call_3_count = len(calls.loc[calls['walt Call 3']== 1.0])

calls_count = pd.DataFrame({'Stage':['Inquiry  Call 1','Walt Call 2','walt Call 3'],'Coach_Count':[inquiry_call_1_count,walt_call_2_count,walt_call_3_count]})
calls_count['Percent'] = round((calls_count['Coach_Count']/calls_count['Coach_Count'][0]),4)*100

################################
# ONBOARDING CALLS BAR GRAPH
################################
#barchart2,barchart3 = st.columns(2)
calls_count.sort_values('Percent',ascending=True,inplace=True)

fig_calls_count = px.bar(calls_count,x="Percent",y='Stage',text='Coach_Count',orientation="h",template="plotly_white",color_discrete_sequence=['lightpink']*len(calls_count))

fig_calls_count.update_layout(
    title={
        'text': '<b>Onboarding Calls Transition Percentages 2021</b>',
        'y':1.0,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig_calls_count.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)))


onboarding2.plotly_chart(fig_calls_count)

####################################
## Role Play (RP)
####################################
st.subheader('ROLE PLAY STATISTICS')
role_play1,role_play2 = st.columns(2)
rp_columns = ['RP 1', 'RP2', 'RP 3']
rp = onboarding[rp_columns]

rp[rp_columns] = rp[rp_columns].apply(pd.to_numeric, errors='coerce', axis=1)
rp = rp.loc[rp['RP 1']==1]

heatmap2 = px.imshow(rp,text_auto=True,aspect="auto",color_continuous_scale='Blues')
heatmap2.update_xaxes(side="top")
role_play1.plotly_chart(heatmap2)


rp_1_count = len(rp.loc[rp['RP 1']== 1.0])
rp_2_count = len(rp.loc[rp['RP2']== 1.0])
rp_3_count = len(rp.loc[rp['RP 3']== 1.0])

rp_count = pd.DataFrame({'Stage':['RP 1', 'RP2', 'RP 3'],'Coach_Count':[rp_1_count,rp_2_count,rp_3_count]})
rp_count['Percent'] = round((rp_count['Coach_Count']/rp_count['Coach_Count'][0]),4)*100

rp_count.sort_values('Percent',ascending=True,inplace=True)

fig_rp_count = px.bar(rp_count,x="Percent",y='Stage',text='Coach_Count',orientation="h",template="plotly_white",color_discrete_sequence=['lightpink']*len(calls_count))

fig_rp_count.update_layout(
    title={
        'text': '<b>Role Play Transition Percentages 2021</b>',
        'y':1.0,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig_rp_count.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)))


role_play2.plotly_chart(fig_rp_count)

######################################
## Lead Generation (LG)
######################################
st.subheader('LEAD GENERATION STATISTICS')
lead_gen_1,lead_gen_2 = st.columns(2)

lg_columns = ['Alan LG 1 Group', 'LG 2']
lg = onboarding[lg_columns]
#lg = lg.loc[lg['Alan LG 1 Group']==1]
lg[lg_columns] = lg[lg_columns].apply(pd.to_numeric, errors='coerce', axis=1)
lg = lg.loc[lg['Alan LG 1 Group']==1.0]

heatmap3 = px.imshow(lg,text_auto=True,aspect="auto",color_continuous_scale='Blues')
heatmap3.update_xaxes(side="top")
lead_gen_1.plotly_chart(heatmap3)


###########################
# LEAD GENERATION BAR
###########################

#barchart4,barchart5 = st.columns(2)

lg_1_count = len(lg.loc[lg['Alan LG 1 Group']== 1.0])
lg_2_count = len(lg.loc[lg['LG 2']== 1.0])

lg_count = pd.DataFrame({'Stage':['Alan LG 1 Group', 'LG 2'],'Coach_Count':[lg_1_count,lg_2_count]})
lg_count['Percent'] = round((lg_count['Coach_Count']/lg_count['Coach_Count'][0]),4)*100

lg_count.sort_values('Percent',ascending=True,inplace=True)

fig_lg_count = px.bar(lg_count,x="Percent",y='Stage',text='Coach_Count',orientation="h",template="plotly_white",color_discrete_sequence=['lightpink']*len(calls_count))

fig_lg_count.update_layout(
    title={
        'text': '<b>Lead Generation Transition Percentages 2021</b>',
        'y':1.0,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig_lg_count.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)))

lead_gen_2.plotly_chart(fig_lg_count)



#######################################
## Cohort Analysis
#######################################
st.header('COHORT ANALYSIS')

#cohort_data = lpw_data.loc[((lpw_data['Start Date'].notnull()) & (lpw_data['Date Cancelled'].notnull()))]
cohort_data = lpw_data.loc[(lpw_data['Start Date'].notnull())]
cohort_data = cohort_data[['Cancelled 2022','Account Manager','new_coach_name','Stopped','Start Date','Date Cancelled']]

## Set last date as 2022-01-01
last_date = pd.to_datetime('2022-01-01')
cohort_data['Date Cancelled'].fillna(last_date,inplace=True)

## Helper fxns
## Define get month fxn

def get_month (x):

  return dt.datetime(x.year,x.month,1)

## call get month for start month

cohort_data['Start_Month'] = cohort_data['Start Date'].apply(get_month)


## get date elements

def get_date_elements (df,column):

  day = df[column].dt.day
  month = df[column].dt.month
  year = df[column].dt.year

  return day,month,year

## Get elements for cohort and cancelled

_,Cohort_Month,Cohort_Year = get_date_elements(cohort_data,'Start Date')
_,Cancelled_Month,Cancelled_Year = get_date_elements(cohort_data,'Date Cancelled')

year_diff = Cancelled_Year - Cohort_Year
month_diff = Cancelled_Month - Cohort_Month
cohort_data['Cohort_Index'] = year_diff*12 + month_diff + 1
## Eliminate peter cockfrot who has cancelled date earlier than start date
cohort_data = cohort_data.loc[cohort_data['Cohort_Index'] >= 1]

## count unique coach by start month and cohort index

cohort_data2 = cohort_data.groupby(['Start_Month','Cohort_Index'])['new_coach_name'].nunique().reset_index()

## Pivot table of Start Month and Months on platform
cohort_table = cohort_data2.pivot(index='Start_Month',columns='Cohort_Index',values='new_coach_name')

## Re-format index
cohort_table.index = cohort_table.index.strftime('%B %Y')

## Cumsum by row
cohort_table2 = cohort_table.cumsum(axis = 1, skipna = True)

## Create percentages
cohort_table3 = cohort_table2.div(cohort_table.sum(axis=1), axis=0)

## Cummulative Drop Off coaches
#plt.figure(figsize=(21,10))
#sns.heatmap(cohort_table3,annot=True,cmap="Blues")
#plt.title('Cumulative Drop Off Rates Cohort Analysis')
#plt.show()

## Retention rates
cohort_table4 = cohort_table3.applymap(lambda value: -value + 1)

############################
# COHORT TABLES
###########################
cohort_table_1,cohort_table_2 = st.columns(2)

rt_heatmap = px.imshow(cohort_table4.round(2), text_auto=True,color_continuous_scale='Blues',title='<b>Retention Rates 2021 Cohort Analysis</b>')
#rt_heatmap.update_xaxes(side="top")
rt_heatmap.update_layout(plot_bgcolor="rgba(0,0,0,0)")
cohort_table_1.plotly_chart(rt_heatmap)


rt_cum_heatmap = px.imshow(cohort_table2, text_auto=True,color_continuous_scale='Blues',title='<b>Cummulative Drop-Off 2021 Cohort Analysis</b>')
#rt_cum_heatmap.update_xaxes(side="top")
rt_cum_heatmap.update_layout(plot_bgcolor="rgba(0,0,0,0)")
cohort_table_2.plotly_chart(rt_cum_heatmap)


start_dates = lpw_data.loc[lpw_data['Start Date'].notnull()]
#start_dates['start_month'] = start_dates['Start Date'].apply(lambda x: x.strftime('%B-%Y')) 
start_dates['start_month'] = start_dates['Start Date'].apply(get_month) 
start_dates_count = start_dates.groupby('start_month')['start_month'].count().reset_index(name="counts")
start_dates_count.sort_values('start_month',ascending=True)
start_dates_count['Start_Month'] = start_dates_count['start_month'].apply(lambda x: x.strftime('%B-%Y'))

fig_start_dates_count = px.bar(data_frame=start_dates_count, x="Start_Month", y="counts",color_discrete_sequence=['Blue'])
fig_start_dates_count.update_layout(
    title={
        'text': '<b>New Coaches on Platform by Month 2021</b>',
        'y':1.0,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig_start_dates_count.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)))

dataset2.plotly_chart(fig_start_dates_count)
