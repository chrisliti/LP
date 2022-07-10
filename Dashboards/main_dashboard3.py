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

################################################################################
## 1.0 LOAD DATA
################################################################################


googleSheetId = '1AcizaWb91JUn8dBcycIkrKpWqA43FRYaVfP6KGAEop8'
worksheetName = 'clean_data'
URL = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}'.format(
	googleSheetId,
	worksheetName
)

focus_data = pd.read_csv(URL)

################################################################################
## 2.0 DATA CLEANING
################################################################################
## 1. Cancelled 2022
## Change 2022 to 1 and missing to 0

focus_data.loc[focus_data['Cancelled 2022'] == '2022', 'Cancelled 2022'] = 1

## 2. Account Manager

## Clean up account manager names
focus_data['Account Manager'] = focus_data['Account Manager'].str.strip()
focus_data['Account Manager'] = focus_data['Account Manager'].str.upper()

## 3. Coach Name

focus_data["Coach Name"] = focus_data["Coach Name"].str.strip()
focus_data["Coach Name"] = focus_data["Coach Name"].str.upper()

## 4. Start Date
focus_data['Start Date'] = pd.to_datetime(focus_data['Start Date'])

## 5. Date Cancelled
focus_data['Date Cancelled'] = pd.to_datetime(focus_data['Date Cancelled'])

## 6. Country

focus_data["Country"] = focus_data["Country"].str.strip()
focus_data["Country"] = focus_data["Country"].str.upper()

################################################################################
## 3.0 FEATURE ENGINEERING
################################################################################

## New start  year column
focus_data['Start Year'] = focus_data['Start Date'].dt.year

## New Cancel  year column
focus_data['Cancel Year'] = focus_data['Date Cancelled'].dt.year

## Days on platform
focus_data['Days On Platform'] = focus_data['Date Cancelled'] - focus_data['Start Date']
focus_data['Days On Platform'] = focus_data['Days On Platform'].dt.days

focus_data['Stopped_2021'] = focus_data['Stopped'].copy()
focus_data.loc[((focus_data['Cancel Year'] == 2022) & (focus_data['Start Year'] == 2021) & (focus_data['Stopped'] == 1)),'Stopped_2021'] = 0


################################################################################
## STREAMLIT INITIALIZE
################################################################################
## Configure page
st.set_page_config(page_title="FOCUSED ANALYTICS",page_icon=":bar_chart:",layout='wide')

## Set title
st.title('FOCUSED - METRICS DASHBOARD')

st.header("SUMMARY STATISTICS")
################################################################################
## 4.0 DESCRIPTIVE STATS
################################################################################

#year_selections = st.multiselect('Select start year of interest',focus_data['Start Year'].unique(),default=2021)

#focus_data = focus_data.loc[focus_data['Start Year'].isin(year_selections)]

year_selections = st.selectbox('Select start year of interest',focus_data['Start Year'].unique())

focus_data = focus_data.loc[focus_data['Start Year'] == year_selections]


## 1. Count of Account managers
account_managers_count = focus_data['Account Manager'].nunique()
#account_managers_count
## 2.1 Count of new coaches
new_coaches_count = focus_data['Coach Name'].nunique()
#new_coaches_count
## 2.2 Count of new coaches by account manager
coaches_by_am = focus_data.groupby('Account Manager')['Coach Name'].nunique().reset_index().rename(columns={'Coach Name':'new_coaches_count'})
coaches_by_am['percent'] = round(coaches_by_am['new_coaches_count'] / coaches_by_am['new_coaches_count'].sum(),4)*100

## 2.3 Top Account managers with most coaches
n=30

coach_count_topn = coaches_by_am.sort_values('percent',ascending=False).head(n)
coach_count_topn.sort_values('percent',ascending=True,inplace=True)
#px.bar(coach_count_topn,x="Percent",y='Account Manager',orientation="h",title="<b>Coach Distribution (top {}) by Account Manager</b>".format(n),template="plotly_white",color_discrete_sequence=['Blue']*len(am_attrition_topn))

################################################################################
## KPI'S
################################################################################
## Row 1
kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(label="Account Managers Count",value=f"{account_managers_count:,}")

kpi2.metric(label="New Coaches Count",value=f"{new_coaches_count:,}")


##3.0 Stopped 2021 (Churn Rate)
stopped_df = focus_data['Stopped_2021'].value_counts().rename_axis('Stopped_2021').reset_index(name='count')
stopped_df.loc[stopped_df['Stopped_2021'] == 0,'Stopped_2021'] = 'Active'
stopped_df.loc[stopped_df['Stopped_2021'] == 1,'Stopped_2021'] = 'Cancelled'



#############
## PIE CHARTS
#############

pie1, pie2 = st.columns(2)

fig1 = px.bar(coach_count_topn,x="percent",y='Account Manager',text='new_coaches_count',orientation="h",title="<b>".format(n),template="plotly_white",color_discrete_sequence=['Blue']*len(coach_count_topn)) 
fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)),
                   title={
                       'text': "<b>Coach Count (top {}) by Account Manager</b>".format(n),
                       'y':0.9,
                       'x':0.5,
                       'xanchor': 'center',
                       'yanchor': 'top'})          


fig2 = go.Figure(data=[go.Pie(labels=stopped_df['Stopped_2021'], values=stopped_df['count'], hole=.3)])
fig2.update_layout(
    title={
        'text': "<b>Churn Rate</b>",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})


pie1.plotly_chart(fig1)
pie2.plotly_chart(fig2)

##3.1 Number of coaches cancelled in 2021 by am
coaches_cancelled_by_am = focus_data.groupby('Account Manager')['Stopped_2021'].sum().reset_index().rename(columns={'Stopped_2021':'Cancelled_count'})

am_attrition = pd.merge(coaches_by_am[['Account Manager','new_coaches_count']],coaches_cancelled_by_am)
am_attrition['Attrition Percent'] = (round(am_attrition['Cancelled_count'] / am_attrition['new_coaches_count'],4))*100
am_attrition.sort_values('Attrition Percent',ascending=True,inplace=True)

n=20
am_attrition_topn = am_attrition.sort_values('Attrition Percent',ascending=False).head(n)
am_attrition_topn.sort_values('Attrition Percent',ascending=True,inplace=True)

fig_atr_by_ch = px.bar(am_attrition_topn,x="Attrition Percent",y='Account Manager',text='Cancelled_count',orientation="h",template="plotly_white",color_discrete_sequence=['Blue']*len(am_attrition_topn))
fig_atr_by_ch.update_layout(
    title={
        'text': '<b>Attrition percent (top {}) by Account Manager</b>'.format(n),
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
attriting = focus_data.loc[focus_data['Stopped_2021']==1]

attriting2 = attriting.loc[attriting['Days On Platform'] >= 0]

min_days = attriting2['Days On Platform'].describe()['min']
percentile_25_days = attriting2['Days On Platform'].describe()['25%']
median_days = attriting2['Days On Platform'].describe()['50%']
mean_days = attriting2['Days On Platform'].describe()['mean']
percentile_75_days = attriting2['Days On Platform'].describe()['75%']
max_days = attriting2['Days On Platform'].describe()['max']
attriting_times = pd.DataFrame({'Statistic':['min','percentile_25','median','mean','percentile_75','max'],'Days':[min_days,percentile_25_days,median_days,mean_days,percentile_75_days,max_days]})
attriting_times['Days'] = np.ceil(attriting_times['Days'])


##############################################
## Days on platform df
#############################################
#st.write('Days on platform for Cancelled Coaches')
st.header("TIME STATISTICS")
dataset1,dataset2 = st.columns(2)
dataset1.markdown('**Days on platform for Cancelled Coaches**')
dataset1.dataframe(attriting_times)


##6.1 Distribution by country

country_reach = focus_data['Country'].nunique()

kpi3.metric(label="Country Reach",value=f"{country_reach:,}")


country_dist = focus_data['Country'].value_counts().rename_axis('Country').reset_index(name='count')

country_dist2 = country_dist.loc[country_dist['Country'] != 'Unknown']
country_dist2['percent'] = round(country_dist2['count'] / country_dist2['count'].sum(),4)*100
country_dist2.sort_values('count',ascending=True,inplace=True)
#px.bar(country_dist2,x="count",y='Country',orientation="h",title="<b>Attrition percentad(10) by Account Manager</b>",template="plotly_white",color_discrete_sequence=['Blue']*len(country_dist2))


########################
#COUNTRY PIE CHART
#######################
country_bar_plot = px.bar(country_dist2,x="count",y='Country',orientation="h",template="plotly_white",color_discrete_sequence=['Blue']*len(country_dist2))
country_bar_plot.update_layout(
    title={
        'text': '<b>Coach Count by Country</b>',
        'y':1.0,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

country_bar_plot.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)))

barchart2.plotly_chart(country_bar_plot)

#################################################

## Active Months on Platform for Cancelled Coaches
##################################################

barchart3,barchart4 = st.columns(2)

################################################
## 7.1 Where do they drop off

onboarding = focus_data[['Coach Name','Inquiry  Call 1', 'Walt Call 2', 'walt Call 3', 'RP 1', 'RP 2', 'RP 3', 'Alan LG 1 Group','LG 2']]
onboarding.set_index('Coach Name', inplace=True)


calls_columns = ['Inquiry  Call 1', 'Walt Call 2', 'walt Call 3']
calls = onboarding[calls_columns]
calls[calls_columns] = calls[calls_columns] .apply(pd.to_numeric, errors='coerce', axis=1)
calls = calls.loc[((calls['Inquiry  Call 1']==1)| (calls['Walt Call 2']==1) | (calls['walt Call 3']==1))]

#########################
#ONBOARDING CALLS HEATMAP
#########################
st.header("ONBOARDING STATISTICS")
st.subheader('CALLS STATISTICS')
onboarding_1, onboarding2 = st.columns(2)
heatmap1 = px.imshow(calls,text_auto=True,aspect="auto",color_continuous_scale='Blues')
heatmap1.update_xaxes(side="top")

onboarding_1.markdown('**Onboarding Calls Heatmap**')
onboarding_1.plotly_chart(heatmap1)


inquiry_call_1_count = len(calls.loc[calls['Inquiry  Call 1']== 1])
walt_call_2_count = len(calls.loc[calls['Walt Call 2']== 1])
walt_call_3_count = len(calls.loc[calls['walt Call 3']== 1])

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
        'text': '<b>Onboarding Calls Transition Percentages</b>',
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
rp_columns = ['RP 1', 'RP 2', 'RP 3']
rp = onboarding[rp_columns]

rp[rp_columns] = rp[rp_columns].apply(pd.to_numeric, errors='coerce', axis=1)
rp = rp.loc[((rp['RP 1']==1) | (rp['RP 2']==1) | (rp['RP 3']==1))]

heatmap2 = px.imshow(rp,text_auto=True,aspect="auto",color_continuous_scale='Blues')
heatmap2.update_xaxes(side="top")
role_play1.markdown('**Role Plays Heatmap**')
role_play1.plotly_chart(heatmap2)


rp_1_count = len(rp.loc[rp['RP 1']== 1])
rp_2_count = len(rp.loc[rp['RP 2']== 1])
rp_3_count = len(rp.loc[rp['RP 3']== 1])

rp_count = pd.DataFrame({'Stage':['RP 1', 'RP 2', 'RP 3'],'Coach_Count':[rp_1_count,rp_2_count,rp_3_count]})
rp_count['Percent'] = round((rp_count['Coach_Count']/rp_count['Coach_Count'][0]),4)*100

rp_count.sort_values('Percent',ascending=True,inplace=True)

fig_rp_count = px.bar(rp_count,x="Percent",y='Stage',text='Coach_Count',orientation="h",template="plotly_white",color_discrete_sequence=['lightpink']*len(calls_count))

fig_rp_count.update_layout(
    title={
        'text': '<b>Role Play Transition Percentages</b>',
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
lg = lg.loc[((lg['Alan LG 1 Group']==1) | (lg['LG 2']==1))]

heatmap3 = px.imshow(lg,text_auto=True,aspect="auto",color_continuous_scale='Blues')
heatmap3.update_xaxes(side="top")
lead_gen_1.markdown('**Lead Generation Heatmap**')
lead_gen_1.plotly_chart(heatmap3)


###########################
# LEAD GENERATION BAR
###########################

#barchart4,barchart5 = st.columns(2)

lg_1_count = len(lg.loc[lg['Alan LG 1 Group']== 1])
lg_2_count = len(lg.loc[lg['LG 2']== 1])

lg_count = pd.DataFrame({'Stage':['Alan LG 1 Group', 'LG 2'],'Coach_Count':[lg_1_count,lg_2_count]})
lg_count['Percent'] = round((lg_count['Coach_Count']/lg_count['Coach_Count'][0]),4)*100

lg_count.sort_values('Percent',ascending=True,inplace=True)

fig_lg_count = px.bar(lg_count,x="Percent",y='Stage',text='Coach_Count',orientation="h",template="plotly_white",color_discrete_sequence=['lightpink']*len(calls_count))

fig_lg_count.update_layout(
    title={
        'text': '<b>Lead Generation Transition Percentages</b>',
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

cohort_data = focus_data.loc[((focus_data['Start Date'].notnull()) & (focus_data['Date Cancelled'].notnull()))]
#cohort_data = cohort_data.loc[(focus_data['Start Date'].notnull())]
cohort_data = cohort_data.loc[cohort_data['Stopped_2021']==1]
cohort_data = cohort_data[['Cancelled 2022','Account Manager','Coach Name','Stopped_2021','Start Date','Date Cancelled']]

## Set last date as 2022-01-01
#last_date = pd.to_datetime('2022-01-01')
#cohort_data['Date Cancelled'].fillna(last_date,inplace=True)

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

cohort_data2 = cohort_data.groupby(['Start_Month','Cohort_Index'])['Coach Name'].count().reset_index()

## Pivot table of Start Month and Months on platform
cohort_table = cohort_data2.pivot(index='Start_Month',columns='Cohort_Index',values='Coach Name')

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

rt_heatmap = px.imshow(cohort_table3.round(2), text_auto=True,color_continuous_scale='Blues',title='<b>Attrition Rates Cohort Analysis</b>')
#rt_heatmap.update_xaxes(side="top")
rt_heatmap.update_layout(plot_bgcolor="rgba(0,0,0,0)")
cohort_table_2.plotly_chart(rt_heatmap)


rt_cum_heatmap = px.imshow(cohort_table2, text_auto=True,color_continuous_scale='Blues',title='<b>Cummulative Drop-Off Cohort Analysis</b>')
#rt_cum_heatmap.update_xaxes(side="top")
rt_cum_heatmap.update_layout(plot_bgcolor="rgba(0,0,0,0)")
cohort_table_1.plotly_chart(rt_cum_heatmap)


start_dates = focus_data.loc[focus_data['Start Date'].notnull()]
#start_dates['start_month'] = start_dates['Start Date'].apply(lambda x: x.strftime('%B-%Y')) 
start_dates['start_month'] = start_dates['Start Date'].apply(get_month) 
start_dates_count = start_dates.groupby('start_month')['start_month'].count().reset_index(name="counts")
start_dates_count.sort_values('start_month',ascending=True)
start_dates_count['Start_Month'] = start_dates_count['start_month'].apply(lambda x: x.strftime('%B-%Y'))

fig_start_dates_count = px.bar(data_frame=start_dates_count, x="Start_Month", y="counts",color_discrete_sequence=['Blue'])
fig_start_dates_count.update_layout(
    title={
        'text': '<b>New Coaches on Platform by Month</b>',
        'y':1.0,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig_start_dates_count.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)))

dataset2.plotly_chart(fig_start_dates_count)

########################################################
## Active Months on Platform for Cancelled Coaches
########################################################

cancelled_coaches = focus_data.loc[focus_data['Date Cancelled'].notnull()]
cancelled_coaches = cancelled_coaches.loc[focus_data['Stopped_2021']==1]


## Get elements for cohort and cancelled

_,Start_Month,Start_Year = get_date_elements(cancelled_coaches,'Start Date')
_,Cancelled_Month,Cancelled_Year = get_date_elements(cancelled_coaches,'Date Cancelled')

year_diff = Cancelled_Year - Start_Year
month_diff = Cancelled_Month - Start_Month
cancelled_coaches['Active_Months'] = year_diff*12 + month_diff + 1

cancelled_coaches_grouped = cancelled_coaches.groupby(by=["Active_Months"]).size().reset_index(name="Number of Coaches")
cancelled_coaches_grouped['Percent'] = round((cancelled_coaches_grouped['Number of Coaches']/cancelled_coaches_grouped['Number of Coaches'].sum())*100,2)


active_months_count = px.bar(data_frame=cancelled_coaches_grouped, x="Active_Months",y="Number of Coaches",text='Percent',color_discrete_sequence=['Blue'])

active_months_count.update_layout(
    title={
        'text': '<b>Active Months on Platform for Cancelled Coaches</b>',
        'y':1.0,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

active_months_count.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)))

barchart3.plotly_chart(active_months_count)

#####################################################################################
## Attrition over 12 window to_period
###################################################################################

if year_selections == 2021 :
  jan2021_jan2022 = focus_data.loc[((focus_data['Start Date'] >= '2021-01-01') & (focus_data['Start Date'] < '2022-01-01'))]
  jan2021_jan2022['Churned'] = jan2021_jan2022['Stopped'].copy()
  jan2021_jan2022.loc[jan2021_jan2022['Date Cancelled'] >= '2022-01-01','Churned'] = 0
  jan_window_coaches = len(jan2021_jan2022)
  jan_churn_rate = np.round(jan2021_jan2022['Churned'].value_counts(normalize=True)[1],4)*100
  

  feb2021_feb2022 = focus_data.loc[((focus_data['Start Date'] >= '2021-02-01') & (focus_data['Start Date'] < '2022-02-01'))]
  feb2021_feb2022['Churned'] = feb2021_feb2022['Stopped'].copy()
  feb2021_feb2022.loc[feb2021_feb2022['Date Cancelled'] >= '2022-02-01','Churned'] = 0
  feb_window_coaches = len(feb2021_feb2022)
  feb_churn_rate = np.round(feb2021_feb2022['Churned'].value_counts(normalize=True)[1],4)*100


  mar2021_mar2022 = focus_data.loc[((focus_data['Start Date'] >= '2021-03-01') & (focus_data['Start Date'] < '2022-03-01'))]
  mar2021_mar2022['Churned'] = mar2021_mar2022['Stopped'].copy()
  mar2021_mar2022.loc[mar2021_mar2022['Date Cancelled'] >= '2022-03-01','Churned'] = 0
  mar_window_coaches = len(mar2021_mar2022)
  mar_churn_rate = np.round(mar2021_mar2022['Churned'].value_counts(normalize=True)[1],4)*100


  apr2021_apr2022 = focus_data.loc[((focus_data['Start Date'] >= '2021-04-01') & (focus_data['Start Date'] < '2022-04-01'))]
  apr2021_apr2022['Churned'] = apr2021_apr2022['Stopped'].copy()
  apr2021_apr2022.loc[apr2021_apr2022['Date Cancelled'] >= '2022-04-01','Churned'] = 0
  apr_window_coaches = len(apr2021_apr2022)
  apr_churn_rate = np.round(apr2021_apr2022['Churned'].value_counts(normalize=True)[1],4)*100


  may2021_may2022 = focus_data.loc[((focus_data['Start Date'] >= '2021-05-01') & (focus_data['Start Date'] < '2022-05-01'))]
  may2021_may2022['Churned'] = may2021_may2022['Stopped'].copy()
  may2021_may2022.loc[may2021_may2022['Date Cancelled'] >= '2022-05-01','Churned'] = 0
  may_window_coaches = len(may2021_may2022)
  may_churn_rate = np.round(may2021_may2022['Churned'].value_counts(normalize=True)[1],4)*100


  june2021_june2022 = focus_data.loc[((focus_data['Start Date'] >= '2021-06-01') & (focus_data['Start Date'] < '2022-06-01'))]
  june2021_june2022['Churned'] = june2021_june2022['Stopped'].copy()
  june2021_june2022.loc[june2021_june2022['Date Cancelled'] >= '2022-06-01','Churned'] = 0
  june_window_coaches = len(june2021_june2022)
  june_churn_rate = np.round(june2021_june2022['Churned'].value_counts(normalize=True)[1],4)*100


  window_period = ['Jan2021 - Jan2022','Feb2021 - Feb2022','Mar2021 - Mar2022','Apr2021 - Apr2022','May2021 - May2022','June2021 - June2022']
  window_coach_count = [jan_window_coaches,feb_window_coaches,mar_window_coaches,apr_window_coaches,may_window_coaches,june_window_coaches]
  window_churn_rate = [jan_churn_rate,feb_churn_rate,mar_churn_rate,apr_churn_rate,may_churn_rate,june_churn_rate]

  window_churn_data = pd.DataFrame({'Window Period':window_period,'Window Coach Count':window_coach_count,'Window Churn Rate Percentage':window_churn_rate})


  window_churn_bar = px.bar(data_frame = window_churn_data, x="Window Period", y = 'Window Churn Rate Percentage',text="Window Coach Count",color_discrete_sequence=['Blue'])
  window_churn_bar.update_layout(
      title={
          'text': '<b>Churn Rate Over 12 Months</b>',
          'y':1.0,
          'x':0.5,
          'xanchor': 'center',
          'yanchor': 'top'})

  window_churn_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)))

  barchart4.plotly_chart(window_churn_bar)

else:

  jan2022_jan2023 = focus_data.loc[((focus_data['Start Date'] >= '2022-01-01') & (focus_data['Start Date'] < '2023-01-01'))]
  jan2022_jan2023['Churned'] = jan2022_jan2023['Stopped'].copy()
  jan2022_jan2023.loc[jan2022_jan2023['Date Cancelled'] >= '2023-01-01','Churned'] = 0
  jan_window_coaches = len(jan2022_jan2023)
  jan_churn_rate = np.round(jan2022_jan2023['Churned'].value_counts(normalize=True)[1],4)*100
  


  feb2022_feb2023 = focus_data.loc[((focus_data['Start Date'] >= '2022-02-01') & (focus_data['Start Date'] < '2023-02-01'))]
  feb2022_feb2023['Churned'] = feb2022_feb2023['Stopped'].copy()
  feb2022_feb2023.loc[feb2022_feb2023['Date Cancelled'] >= '2023-02-01','Churned'] = 0
  feb_window_coaches = len(feb2022_feb2023)
  feb_churn_rate = np.round(feb2022_feb2023['Churned'].value_counts(normalize=True)[1],4)*100


  mar2022_mar2023 = focus_data.loc[((focus_data['Start Date'] >= '2022-03-01') & (focus_data['Start Date'] < '2023-03-01'))]
  mar2022_mar2023['Churned'] = mar2022_mar2023['Stopped'].copy()
  mar2022_mar2023.loc[mar2022_mar2023['Date Cancelled'] >= '2023-03-01','Churned'] = 0
  mar_window_coaches = len(mar2022_mar2023)
  mar_churn_rate = np.round(mar2022_mar2023['Churned'].value_counts(normalize=True)[1],4)*100


  apr2022_apr2023 = focus_data.loc[((focus_data['Start Date'] >= '2022-04-01') & (focus_data['Start Date'] < '2023-04-01'))]
  apr2022_apr2023['Churned'] = apr2022_apr2023['Stopped'].copy()
  apr2022_apr2023.loc[apr2022_apr2023['Date Cancelled'] >= '2023-04-01','Churned'] = 0
  apr_window_coaches = len(apr2022_apr2023)
  apr_churn_rate = np.round(apr2022_apr2023['Churned'].value_counts(normalize=True)[1],4)*100


  window_period = ['Jan2022 - Jan2023','Feb2022 - Feb2023','Mar2022 - Mar2023','Apr2022 - Apr2023']
  window_coach_count = [jan_window_coaches,feb_window_coaches,mar_window_coaches,apr_window_coaches]
  window_churn_rate = [jan_churn_rate,feb_churn_rate,mar_churn_rate,apr_churn_rate]

  window_churn_data = pd.DataFrame({'Window Period':window_period,'Window Coach Count':window_coach_count,'Window Churn Rate Percentage':window_churn_rate})

  window_churn_bar = px.bar(data_frame = window_churn_data, x="Window Period", y = 'Window Churn Rate Percentage',text="Window Coach Count",color_discrete_sequence=['Blue'])
  window_churn_bar.update_layout(
      title={
          'text': '<b>Churn Rate Over 12 Months</b>',
          'y':1.0,
          'x':0.5,
          'xanchor': 'center',
          'yanchor': 'top'})

  window_churn_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)))

  barchart4.plotly_chart(window_churn_bar)
