#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from pathlib import Path
get_ipython().run_line_magic('matplotlib', 'inline')


# In[82]:

if not Path("Ressources/Telco-Customer-Churn.csv").is_file():
    with zipfile.ZipFile("Ressources/Telco-Customer-Churn.csv", 'r') as zip_ref:
        zip_ref.extractall("Ressources")

df = pd.read_csv(r"Ressources/Telco-Customer-Churn.csv")


# In[83]:


df.head()


# In[84]:


df.info()


# In[85]:


# Insérez votre code ici
liste_all = []

for value in df['TotalCharges']:
    try:
        float(value)
    except ValueError:
        liste_all.append(value)

unique_values = set(liste_all)
print(unique_values)


# In[86]:


df.TotalCharges.replace({' ': np.nan}, inplace = True)
df.TotalCharges = df.TotalCharges.astype(float)


# In[87]:


df.TotalCharges.isna().sum()


# In[88]:


for i,j in enumerate(df.TotalCharges) :
    df.TotalCharges.fillna(df['MonthlyCharges'][i]* df['tenure'][i], inplace = True)


# In[89]:


import plotly.express as px
fig = px.box(df, y='TotalCharges', title='Distribution of TotalCharges')
fig.show()


# In[90]:


fig = px.box(df, y='MonthlyCharges', title='Distribution of MonthlyCharges')
fig.show()


# In[91]:


df.describe()


# In[92]:


import plotly.express as px

fig = px.pie(df, values=df.Churn.value_counts(), names=['No', 'Yes'], title='Churn or Not',
             labels={'value': 'Count', 'names': 'Churn'})
fig.show()


# In[93]:


import plotly.express as px

fig = px.histogram(df, x='Churn', color='gender', barmode='group')
fig.update_layout(title='Distribution of Churn by Gender')
fig.show()


# In[94]:


'There are similar churners rate for Male and Female customers'


# In[149]:



fig = px.histogram(df, x='Churn', color='PaymentMethod')
fig.update_layout(
    xaxis_title='Churn',
    yaxis_title='Count',
    title='Count of Customers according to PaymentMethod and Churn'
)
fig.show()


# In[96]:


'Most of churners pay by electronic check. In fact there are almost as many customers who churn as those who stay.'


# In[97]:


df.InternetService.value_counts(normalize = True)


# In[147]:



fig = px.histogram(df, x='Churn', color='InternetService')
fig.update_layout(
    xaxis_title='Churn',
    yaxis_title='Count',
    title='Count of Customers according to InternetService and Churn'
)
fig.show()


# In[99]:


'Most churners have a fiber optic InternetService'


# In[100]:


df.SeniorCitizen.value_counts(normalize = True)


# In[151]:



fig = px.histogram(df, x='SeniorCitizen', color='Churn', barmode='group', nbins=2,
                   labels={'SeniorCitizen': 'Senior Citizen', 'Churn': 'Churn'})
fig.update_layout(title='Churn by Senior Citizen')
fig.show()


# In[103]:


'Even if Senior citizens are not the most seen category, we see a high rate of churners among them. Almost same quantity of churners and stayers.'


# In[152]:


fig = px.histogram(df, x='Churn', color='MultipleLines')
fig.update_layout(
    xaxis_title='Churn',
    yaxis_title='Count',
    title='Count of Partners according to MultipleLines'
)
fig.show()


# In[105]:


'Among the churners, we observe less customers without phone service but the proportion of people with or without multiple lines is the same'


# In[106]:


df.tenure.describe()


# In[107]:


plt.figure(figsize = (10,10))
sns.histplot(x='Churn', hue = pd.cut(df.tenure, bins = 4, labels = ['q1','q2','q3','q4']), data = df, multiple="dodge", shrink = .8);


# In[108]:


'Most churners with tenure 0-9 months (duration of staying with the company for customer)'


# In[140]:




fig = px.histogram(df, x='Churn', color='Partner')
fig.update_layout(
    xaxis_title='Churn',
    yaxis_title='Partner Count',
    title='Count of Partners according to Churn'
)
fig.show()


# In[110]:


'Most churners are not partners'


# In[153]:


fig = px.histogram(df, x='Churn', color='OnlineSecurity')
fig.update_layout(
    xaxis_title='Churn',
    yaxis_title='Count',
    title='Count of customers according to OnlineSecurity and Churn'
)
fig.show()


# In[112]:


'most churners do not have Online Security option'


# In[113]:


df.columns


# In[136]:


fig = px.histogram(df, x='Contract', color='Churn')
fig.update_layout(
    xaxis_title='Contract',
    yaxis_title='Count',
    title='Count of different Contract types according to churn'
)
fig.show()


# In[115]:


'Most churners have month-to-month contract'


# In[116]:


df.groupby('Contract').median('MonthlyCharges')['MonthlyCharges'].round()


# In[118]:


fig = px.bar(df.groupby('Contract').median('MonthlyCharges')['MonthlyCharges'].reset_index(), x='Contract', y='MonthlyCharges', color='Contract',
             labels={'Contract': 'Contract', 'MonthlyCharges': 'Median Monthly Charges'},
             title='Median Monthly Charges by Contract')
fig.show()


# In[119]:


'We see that median monthly charges of customers with Month to month contract is 4 units more than the one year contract and the one year contract 5 units more than the 2 year contract. There is no much variance between the values of the different categories. '


# In[135]:


fig = px.bar(df.groupby('Churn').median('MonthlyCharges')['MonthlyCharges'].reset_index(), x='Churn', y='MonthlyCharges', color='Churn',
             labels={'Churn': 'Churn', 'MonthlyCharges': 'Median Monthly Charges'},
             title='Median Monthly Charges by Churn')
fig.show()


# In[122]:


'But the monthly charges of the churners are much higher : 14 units more'


# In[154]:


fig = px.histogram(df, x='Churn', color='TechSupport')
fig.update_layout(
    xaxis_title='Churn',
    yaxis_title='Count',
    title='Count of Tech Support options according to Churn'
)
fig.show()


# In[124]:


'For most of the Churners there is TechSupport'


# In[125]:


'To summarize : Customers tend to leave according to those significant parameters : Contract Month-to-Month, No Tech Support, stay 0 to 9 months with the company, InternetService : Fiber Optic, pay by electronic check, age category : SeniorCitizen, no Online Security. We will now analyse the monthly charges for those categories.'


# In[126]:


import seaborn as sns

plt.figure(figsize=(10,10))
ax = sns.barplot(x='InternetService', y='MonthlyCharges', data=df, hue = 'Churn')
plt.xlabel('InternetService')
plt.ylabel('MonthlyCharges')
plt.title('MonthlyCharges by InternetService by Churn');
plt.show()

# In[127]:


'We can also see that the charges for InternetService Fiber Optic are much higher than DSL. Which represents the category where there is the highest rate of churners.'


# In[128]:


import seaborn as sns

plt.figure(figsize=(10,10))
ax = sns.barplot(x='PaymentMethod', y='MonthlyCharges', data=df, hue = 'Churn')
plt.xlabel('PaymentMethod')
plt.ylabel('MonthlyCharges')
plt.title('Mean MonthlyCharges by PaymentMethod by Churn');
plt.show()

# In[129]:



plt.figure(figsize=(10,10))
ax = sns.barplot(x='TechSupport', y='MonthlyCharges', data=df, hue = 'Churn')
plt.xlabel('TechSupport')
plt.ylabel('MonthlyCharges')
plt.title('Mean MonthlyCharges by TechSupport by Churn');
plt.show()

# In[130]:


'We observe that the most related features to churns have also the most expensive monthly charges comparing to other values of respective categories.' 


# In[131]:



plt.figure(figsize=(10,10))
ax = sns.barplot(x='OnlineSecurity', y='MonthlyCharges', data=df, hue = 'Churn')
plt.xlabel('OnlineSecurity')
plt.ylabel('MonthlyCharges')
plt.title('Mean MonthlyCharges by OnlineSecurity by Churn');
plt.show()

# In[132]:


df.groupby(['Contract', 'TechSupport', 'Churn']).mean('MonthlyCharges')['MonthlyCharges']


# In[133]:


'Among the most expensive functionalities, we see fiber optic for internet service. And as we saw before, a majority of churners didn"t take the option Tech Support and Online Support. An interpretation could be that customers who took this internet service and no support nor online security were not using well the functionalities and did not use the fiber optic to the fullest. Paying high monthly charges, they did not want to proceed the contract because they did not see their interest.'


# In[55]:


"Let's now analyse the feature importances regarding the class churner"


# In[56]:


'First we need to change all the categorical non numerical values into numerical ones.'


df.Churn.replace({'No': 0, 'Yes' : 1}, inplace= True)

df.select_dtypes('object')

df.gender.replace({'Male':0, 'Female':1}, inplace = True)

for i in df[['Dependents','Partner','PaperlessBilling','PhoneService']]:
    df[i].replace({'No':0,'Yes':1}, inplace = True)

df.MultipleLines = df.MultipleLines.replace({'No phone service' : 0, 'No': 1, 'Yes' : 2})

df.InternetService.replace({'No':0, 'DSL':1, 'Fiber optic':2}, inplace = True)

for i in df[['InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport', 'StreamingTV','StreamingMovies',]]:
    df[i].replace({"No internet service":0,'No':1,'Yes':2},inplace = True)

df.select_dtypes('object')

df['Contract'].replace({'Month-to-month':0, 'One year':1, 'Two year':2}, inplace = True)

df_train = pd.concat([pd.get_dummies(df.PaymentMethod), df], axis =1).drop('customerID', axis =1)


# In[57]:


df_train.drop('PaymentMethod', axis=1, inplace = True)


# In[58]:


corr_matrix = df.corr()


# In[73]:


fig, ax = plt.subplots(figsize=(10,8))
ax.set_facecolor('white')
ax.imshow(np.ones_like(corr_matrix), cmap='gray_r', interpolation='nearest')

# set the tick labels and rotation for the x and y axes
ax.set_xticks(np.arange(len(corr_matrix.columns)) + 0.5)
ax.set_yticks(np.arange(len(corr_matrix.columns)) + 0.5)

# format ticks
ax.set_yticklabels(corr_matrix.columns, fontsize=10)
ax.set_xticklabels(corr_matrix.columns, fontsize=10, rotation = 45, ha='right')

# create circles with radius proportional to the absolute value of correlation
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        correlation = corr_matrix.iat[i, j]
        norm = plt.Normalize(-1, 1)  # specify the range of values for the colormap
        sm = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
        color = sm.to_rgba(correlation)
        circle = Circle((i+0.5, j+0.5), radius=abs(correlation)/2.5, facecolor=color)
        ax.add_patch(circle)

# create grid lines between the cells of the heatmap
ax.set_xticks(np.arange(len(corr_matrix.columns) + 1), minor=True)
ax.set_yticks(np.arange(len(corr_matrix.columns) + 1), minor=True)
ax.grid(which="minor", color="lightgray", linestyle="solid", linewidth=2)

# add rectangle around the grid
rect = plt.Rectangle((0, 0), len(corr_matrix.columns), len(corr_matrix.columns), linewidth=5, edgecolor='lightgray', facecolor='none')
ax.add_patch(rect)

# add color bar
norm = mcolors.Normalize(vmin=-1, vmax=1)
c_scale = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
cbar = plt.colorbar(c_scale, ax=ax)

plt.show()


# In[1146]:


'According to the heatmap, the features that seem to be the most correlated to the churner classes are : Electronic check, InternetService, tenure, Contract.'


# In[1147]:


data = df_train.drop('Churn', axis = 1)
target = df_train['Churn']


# In[1148]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 123)


# In[1149]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# In[1150]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=-1, random_state = 321)
clf.fit(X_train_scaled,y_train)
y_probas = clf.predict_proba(X_test_scaled)


# In[1156]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
param_grid = {'n_estimators': [10, 50, 100],
              'max_depth': [5, 10, 20],
              'min_samples_split': [2, 5, 10]
              }
#choix du classifieur
rf = RandomForestClassifier()
#grid_search avec liste des paramètres et choix cv et scoring
grid_search = GridSearchCV(rf, param_grid, cv=7, scoring='accuracy')
# entraînement
grid_search.fit(X_train_scaled, y_train)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best score: {:.2f}".format(grid_search.best_score_))


# In[1157]:


rf = RandomForestClassifier(max_depth = 5, min_samples_split = 2, n_estimators = 10)
rf.fit(X_train_scaled,y_train)
y_probas = rf.predict_proba(X_test_scaled)


# In[1158]:


import scikitplot as skplt
from scikitplot import metrics
metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()

# In[1161]:


from imblearn.over_sampling import RandomOverSampler, SMOTE
X_ro,y_ro = RandomOverSampler().fit_resample(X_train,y_train)
X_sm,y_sm = SMOTE().fit_resample(X_train,y_train)
print(dict(pd.Series(y_ro).value_counts()))
print(dict(pd.Series(y_sm).value_counts()))


# In[1163]:


from imblearn.metrics import classification_report_imbalanced

rf1 = RandomForestClassifier(max_depth = 5, min_samples_split = 2, n_estimators = 10)

rf1.fit(X_ro,y_ro)
y_predi = rf1.predict(X_test)
pd.crosstab(y_predi,y_test,rownames=['classe prédite'], colnames=['classe réelle'])

print(classification_report_imbalanced(y_test, y_predi,target_names = ['class 0', 'class 1']))


# In[1164]:


rf1.score(X_test, y_test)


# In[1167]:


from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

X_ru,y_ru = RandomUnderSampler().fit_resample(X_train,y_train)
X_cc,y_cc = ClusterCentroids().fit_resample(X_train,y_train)


# In[1168]:


rf2 = RandomForestClassifier(max_depth = 5, min_samples_split = 2, n_estimators = 10)

rf2.fit(X_ru,y_ru)
y_predi = rf1.predict(X_test)
pd.crosstab(y_predi,y_test,rownames=['classe prédite'], colnames=['classe réelle'])

print(classification_report_imbalanced(y_test, y_predi,target_names = ['class 0', 'class 1']))


# In[1169]:


rf2.score(X_test, y_test)


# In[1170]:


'I tried many options to improve the accuracy of the model but the best one : RandomForest was the best without resampling it.'


# In[1171]:


y_probas1 = rf1.predict_proba(X_test_scaled)
y_probas2 = rf2.predict_proba(X_test_scaled)


# In[1172]:


metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()

# In[ ]:


metrics.plot_cumulative_gain(y_test, y_probas1)
plt.show()

# In[ ]:


metrics.plot_cumulative_gain(y_test, y_probas2)
plt.show()

# In[1175]:


'We see it clearly by the cumulative gain curve that the first is the most relevant one. And it means that the 40 pourcent of customers having the best score (the ones with the higher probability of becoming churners) around 80 % are actual churners. '


# In[1176]:


features = data.columns
features_importance = {}
sorted_features = {}
for x,j in zip(features, rf.feature_importances_):
    features_importance[x] = j
sorted_features = sorted(features_importance.items(), key=lambda x:x[1], reverse=True) 
print(sorted_features[:8])


# In[1279]:


'In the feature importances, the one that are relevant by explaining the choice of leaving for the customers are the Charges. And this feature, as I interpretated before is linked to the following ones : OnlineSecurity, InternetService and TechSupport.'


# In[1265]:


'According to the previous observation, a solution to prevent churns would be to reduce the price of the Month-to-Month contract but the income would also decrease. So the solution would be directed to the better understanding of the client about the tools included in his/her contract. A way would be for example to target (thanks to the predictions and cumulative gain curve) the potential churners and discuss a preferential offer on the tech support and online security. Next step would be to change the Month-to-Month contract to a year or two year one and automatic payment method to ensure that the customer would stay long enough to use all the tools and increase customer satisfaction'


# In[ ]:




