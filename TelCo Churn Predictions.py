#!/usr/bin/env python
# coding: utf-8

# In[1197]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[1198]:
if not Path("Ressources/Telco-Customer-Churn.csv").is_file():
    with zipfile.ZipFile("Ressources/Telco-Customer-Churn.csv", 'r') as zip_ref:
        zip_ref.extractall("Ressources")

df = pd.read_csv(r"Ressources/Telco-Customer-Churn.csv")


# In[1199]:


df.head()


# In[1200]:


df.info()


# In[1201]:


# Insérez votre code ici
liste_all = []

for value in df['TotalCharges']:
    try:
        float(value)
    except ValueError:
        liste_all.append(value)

unique_values = set(liste_all)
print(unique_values)


# In[1202]:


df.TotalCharges.replace({' ': np.nan}, inplace = True)
df.TotalCharges = df.TotalCharges.astype(float)


# In[1203]:


df.TotalCharges.isna().sum()


# In[1258]:


sns.boxplot('TotalCharges', data = df)
plt.title('Distribution of TotalCharges');


# In[1259]:


sns.boxplot('MonthlyCharges', data = df);
plt.title('Distribution of MonthlyCharges');


# In[1206]:


for i,j in enumerate(df.TotalCharges) :
    df.TotalCharges.fillna(df['MonthlyCharges'][i]* df['tenure'][i], inplace = True)


# In[1207]:


df.describe()


# In[1264]:


plt.pie(df.Churn.value_counts(), labels = ['No','Yes'], autopct='%1.1f%%' )
plt.title('Churn or Not');


# In[1214]:


sns.histplot(x='Churn', hue = 'gender', data = df, multiple = 'dodge');


# In[1215]:


'There are similar churners rate for Male and Female customers'


# In[1217]:


plt.figure(figsize = (10,10))
sns.histplot(x='Churn', hue = 'PaymentMethod', data = df, multiple = 'dodge', shrink = 0.8);


# In[1218]:


'Most of churners pay by electronic check. In fact there are almost as many customers who churn as those who stay.'


# In[1219]:


df.InternetService.value_counts(normalize = True)


# In[1220]:


plt.figure(figsize = (10,10))
sns.histplot(x='Churn', hue = 'InternetService', data = df, multiple="dodge", shrink = .8);


# In[1221]:


'Most churners have a fiber optic InternetService'


# In[1222]:


df.SeniorCitizen.value_counts(normalize = True)


# In[1223]:


plt.figure(figsize = (10,10))
sns.histplot(x='Churn', hue = 'SeniorCitizen', data = df, multiple="dodge", shrink = .8);


# In[1224]:


plt.figure(figsize = (10,10))
sns.histplot(x='SeniorCitizen', hue = 'Churn', data = df, multiple="dodge", shrink = .8);


# In[1225]:


'Even if Senior citizens are not the most seen category, we see a high rate of churners among them. Almost same quantity of churners and stayers.'


# In[1226]:


plt.figure(figsize = (10,10))
sns.histplot(x='Churn', hue = 'MultipleLines', data = df, multiple="dodge", shrink = .8);


# In[1227]:


'Among the churners, we observe less customers without phone service but the proportion of people with or without multiple lines is the same'


# In[1228]:


df.tenure.describe()


# In[1229]:


plt.figure(figsize = (10,10))
sns.histplot(x='Churn', hue = pd.cut(df.tenure, bins = 4, labels = ['q1','q2','q3','q4']), data = df, multiple="dodge", shrink = .8);


# In[1266]:


'Most churners with tenure 0-9 months (duration of staying with the company for customer)'


# In[1268]:


plt.figure(figsize = (10,10))
sns.histplot(x='Churn', hue = 'Partner', data = df, multiple="dodge", shrink = .8)
plt.title('Count of Partners of the company according to churn');


# In[1267]:


'Most churners are not partners'


# In[1269]:


plt.figure(figsize = (10,10))
sns.histplot(x='Churn', hue = 'OnlineSecurity', data = df, multiple="dodge", shrink = .8)
plt.title('Count of different Online Security options according to Churn');


# In[1234]:


'most churners do not have Online Security option'


# In[1235]:


df.columns


# In[1270]:


plt.figure(figsize = (10,10))
sns.histplot(x='Churn', hue = 'Contract', data = df, multiple="dodge", shrink = .8)
plt.title('Count of different Contract types according to churn');


# In[1237]:


'Most churners have month-to-month contract'


# In[1238]:


df.groupby('Contract').median('MonthlyCharges')['MonthlyCharges'].round()


# In[1271]:


series = df.groupby('Contract').median('MonthlyCharges')['MonthlyCharges']
plt.bar(series.index, series.values)
plt.xlabel('Contract')
plt.ylabel('Monthly Charges')
plt.title('Median Monthly Charges by Contract');


# In[1240]:


'We see that median monthly charges of customers with Month to month contract is 4 units more than the one year contract and the one year contract 5 units more than the 2 year contract. There is no much variance between the values of the different categories. '


# In[1241]:


df.groupby('Churn').median('MonthlyCharges')['MonthlyCharges'].round()


# In[1272]:


series = df.groupby('Churn').median('MonthlyCharges')['MonthlyCharges']
plt.bar(series.index, series.values)
plt.xlabel('Churn')
plt.ylabel('Monthly Charges')
plt.title('Median Monthly Charges by Churn');


# In[1243]:


'But the monthly charges of the churners are much higher : 14 units more'


# In[1273]:


plt.figure(figsize = (10,10))
sns.histplot(x='Churn', hue = 'TechSupport', data = df, multiple="dodge", shrink = .8)
plt.title('Count of Tech Support options according to Churn');


# In[1245]:


'For most of the Churners there is TechSupport'


# In[1246]:


'To summarize : Customers tend to leave according to those significant parameters : Contract Month-to-Month, No Tech Support, stay 0 to 9 months with the company, InternetService : Fiber Optic, pay by electronic check, age category : SeniorCitizen, no Online Security. We will now analyse the monthly charges for those categories.'


# In[1274]:


import seaborn as sns

plt.figure(figsize=(10,10))
ax = sns.barplot(x='InternetService', y='MonthlyCharges', data=df, hue = 'Churn')
plt.xlabel('InternetService')
plt.ylabel('MonthlyCharges')
plt.title('Median MonthlyCharges by InternetService by Churn');


# In[1248]:


'We can also see that the charges for InternetService Fiber Optic are much higher than DSL. Which represents the category where there is the highest rate of churners.'


# In[1275]:


import seaborn as sns

plt.figure(figsize=(10,10))
ax = sns.barplot(x='PaymentMethod', y='MonthlyCharges', data=df, hue = 'Churn')
plt.xlabel('PaymentMethod')
plt.ylabel('MonthlyCharges')
plt.title('Mean MonthlyCharges by PaymentMethod by Churn');


# In[1276]:



plt.figure(figsize=(10,10))
ax = sns.barplot(x='TechSupport', y='MonthlyCharges', data=df, hue = 'Churn')
plt.xlabel('TechSupport')
plt.ylabel('MonthlyCharges')
plt.title('Mean MonthlyCharges by TechSupport by Churn');


# In[1251]:


'We observe that the most related features to churns have also the most expensive monthly charges comparing to other values of respective categories.' 


# In[1277]:



plt.figure(figsize=(10,10))
ax = sns.barplot(x='OnlineSecurity', y='MonthlyCharges', data=df, hue = 'Churn')
plt.xlabel('OnlineSecurity')
plt.ylabel('MonthlyCharges')
plt.title('Mean MonthlyCharges by OnlineSecurity by Churn');


# In[1253]:


df.groupby(['Contract', 'TechSupport', 'Churn']).mean('MonthlyCharges')['MonthlyCharges']


# In[1278]:


'Among the most expensive functionalities, we see fiber optic for internet service. And as we saw before, a majority of churners didn"t take the option Tech Support and Online Support. An interpretation could be that customers who took this internet service and no support nor online security were not using well the functionalities and did not use the fiber optic to the fullest. Paying high monthly charges, they did not want to proceed the contract because they did not see their interest.'


# In[1142]:


"Let's now analyse the feature importances regarding the class churner"


# In[1143]:


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


# In[1144]:


df_train.drop('PaymentMethod', axis=1, inplace = True)


# In[1145]:


sns.heatmap(df_train.corr())


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


import scikitplot as skplt
from scikitplot import metrics
metrics.plot_cumulative_gain(y_test, y_probas)


# In[ ]:


import scikitplot as skplt
from scikitplot import metrics
metrics.plot_cumulative_gain(y_test, y_probas1)


# In[ ]:


import scikitplot as skplt
from scikitplot import metrics
metrics.plot_cumulative_gain(y_test, y_probas2)


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




