#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns  
import matplotlib.pyplot as plt
import time
from subprocess import check_output

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import wittgenstein as lw
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# ## 1. Data

# In[2]:


data = pd.read_csv('data.csv')

data.head() 


# ## 2. Data Preparation

# In[4]:


data.describe().T


# In[ ]:


data.describe().T.to_csv("description.csv")


# In[ ]:


print(data.isnull().sum())


# In[5]:


# y includes target feature
y = data.price_range
x = data.drop(columns=['price_range'])

x.head()


# ## 3. Data Visualization

# In[ ]:


ax = sns.countplot(y, label="Count", palette = 'hls')

range0, range1, range2, range3 = y.value_counts()

print('Number of range0: ', range0)
print('Number of range1: ', range1)
print('Number of range2: ', range2)
print('Number of range3: ', range3)


# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Price Range vs all numerical factor')

sns.countplot(ax=axes[0, 0], data=data, x='three_g',palette='BuPu')
sns.countplot(ax=axes[0, 1], data=data, x='touch_screen',palette='BuPu')
sns.countplot(ax=axes[0, 2], data=data, x='four_g',palette='BuPu')
sns.countplot(ax=axes[1, 0], data=data, x='wifi',palette='BuPu')
sns.countplot(ax=axes[1,1], data=data, x ='fc' ,palette='BuPu')
sns.countplot(ax=axes[1,2], data=data, x ='dual_sim',palette='BuPu' )
plt.show()


# In[ ]:


categorical_feature = [feature for feature in data.columns if len(data[feature].unique()) < 3]

plt.figure(figsize=(20, 10))
count = 0
for feature in categorical_feature:
    data = data.copy()
    explode = [0.2, 0]

    labels = data[feature].value_counts().index
    sizes = data[feature].value_counts().values

    plt.subplot(2, 3, count+1)
    plt.pie(sizes, labels=labels, explode=explode, startangle=90, autopct='%1.1f%%')
    plt.title(f'Distribution of {categorical_feature[count]}', color='black', fontsize=15)
    count += 1


# In[ ]:


sns.lmplot(x='ram', y='price_range', data=data, line_kws={'color':'black'})
plt.yticks([0, 1, 2, 3])
plt.show()


# In[ ]:


plt.figure(figsize=(10, 10))
sns.scatterplot(data=data, x='battery_power', y='ram', hue='price_range',markers=['8', 'p'], s=60)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='price_range')


# ## 4. Analysis with all variables
# 
# ### 4.1 KNN

# In[6]:


scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(x)

#showing data
print('X \n' , x[:10])
print('y \n' , y[:10])


# In[7]:


SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

print('Dimension of x: ',X_train.shape, X_test.shape)
print('Dimension of y: ',y_train.shape, y_test.shape)


# In[8]:


error_rate = []

for i in range(50,100):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train.ravel()) #.ravel() converts the column vector into a row vector (1d array). warning without this and takes a lot of time. 
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(50,100),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()


# In[9]:


# KNN

knn_classifier = KNeighborsClassifier(n_neighbors=77)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

knn_acc =  knn_classifier.score(X_test, y_test)
print('KNN accuracy: ', knn_acc) #0.5433333333333333


# In[10]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu");


# ### 4.2 SVM

# In[11]:


# SVM

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

svm_acc =  svclassifier.score(X_test, y_test)
print(svm_acc) #0.9733333333333334


# In[13]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu");


# ### 4.3 Classification Trees

# In[14]:


tree_classifier = DecisionTreeClassifier(random_state=SEED)
tree_classifier.fit(X_train, y_train)

y_pred = tree_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

tree_acc =  tree_classifier.score(X_test, y_test)
print(tree_acc) #0.8183333333333334


# In[15]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu");


# ### 4.4 Artificial Neural Network

# In[ ]:


y.head()


# In[16]:


y_ann = data['price_range'].values
y_ann


# In[17]:


X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(x, y_ann, test_size=0.30, random_state=SEED)

print('Dimension of x: ',X_train_ann.shape, X_test_ann.shape)
print('Dimension of y: ',y_train_ann.shape, y_test_ann.shape)


# In[18]:


mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, random_state=SEED)
mlp.fit(X_train_ann, y_train_ann)

predict_train = mlp.predict(X_train_ann)
predict_test = mlp.predict(X_test_ann)

print(confusion_matrix(y_train_ann,predict_train))
print(classification_report(y_train_ann,predict_train))

ann_acc =  mlp.score(X_test_ann, y_test_ann)
print(ann_acc)    #0.6016666666666667


# In[21]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_train_ann,predict_train), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu");


# In[22]:


# Creting a table to compare the algo

allvar_acc = [knn_acc, svm_acc, tree_acc, ann_acc]

acc_df = pd.DataFrame(allvar_acc, 
                      columns = ['All Variables'],
                      index = ['KNN', 'SVM', 'CL Tree', 'ANN']) 
acc_df


# In[26]:


acc_df.to_csv("1acc_df.csv")


# In[23]:


acc_df.plot(kind="bar", figsize=(10, 5))

plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")


# In[ ]:





# In[ ]:





# ## 5. Univariate Filter Feature Subset Selection

# ### 5.1 Chi-squared

# In[27]:


BestFeatures = SelectKBest(score_func=chi2, k=5)
fit = BestFeatures.fit(X, y)

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(x.columns)

#concatenating two dataframes for better visualization
f_Scores = pd.concat([df_columns,df_scores],axis=1)      
f_Scores.columns = ['Specs','Score']  

print(f_Scores.nlargest(5,'Score'))      

X_kbest = BestFeatures.fit_transform(X, y)

print()
print('Original number of features:', X.shape)
print('Reduced number of features:', X_kbest.shape)


# In[28]:


X_kbest_norm = scaler.fit_transform(X_kbest)

#showing data
print('X \n' , x[:10])
print('y \n' , y[:10])


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X_kbest_norm, y, test_size=0.3, random_state=SEED)

print('Dimension of x: ',X_train.shape, X_test.shape)
print('Dimension of y: ',y_train.shape, y_test.shape)


# In[30]:


# KNN
# n = 21, accu = 0.8666666666666667

chi_knn_classifier = KNeighborsClassifier(n_neighbors=21)
chi_knn_classifier.fit(X_train, y_train)

y_pred = chi_knn_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

chi_knn_acc =  chi_knn_classifier.score(X_test, y_test)
print(chi_knn_acc)


# In[32]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="BuPu");


# In[33]:


# SVM
# chi = 7, accu = 0.986

chi_svclassifier = SVC(kernel='linear')
chi_svclassifier.fit(X_train, y_train)

y_pred = chi_svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

chi_svm_acc =  chi_svclassifier.score(X_test, y_test)
print(chi_svm_acc)


# In[34]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="BuPu");


# In[35]:


# Classification Tree
# chi = 7, accu = 0.828

chi_tree_classifier = DecisionTreeClassifier(random_state=SEED)
chi_tree_classifier.fit(X_train, y_train)

y_pred = chi_tree_classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

chi_tree_acc =  chi_tree_classifier.score(X_test, y_test)
print(chi_tree_acc)


# In[36]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="BuPu");


# In[37]:


# ANN
# chi = 7, accu = 0.648

X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(X_kbest_norm, y_ann, test_size=0.30, random_state=SEED)

chi_mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, random_state=SEED)
chi_mlp.fit(X_train_ann, y_train_ann)

predict_train = chi_mlp.predict(X_train_ann)
predict_test = chi_mlp.predict(X_test_ann)

print(confusion_matrix(y_train_ann,predict_train))
print(classification_report(y_train_ann,predict_train))

chi_ann_acc =  chi_mlp.score(X_test_ann, y_test_ann)
print(chi_ann_acc)


# In[39]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_train_ann,predict_train), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="BuPu");


# In[40]:


# Creting a table to compare the algo

chi_acc = [chi_knn_acc, chi_svm_acc, chi_tree_acc, chi_ann_acc]
acc_df['ChiSquared'] = chi_acc
acc_df


# In[43]:


acc_df.to_csv("2acc_df.csv")


# In[41]:


acc_df.plot(kind="bar", figsize=(10, 5))

plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:





# ## 6. Multivariate Filter Feature Subset Selection

# ### 6.1 Correlation Matrix

# In[44]:


corrmat = x.corr(method='spearman')
f, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)


# In[45]:


to_drop = ['fc','three_g', 'sc_w', 'px_width']
x_nocorr = x.drop(to_drop, axis=1)
x_nocorr.head()


# In[46]:


X_nocorr = scaler.fit_transform(x_nocorr)

X_train, X_test, y_train, y_test = train_test_split(X_nocorr, y, test_size=0.3, random_state=SEED)

print('Dimension of x: ',X_train.shape, X_test.shape)
print('Dimension of y: ',y_train.shape, y_test.shape)


# In[47]:


# KNN

nocorr_knn_classifier = KNeighborsClassifier(n_neighbors=77)
nocorr_knn_classifier.fit(X_train, y_train)

y_pred = nocorr_knn_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

nocorr_knn_acc =  nocorr_knn_classifier.score(X_test, y_test)
print(nocorr_knn_acc)


# In[48]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="Greens");


# In[49]:


# SVM

nocorr_svclassifier = SVC(kernel='linear')
nocorr_svclassifier.fit(X_train, y_train)

y_pred = nocorr_svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

nocorr_svm_acc =  nocorr_svclassifier.score(X_test, y_test)
print(nocorr_svm_acc)


# In[50]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="Greens");


# In[51]:


# Classification Tree

nocorr_tree_classifier = DecisionTreeClassifier(random_state=SEED)
nocorr_tree_classifier.fit(X_train, y_train)

y_pred = nocorr_tree_classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

nocorr_tree_acc =  nocorr_tree_classifier.score(X_test, y_test)
print(nocorr_tree_acc)


# In[52]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="Greens");


# In[53]:


# ANN

X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(x_nocorr, y_ann, test_size=0.30, random_state=SEED)

nocorr_mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, random_state=SEED)
nocorr_mlp.fit(X_train_ann, y_train_ann)

predict_train = nocorr_mlp.predict(X_train_ann)
predict_test = nocorr_mlp.predict(X_test_ann)

print(confusion_matrix(y_train_ann,predict_train))
print(classification_report(y_train_ann,predict_train))

nocorr_ann_acc =  nocorr_mlp.score(X_test_ann, y_test_ann)
print(nocorr_ann_acc)


# In[54]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_train_ann,predict_train), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="Greens");


# In[55]:


corr_acc = [nocorr_knn_acc, nocorr_svm_acc, nocorr_tree_acc, nocorr_ann_acc]
acc_df['Matrix Correlation'] = corr_acc
acc_df


# In[56]:


acc_df.to_csv("3acc_df.csv")


# In[57]:


acc_df.plot(kind="bar", figsize=(10, 5))

plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:





# ## 7. Multivariate Wrapper Feature Subset Selection

# In[58]:


knn_sfs = SFS(knn_classifier, 
           k_features=5, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           n_jobs=-1,
           cv=0)

knn_sfs_fit = knn_sfs.fit(X, y)
pd.DataFrame.from_dict(knn_sfs_fit.get_metric_dict())


# In[59]:


knn_x = x.iloc[:,list(knn_sfs_fit.k_feature_idx_)]
knn_x.head()


# In[66]:


scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
knn_x = scaler.fit_transform(knn_x)

X_train, X_test, y_train, y_test = train_test_split(knn_x, y, test_size=0.3, random_state=SEED)

knn_classifier = KNeighborsClassifier(n_neighbors=77)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

sfs_knn_acc =  knn_classifier.score(X_test, y_test)
print('KNN accuracy: ', sfs_knn_acc) #0.9183333333333333


# In[67]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues");


# In[ ]:





# In[77]:


svc_sfs = SFS(svclassifier, 
           k_features=5, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           n_jobs=-1,
           cv=0)

svc_sfs_fit = svc_sfs.fit(x, y)

svc_x = x.iloc[:,list(svc_sfs_fit.k_feature_idx_)]
svc_x.head()


# In[78]:


# SVM

X_train, X_test, y_train, y_test = train_test_split(svc_x, y, test_size=0.3, random_state=SEED)

sfs_svclassifier = SVC(kernel='linear')
sfs_svclassifier.fit(X_train, y_train)

y_pred = sfs_svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

sfs_svm_acc =  sfs_svclassifier.score(X_test, y_test)
print(sfs_svm_acc)


# In[79]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues");


# In[61]:


tree_sfs = SFS(tree_classifier, 
           k_features=5, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           n_jobs=-1,
           cv=0)

tree_sfs_fit = tree_sfs.fit(X, y)
pd.DataFrame.from_dict(tree_sfs_fit.get_metric_dict())


# In[62]:


tree_x = x.iloc[:,list(tree_sfs_fit.k_feature_idx_)]
tree_x.head()


# In[68]:


scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
tree_x = scaler.fit_transform(tree_x)

X_train, X_test, y_train, y_test = train_test_split(tree_x, y, test_size=0.3, random_state=SEED)

tree_classifier = DecisionTreeClassifier(random_state=SEED)
tree_classifier.fit(X_train, y_train)

y_pred = tree_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

sfs_tree_acc =  tree_classifier.score(X_test, y_test)
print(sfs_tree_acc) #0.7183333333333334


# In[69]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues");


# In[64]:


ann_sfs = SFS(mlp, 
           k_features=5, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           n_jobs=-1,
           cv=0)

ann_sfs_fit = ann_sfs.fit(X, y)
pd.DataFrame.from_dict(ann_sfs_fit.get_metric_dict())


# In[70]:


ann_x = x.iloc[:,list(tree_sfs_fit.k_feature_idx_)]
ann_x.head()


# In[71]:


scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
ann_x = scaler.fit_transform(ann_x)

X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(ann_x, y_ann, test_size=0.30, random_state=SEED)

ann_sfs_mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, random_state=SEED)
ann_sfs_mlp.fit(X_train_ann, y_train_ann)

predict_train = ann_sfs_mlp.predict(X_train_ann)
predict_test = ann_sfs_mlp.predict(X_test_ann)

print(confusion_matrix(y_train_ann,predict_train))
print(classification_report(y_train_ann,predict_train))

sfs_ann_acc =  ann_sfs_mlp.score(X_test_ann, y_test_ann)
print(sfs_ann_acc)


# In[72]:


classes_names = ['range0', 'range1', 'range2', 'range3']

cm = pd.DataFrame(confusion_matrix(y_train_ann,predict_train), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues");


# In[73]:


sfs_svm_acc = 0.9733333333333334


# In[74]:


sfs_acc = [sfs_knn_acc, sfs_svm_acc, sfs_tree_acc, sfs_ann_acc]
acc_df['SFS Method'] = sfs_acc
acc_df


# In[76]:


acc_df.plot(kind="bar", figsize=(10, 5))

plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

