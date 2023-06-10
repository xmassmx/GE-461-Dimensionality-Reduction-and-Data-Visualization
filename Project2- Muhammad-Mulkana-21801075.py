

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE




'''load data, split, and scale '''
digits = scipy.io.loadmat('digits.mat')
data = digits['digits']
labels = digits['labels']

df = pd.DataFrame(np.append(data, labels, axis = 1))
train_df, test_df = train_test_split(df, test_size=0.5, stratify = df[[400]])
train_data = train_df.drop(400, axis = 1).to_numpy()
train_labels = train_df[400].to_numpy()
test_data = test_df.drop(400, axis = 1).to_numpy()
test_labels = test_df[400].to_numpy()
train_scaler = preprocessing.StandardScaler()
train_scaler.fit(train_data)
train_data_norm = train_scaler.transform(train_data)
test_scaler = preprocessing.StandardScaler()
test_scaler.fit(test_data)
test_data_norm = test_scaler.transform(test_data)


''' PCA'''
pca = PCA(n_components=400) 
pca.fit(train_data_norm) 
plt.plot(pca.explained_variance_)
plt.title('Explained Variance vs Principal Components') 
plt.xlabel('Principal Components') 
plt.ylabel('Explained Variance')
plt.grid()
plt.show()


plt.plot(np.cumsum(pca.explained_variance_ratio_)*100)
plt.title('Explained Variance Ratio vs Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio (%)')
plt.grid()
plt.show()
plt.imshow(np.reshape(np.ravel(train_scaler.mean_, order='F'), (20, 20), order='F'), cmap='gray')
plt.axis('off')
k_chosen = 81
pca_100 = PCA(n_components=k_chosen)
pca_100.fit(train_data_norm)
fig = plt.figure(figsize=(15, 10))
plt.title('First ' + str(k_chosen) + ' eigenvectors displayed as 20x20 images')
plt.axis('off')

for i in range(k_chosen):
  fig.add_subplot(9, 9, i+1) 
  plt.imshow(np.reshape(np.ravel(pca_100.components_[i], order='F'), (20, 20), order='F'), cmap = 'gray')
  plt.axis('off')

plt.show()
accuracy_train = []
accuracy_test = []


accuracy_train = []
accuracy_test = []



for k in range(1,200, 5):
  pca = PCA(n_components=k)
  pca.fit(train_data_norm)
  train_data_pca = pca.transform(train_data_norm)
  test_data_pca = pca.transform(test_data_norm)

  qda = QuadraticDiscriminantAnalysis()
  qda.fit(train_data_pca, train_labels.ravel())

  qda_pred_train = qda.predict(train_data_pca)
  qda_acc_train = metrics.accuracy_score(train_labels.ravel(), qda_pred_train.ravel())

  qda_pred_test = qda.predict(test_data_pca)
  qda_acc_test = metrics.accuracy_score(test_labels.ravel(), qda_pred_test.ravel())

  print('-------')
  print('k: ' + str(k))
  print('Train error: ' + str(1-qda_acc_train))
  print('Test error: ' + str(1-qda_acc_test))

  accuracy_train.append((1-qda_acc_train)*100)
  accuracy_test.append((1-qda_acc_test)*100)

plt.plot(np.arange(1,200, 5),accuracy_train)
plt.title('Train Error vs PCA dimensions')
plt.xlabel('PCA dimensions')
plt.ylabel('Error %')
plt.show()

plt.plot(np.arange(1,200, 5),accuracy_test)
plt.title('Test Error vs PCA dimensions')
plt.xlabel('PCA dimensions')
plt.ylabel('Error %')
plt.show()



accuracy_train = []
accuracy_test = []



''' LDA '''
lda = LinearDiscriminantAnalysis()
lda.fit(train_data_norm, train_labels.ravel())


fig = plt.figure(figsize=(15, 10))

for i in range(9):
  fig.add_subplot(2, 5, i+1 )
  plt.imshow(np.reshape(np.ravel(lda.scalings_.T[i], order='F'), (20, 20), order='F'), cmap = 'gray')
  plt.axis('off')
  
plt.show()
  
  
  
  
  
  
  
accuracy_train = []
accuracy_test = []
accuracy_test1 = []

for k in range(1,10):
  lda = LinearDiscriminantAnalysis(n_components=k)
  lda.fit(train_data_norm, train_labels.ravel())
  train_data_lda = lda.transform(train_data_norm)
  test_data_lda = lda.transform(test_data_norm)

  qda = QuadraticDiscriminantAnalysis()
  qda.fit(train_data_lda, train_labels.ravel())

  qda_pred_train = qda.predict(train_data_lda)
  qda_acc_train = metrics.accuracy_score(train_labels.ravel(), qda_pred_train.ravel())

  qda_pred_test = qda.predict(test_data_lda)
  qda_acc_test = metrics.accuracy_score(test_labels.ravel(), qda_pred_test.ravel())


  print('-------')
  print('k: ' + str(k))
  print('Train error: ' + str(1-qda_acc_train))
  print('Test error: ' + str(1-qda_acc_test))

  accuracy_train.append((1-qda_acc_train)*100)
  accuracy_test.append((1-qda_acc_test)*100)

plt.plot(np.arange(1,10),accuracy_train)
plt.title('Train Error vs LDA dimensions')
plt.xlabel('LDA dimensions')
plt.ylabel('Error %')
plt.show()

plt.plot(np.arange(1,10),accuracy_test)
plt.title('Test Error vs LDA dimensions')
plt.xlabel('LDA dimensions')
plt.ylabel('Error %')
plt.show()


data_scaler = preprocessing.StandardScaler()
data_scaler.fit(data)
data_norm = data_scaler.transform(data)

tsne = TSNE(n_components=2, random_state= 0,init='random')
data_tsne = tsne.fit_transform(data_norm)
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html


fig = plt.figure(figsize=(12, 10))


for classes in range(10):
  label_ind = np.where(labels == classes)[0]
  data_class = data_tsne[label_ind,:]
  plt.scatter(data_class[:,0] , data_class[:,1] , label = classes, s=10)

plt.legend()
plt.show()



from sammon.sammon import sammon # first git repo is cloned. reference given in report

[y, E] = sammon(data_norm, 2, display = 2, inputdist = 'raw', maxhalves = 50, maxiter = 500, tolfun = 1e-9, init = 'default')
# https://data-farmers.github.io/2019-06-10-sammon-mapping/

fig = plt.figure(figsize=(12, 10))


for classes in range(10):
  label_ind = np.where(train_labels == classes)[0]
  data_class = y[label_ind,:]
  plt.scatter(data_class[:,0] , data_class[:,1] , label = classes, s=10)
plt.legend()
plt.show()




