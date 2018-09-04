import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df_train = pd.read_csv('C:/Users/sheruaravindreddy/Downloads/Kaggle-data-set/train.csv')
df_train = df_train.apply(le.fit_transform)

Y_train = df_train[['Survived']]
X_train = df_train.drop(columns = ['Survived'])
#...Importing the testing dataset
X_test = pd.read_csv('C:/Users/sheruaravindreddy/Downloads/Kaggle-data-set/test.csv')
X_test = X_test.apply(le.fit_transform)


X_test_actual_output = pd.read_csv('C:/Users/sheruaravindreddy/Downloads/Kaggle-data-set/gender_submission.csv')["Survived"]

def fit_clf(X_train, Y_train, X_test, X_test_actual_output,k):
    clf = KNN(n_neighbors = k)
    clf.fit(X_train,Y_train.values.ravel())
    X_test_clf_output =  clf.predict(X_test)
    
    if k == 5:
        print clf.kneighbors(X_test)
        print clf.kneighbors_graph(X_test)
        
    length = len(X_test_actual_output)
    count  = 0.0
    for i in range(length):
        if X_test_clf_output[i] != X_test_actual_output[i]:
            count += 1
    
    print count/length
    
for k in range(1,40):
    fit_clf(X_train, Y_train, X_test, X_test_actual_output,k)