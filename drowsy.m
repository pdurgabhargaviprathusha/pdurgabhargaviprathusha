import pandas as pd
from sklearn.neighbors  import KNeighborsClassifier
a=pd.read_csv('drowsy.csv')
feature=pd.read_csv('drowsy.csv',usecols=[0,1,2,3])
label=pd.read_csv('drowsy.csv',usecols=[4])
b=KNeighborsClassifier(n_neighbors=2)
b.fit(feature,label)
print(b.predict([[0.3,10,1,9]]))
print(b.score(feature,label))
