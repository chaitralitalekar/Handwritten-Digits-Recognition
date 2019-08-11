import numpy as n
import pandas as p
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

dataSet = p.read_csv("dataset/train.csv").as_matrix()
classfy = DecisionTreeClassifier()

train = dataSet[0:21000 , 1:]
train_label = dataSet[0:21000 , 0]

classfy.fit(train,train_label)

test = dataSet[21000: , 1:]
actual_label = data[21000: , 0]

d = test[5]
d.shape = (28,28)
pt.imshow(255-d , cmap = 'gray')

print (clf.predict([test[5]]))

plt.show()

pred = classfy.predict(test)
count = 0
for i in range(0,21000):
    count+=1 if pre[i]==actual_label else 0
print ("Accuracy=",(count/21000)*100)
