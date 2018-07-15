import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
data=pd.read_csv('C:\\Users\\User\\Desktop\\mnist.csv')
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.1, random_state=4)
#print(x_test)
#print(y_test)
nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(10,15),random_state=1)
nn.fit(x_train,y_train)
pred=nn.predict(x_test)
a=y_test.values
#print(images_array)
count = 0
for i in range(len(pred)):
    if pred[i]==a[i]:
        count = count+1
accuracy = ((count)/(len(pred)))
print("The accuracy is: ", accuracy)
##now lets actually try and test it out:
while 1==1:
    n = int(input("Please input the value of the array that you woruld like to test: "))
    test_array_1 = df_x.iloc[n,:]
    test_label = df_y.iloc[n]
    test_array = test_array_1.as_matrix()
    pixels = test_array.reshape((28,28))
    pred_x =nn.predict([test_array])
    print(pred_x)
    plt.imshow(pixels, cmap='gray')
    plt.show()

