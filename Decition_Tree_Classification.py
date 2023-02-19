import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import random

def glass_dataset():
    glass_data = pd.read_csv("glass.csv")
    # glass_data.head()
    all_inputs = glass_data[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']].values
    all_classes = glass_data['Type'].values
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = train_test_split(all_inputs, all_classes, test_size=0.3, random_state=3, shuffle=True)
    return training_inputs, training_classes, testing_inputs, testing_classes


def tictac_dataset():
    data = pd.read_csv("tic-tac-toe.data.txt", sep=",")
    data_copy = pd.read_csv("tic-tac-toe.data.txt", sep=",")
    data.columns = ["first_row_left", "first_row_middle", "first_row_right", "center_row_left", "center_row_middle",
                    "center_row_right", "bottom_row_left", "bottom_row_middle", "bottom_row_right", "is_win"]
    data_copy.columns = ["first_row_left", "first_row_middle", "first_row_right", "center_row_left",
                         "center_row_middle",
                         "center_row_right", "bottom_row_left", "bottom_row_middle", "bottom_row_right", "is_win"]
    mapping_for_moves = {'x': 1, "o": 0}  # For b, we put mean of the data.
    mapping_for_wins = {"positive": 1, "negative": 0}  # Positive is win, negative is lose
    data.is_win = data.is_win.map(mapping_for_wins)
    data_copy.is_win = data_copy.is_win.map(mapping_for_wins)
    data = data.drop(columns=["is_win"], axis=1)
    for i in data.columns:  # Applying map to all the columns except is_win.
        data[i] = data[i].map(mapping_for_moves)
    features = data.values
    labels = data_copy.is_win.values
    features = (SimpleImputer().fit_transform(features))
    features = features.astype(np.int)
    labels = labels.astype(np.int)
    training_inputs, testing_inputs, training_classes, testing_classes = train_test_split(features, labels,
                                                                                          random_state=3, shuffle=True)
    return training_inputs, training_classes, testing_inputs, testing_classes


def DecitionTree(training_inputs, training_classes, testing_inputs, testing_classes):
    dtc = DTC()
    dtc.fit(training_inputs, training_classes)
    score = dtc.score(testing_inputs, testing_classes)
    # print("score:", score)
    score = score * 100
    if (len(training_inputs) > 400):
        print('Accuracy on tic_tac_toe dataset: %.2f' % score)
        return score
    elif (len(training_inputs) < 400):
        print('Accuracy on glass dataset: %.2f' % score)
        return score



def attribute_noise(data, percent):
    for i in range(data.shape[1]):
        data_noise=np.random.normal(0,((np.var(data[:,i]))*percent),len(data[:,i]))
        data[:,i]=data[:,i]+data_noise
    return data


def add_noise(data):
    copy_data=data
    data1=attribute_noise(copy_data, 0.05)
    data2=attribute_noise(copy_data, 0.10)
    data3=attribute_noise(copy_data, 0.15)

    return data1,data2,data3

CvsC=[]
DvsC=[]
CvsD=[]
DvsD=[]


xlabel=[5,10,15]
# CvsC
print("On CvsC Result Is:")
for i in range(3):
    training_inputs, training_classes, testing_inputs, testing_classes = tictac_dataset()
    CvsC.append(DecitionTree(training_inputs, training_classes, testing_inputs, testing_classes))

# CvsD
print("On CvsD Result Is:")
t1,t2,t3=add_noise(testing_inputs)
print("5%")
CvsD.append(DecitionTree(training_inputs, training_classes, t1, testing_classes))
print("10%")
CvsD.append(DecitionTree(training_inputs, training_classes, t2, testing_classes))
print("15%")
CvsD.append(DecitionTree(training_inputs, training_classes, t3, testing_classes))

# DvsC
print("On DvsC Result Is:")
t1,t2,t3=add_noise(training_inputs)
print("5%")
DvsC.append(DecitionTree(t1, training_classes, testing_inputs, testing_classes))
print("10%")
DvsC.append(DecitionTree(t2, training_classes, testing_inputs, testing_classes))
print("15%")
DvsC.append(DecitionTree(t3, training_classes, testing_inputs, testing_classes))

# DvsD
print("On DvsD Result Is:")
t1,t2,t3=add_noise(training_inputs)
tt1,tt2,tt3=add_noise(testing_inputs)
print("5%")
DvsD.append(DecitionTree(t1, training_classes, tt1, testing_classes))
print("10%")
DvsD.append(DecitionTree(t2, training_classes, tt2, testing_classes))
print("15%")
DvsD.append(DecitionTree(t3, training_classes, tt3, testing_classes))


def label_noise_contradictory(X_train,y_train,percent):
    new_X_train=X_train
    new_y_train=y_train
    range_data=int(percent*len(new_y_train))
    for i in range(range_data):
        rand=random.randint(0,len(new_y_train)-1)
        data_noise=new_X_train[rand]
        numbers = [0,1]
        numbers.remove(new_y_train[rand])
        label_noise = random.choice(numbers)
        data_noise = np.array(data_noise).reshape((1,9))
        new_X_train=np.concatenate((new_X_train,data_noise),axis=0)
        new_y_train=np.concatenate((new_y_train,label_noise),axis=None)
        #y_train=np.append(y_train,label_noise)
    return new_X_train,new_y_train


def label_noise_missclassification(y_train,percent):
    new_y_train = y_train
    range_data=int(percent*len(new_y_train))
    for i in range(range_data):
        rand=random.randint(0,len(new_y_train)-1)
        numbers = [0,1]
        numbers.remove(new_y_train[rand])
        new_y_train[rand]=random.choice(numbers)
    return new_y_train


contradic=[]
missclass=[]
print('On Attribute Contradictory Noise Result Is:')
t1,y1=label_noise_contradictory(training_inputs,training_classes,0.05)
t2,y2=label_noise_contradictory(training_inputs,training_classes,0.10)
t3,y3=label_noise_contradictory(training_inputs,training_classes,0.15)

print("5%")
contradic.append(DecitionTree(t1, y1, testing_inputs, testing_classes))
print("10%")
contradic.append(DecitionTree(t2, y2, testing_inputs, testing_classes))
print("15%")
contradic.append(DecitionTree(t3, y3, testing_inputs, testing_classes))

print('On Attribute MissClassification Noise Result Is:')
y1=label_noise_missclassification(training_classes,0.05)
y2=label_noise_missclassification(training_classes,0.10)
y3=label_noise_missclassification(training_classes,0.15)

print("5%")
missclass.append(DecitionTree(training_inputs, y1, testing_inputs, testing_classes))
print("10%")
missclass.append(DecitionTree(training_inputs, y2, testing_inputs, testing_classes))
print("15%")
missclass.append(DecitionTree(training_inputs, y3, testing_inputs, testing_classes))



plt.plot(xlabel,CvsC,marker='*',linestyle='-',color='g',label='CvsC')
plt.plot(xlabel,CvsD,marker='*',linestyle='-',color='b',label='CvsD')
plt.plot(xlabel,DvsC,marker='*',linestyle='-',color='y',label='DvsC')
plt.plot(xlabel,DvsD,marker='*',linestyle='-',color='r',label='DvsD')
plt.xlabel('Noise Level')
plt.ylabel('Accuracy Rate')
plt.legend(loc='best')
plt.title('Section B - Tic-Tac_Toe  ')


plt.figure()
plt.plot(xlabel,contradic,marker='*',linestyle='-',color='g',label='missclass')
plt.plot(xlabel,missclass,marker='*',linestyle='-',color='y',label='contradic')
plt.xlabel('Noise Level')
plt.ylabel('Accuracy Rate')
plt.legend(loc='best')
plt.title('Section C - Tic-Tac_Toe ')
plt.show()








