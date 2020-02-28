import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from sklearn import datasets
from tensorflow.contrib.learn.python import SKCompat
from pandas import DataFrame
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# add layer
def add_hidden_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random.normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_biases = tf.matmul(inputs,weights) + biases
    if activation_function is None:
        outputs = Wx_plus_biases
    else:
        outputs = activation_function(Wx_plus_biases)
    return outputs

#Neurql network with no hidden layer
def NN_no_hidden_layer(x_data, y_data):
    weights = tf.Variable(1.0, name="w")
    noise = np.random.normal(0,0.05, x_data.shape)
    y = weights*x_data**3+weights*x_data**2-weights*x_data-1+noise
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_data), reduction_indices=1))
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for step in range(5000):
        sess.run(train)
        prediction_value = sess.run(weights) * x_data**3 + sess.run(weights)*x_data**2 - sess.run(weights)*x_data-1+noise
        # if step % 100 == 0:
            # print("loss: ", sess.run(loss))
    return prediction_value

#Neurql network with one hidden layer
def NN_one_hidden_layer(x_data, y_data, num_of_nerve):
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
    h1 = add_hidden_layer(xs, 1, num_of_nerve, activation_function=tf.nn.relu)
    prediction = add_hidden_layer(h1, num_of_nerve, 1, activation_function=None)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=1))
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for step in range(5000):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
    return prediction_value

#Neurql network with two hidden layer
def NN_two_hidden_layer(x_data, y_data, num_of_nerve):
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
    h1 = add_hidden_layer(xs, 1, num_of_nerve, activation_function=tf.nn.relu)
    h2 = add_hidden_layer(h1, num_of_nerve, num_of_nerve, activation_function=tf.nn.relu)
    prediction = add_hidden_layer(h2, num_of_nerve, 1, activation_function=None)

    # create tensorflow structure
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=1))
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for step in range(5000):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # if step%100 == 0:
        #     print("loss: ", sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    return prediction_value

def make_circke_data():
    circles_data, circles_data_labels = datasets.make_circles(n_samples=500, factor=0.1, noise=0.1)
    return circles_data,circles_data_labels

def classify_circle_data(circles_data, circles_data_labels):
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=2)
    DNN_classifier = classifier.fit(circles_data, circles_data_labels, steps=2000)
    return DNN_classifier

#classify's accuracy score of DNNClassifier
def classify_accuracy_score():
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=2)
    accuracy_score = classifier.evaluate(circles_data, circles_data_labels, steps=1)["accuracy"]
    return accuracy_score

def main():
    # part1
    # 利用tensorflow
    # 撰寫一個多層的神經網路去模擬一個函數產生器。請比較採用不同層級、不同神經元個數所達到的模擬效
    # 果。並請將實際及模擬的結果顯示在圖形上。
    # create data
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = x_data ** 3 + x_data ** 2 - x_data - 1
    y_data_noise = y_data + noise

    # print('prediction_value:' + str(prediction_value))
    # print(prediction_value.shape)
    # print('x_data:' + str(x_data))
    # print('y_data:' + str(y_data))
    # print('x_data.shape:' + str(x_data.shape))
    # print('y_data.shape:' + str(y_data.shape))

    #plot scatter
    fig = plt.figure(figsize=(10, 8))
    bx = fig.add_subplot(1, 1, 1)
    plt.xlabel('x_data')
    plt.ylabel('y_data')
    prediction_no_hidden = NN_no_hidden_layer(x_data, y_data_noise)
    prediction_one_hidden = NN_one_hidden_layer(x_data, y_data_noise, 20)
    prediction_two_hidden = NN_two_hidden_layer(x_data, y_data_noise, 20)
    prediction_one_hidden_less_nerve = NN_one_hidden_layer(x_data, y_data_noise, 5)
    prediction_two_hidden_less_nerve = NN_two_hidden_layer(x_data, y_data_noise, 5)
    bx.scatter(x_data, y_data, color='black') #原資料的折線圖
    bx.scatter(x_data, prediction_no_hidden, color='grey') #沒有隱藏層的圖用灰色線表示
    bx.scatter(x_data, prediction_one_hidden, color='darkgreen') #一個隱藏層用深綠線表示
    bx.scatter(x_data, prediction_two_hidden, color='darkred') #兩個隱藏層用深紅線表示
    bx.scatter(x_data, prediction_one_hidden_less_nerve, color='lightgreen') #一個隱藏層較少神經元用萊姆綠表示
    bx.scatter(x_data, prediction_two_hidden_less_nerve, color='pink') #兩個隱藏層較少神經元用粉紅表示
    plt.show()

    # part2
    # 請利用tensorflow撰寫一個神經網路針對產生的資料集做分類。
    # 將此一神經網路所判斷不同類別的區域分別塗上不同顏色。並且將資料集的資料也標示於圖上
    circles_data, circles_data_labels = make_circke_data()
    df = DataFrame(dict(x=circles_data[:, 0], y=circles_data[:, 1], label=circles_data_labels))
    DNN_classifier = classify_circle_data(circles_data, circles_data_labels)
    classifier_result = list(DNN_classifier.predict(circles_data))
    x_min, x_max = circles_data[circles_data_labels == 0, 0].min() - 0.2, circles_data[circles_data_labels == 0, 0].max() + 0.2
    y_min, y_max = circles_data[circles_data_labels == 0, 1].min() - 0.2, circles_data[circles_data_labels == 0, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))
    Z = list(DNN_classifier.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z=np.array(Z)

    X0,X1 = circles_data[circles_data_labels == 0],circles_data[circles_data_labels == 1]
    tp = (circles_data_labels == classifier_result)
    tp0, tp1 = tp[circles_data_labels == 0],tp[circles_data_labels == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    cm_bright=ListedColormap(['pink','#0D8ECF'])
    Z=Z.reshape(xx.shape)
    plt.pcolormesh(xx,yy,Z,cmap=cm_bright)
    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
                s=20, color='#FFF3EE')  # dark red

    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
    plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
                s=20, color='#842B00')  # dark blue
    plt.show()
main()

#pytorch
# circles_data, circles_data_labels = datasets.make_circles(n_samples=50, factor=0.5, noise=0.1)
# x0,x1 = torch.from_numpy(circles_data[circles_data_labels == 0]),torch.from_numpy(circles_data[circles_data_labels == 1])
# y0 = torch.from_numpy(circles_data_labels[circles_data_labels==0])
# y1 = torch.from_numpy(circles_data_labels[circles_data_labels==1])
# x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
# y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer
#
# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
#         self.out = torch.nn.Linear(n_hidden, n_output)   # output layer
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))      # activation function for hidden layer
#         x = self.out(x)
#         return x
#
# net = Net(2,10,2)
#
# optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
#
# for t in range(10000):
#     out = net(x)                 # input x and predict based on x
#     loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
#
#     optimizer.zero_grad()   # clear gradients for next train
#     loss.backward()         # backpropagation, compute gradients
#     optimizer.step()        # apply gradients
    # if t % 2 == 0:
        # plot and show learning process

# clf = LDA()
# clf.fit(circles_data, circles_data_labels)
#
# x_min, x_max = circles_data[circles_data_labels==0, 0].min()-0.2, circles_data[circles_data_labels==0, 0].max()+0.2
# y_min, y_max = circles_data[circles_data_labels==0, 1].min()-0.2, circles_data[circles_data_labels==0, 1].max()+0.2
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
# print(xx,yy)
# cm_bright = ListedColormap(['#D9E021', '#0D8ECF'])
#
#
# plt.figure(figsize=(8,4))
# plt.cla()
# prediction = torch.max(out, 1)[1]
# #Z = prediction.data.numpy()
# #Z = Z.reshape(xx.shape)
#
# target_y = y.data.numpy()
# #plt.pcolormesh(xx, yy, pred_y, cmap=cm_bright, alpha=0.6)
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn',marker='.')
# accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
# plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
# plt.pause(0.1)
# #print(pred_y)
# plt.show()
