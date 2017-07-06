import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("/Users/Mohamad/Projects/PyCharm Projects/Regression on Yahoo! Finance/data.csv")

# Drop date info
data = data.drop(['Date'], axis=1)

# Remove all nan entries
data = data.dropna(inplace=False)

data = data.drop(['Adj Close', 'Volume'], axis=1)

data_train = data[:2993]
data_test = data[2993:]

# For normalizing dataset
scaler = MinMaxScaler()

# Predicting the 'close' value of stocks
X_train = scaler.fit_transform(data_train.drop(['Close'],axis=1).as_matrix())
y_train = scaler.fit_transform(data_train['Close'].as_matrix())

X_test = scaler.fit_transform(data_test.drop(['Close'], axis=1).as_matrix())
y_test = scaler.fit_transform(data_test['Close'].as_matrix())
print(X_train.shape)


def neural_net_model(X_data, input_dim):
    """
    neural_net_model is function applying 2 hidden layer feed forward neural net.
    Weights and biases are abberviated as W_1,W_2 and b_1, b_2
    These are variables with will be updated during training.
    """

    # layer 1 multiplying and adding bias then activation function
    W_1 = tf.Variable(tf.random_uniform([input_dim, 10]))
    b_1 = tf.Variable(tf.zeros([10]))

    layer_1 = tf.add(tf.matmul(X_data, W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)

    # layer 2 multiplying and adding bias then activation function
    W_2 = tf.Variable(tf.random_uniform([10,10]))
    b_2 = tf.Variable(tf.zeros(10))

    layer_2 = tf.add(tf.matmul(layer_1, W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)

    # O/p layer multiplying and adding bias then activation function
    # notice output layer has one node only since performing #regression
    W_output = tf.Variable(tf.random_uniform([10,1]))
    b_output = tf.Variable(tf.zeros(1))

    layer_out = tf.add(tf.matmul(layer_2, W_output), b_output)

    return layer_out


xs = tf.placeholder('float')
ys = tf.placeholder('float')

output = neural_net_model(xs, 3)

# Mean squared error cost function
cost = tf.reduce_mean(tf.square(output-ys))

# Start training
train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:

    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for i in range(100):
        for j in range(X_train.shape[0]):
            sess.run([cost, train], feed_dict={xs:X_train[j,:].reshape(1,3), ys:y_train[j]})
            # Run cost and train with each sample
        sess.run(cost, feed_dict={xs:X_train, ys:y_train})
        sess.run(cost, feed_dict={xs:X_test, ys:y_test})

    # predict output of test data after training
    pred = sess.run(output, feed_dict={xs:X_test})

    print('Cost :',sess.run(cost, feed_dict={xs:X_test, ys:y_test}))

    plt.plot(range(1282), y_test, label="Original Data")
    plt.plot(range(1282), pred, label="Predicted Data")
    plt.legend(loc='best')
    plt.ylabel('Stock Value')
    plt.xlabel('Days')
    plt.title('Stock Market Nifty')
    plt.show()
    if input('Save model ? [Y/N]') == 'Y':
        saver.save(sess,'yahoo_dataset.ckpt')
        print('Model Saved')

