import pandas
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/Smoking.xls'

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = []
for i in range(1, sheet.nrows):
    data.append([sheet.cell(i, j).value for j in (1,5)])
#data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
print(data)
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (Avg. Area House Age) and label Y (Price)
X1 = tf.Variable(0.0, name='weights')
X2 = tf.Variable(0.0, name='weights')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
W1 = tf.Variable(0.0, name= 'bias')
W2 = tf.Variable(0.0, name= 'bias')
b = tf.Variable(0.0, name='bias')

# Step 4: build model to predict Y
Y_predicted = (X1 * W1) + (X2 * W2) + b

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    # Step 8: train the model
    for i in range(18):  # train the model 50 epochs
        total_loss = 0
        for x, y in data:
            # Session runs train_op and fetch values of loss
            _, l = sess.run([optimizer, loss], feed_dict={X1: x, X2: y})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # close the writer when you're done using it
    writer.close()

    # Step 9: output the values of w and b
    w, b = sess.run([X1, X2])

# plot the results
X, Y = data[0], data[1]
plt.plot(X1, X2, 'bo', label='Real data')
plt.plot(X1, Y_predicted, 'r', label='Predicted data')
plt.legend()
plt.show()