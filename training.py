[9]:
def train_neural_network(x):

    train_data = np.load('traindata-50-50-20.npy')
    test_data = np.load('testdata-50-50-20.npy')
    
    train = train_data[:10]
    test = train_data[10:12]

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            success_total = 0
            attempt_total = 0
            for data in train:
                attempt_total += 1
                try:

                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    success_total += 1

                except Exception as e:
                    print('Error occured')

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss, 'success_rate:', success_total/attempt_total)
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x: [i[0] for i in test], y: [i[1] for i in test]}))

train_neural_network(x)
    return binary_image