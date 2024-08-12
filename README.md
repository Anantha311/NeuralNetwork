# NeuralNetwork

## You have to first import Model, CrossEntropyLoss or MSE, SGD, Linear, ReLU or whatever activation function you require, Softmax

## Add the layers by specifying the number of neurons and the activation function used, You can do this using model = Model(), model.add_layer(Linear(784, 128)), model.add_layer(ReLU())

## Specify the loss using loss = CrossEntropyLoss(), learning rate using the optimizer (optimizer = SGD(learning_rate=0.01)), then use model.compile(loss, optimizer)

## Train the model using model.train(x_train, y_train, epochs=20, batch_size=64) and make sure that y_train is one hot coded, for that you can use y_train = model.one_hot(train['label'].value)

## Then you can evaluate the model using test_loss, test_accuracy = model.evaluate(x_test, y_test) and print test_loss and test_accuracy

## You can save and load using model.save('file_name.npz') and model.load('file_name.npz')

## You can even print the graph of losses vs epoch using model.plot_loss()

