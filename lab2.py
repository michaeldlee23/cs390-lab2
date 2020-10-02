
import os, sys, getopt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import matplotlib as plt
import random


random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ALGORITHM = "guesser"
#ALGORITHM = "tf_net"
#ALGORITHM = "tf_conv"

#DATASET = "mnist_d"
#DATASET = "mnist_f"
#DATASET = "cifar_10"
#DATASET = "cifar_100_f"
#DATASET = "cifar_100_c"

EPOCHS = 10
BATCH_SIZE = 100
HN = 512            # Number of Hidden Neurons

def setDataDimensions():
    global NUM_CLASSES, IH, IW, IZ, IS
    if DATASET == "mnist_d":
        NUM_CLASSES = 10
        IH = 28             # Input Height
        IW = 28             # Input Width
        IZ = 1              # Input Depth (Color)
        IS = 784            # Input Size (ANN use)
    elif DATASET == "mnist_f":
        NUM_CLASSES = 10
        IH = 28
        IW = 28
        IZ = 1
        IS = 784
    elif DATASET == "cifar_10":
        NUM_CLASSES = 10
        IH = 32
        IW = 32
        IZ = 3
        IS = IH * IW
    elif DATASET == "cifar_100_f":
        NUM_CLASSES = 100
        IH = 32
        IW = 32
        IZ = 3
        IS = IH * IW
    elif DATASET == "cifar_100_c":
        NUM_CLASSES = 20
        IH = 32
        IW = 32
        IZ = 3
        IS = IH * IW


#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for _ in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, batchSize=BATCH_SIZE, eps = EPOCHS):
    ann = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                      tf.keras.layers.Dense(HN, activation=tf.nn.sigmoid),
                                      tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)])
    ann.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    ann.fit(x, y, validation_split=0.1, batch_size=batchSize, epochs=eps, shuffle=True)
    return ann


def buildTFConvNet(x, y, batchSize=BATCH_SIZE, eps = EPOCHS, dropout = True, dropRate = 0.2):
    # layers = [tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IH, IW, IZ)),
    #           tf.keras.layers.MaxPooling2D((2, 2)),
    #           tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #         #   tf.keras.layers.MaxPooling2D((2, 2)),
    #         #   tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #           tf.keras.layers.Flatten(),
    #           tf.keras.layers.Dense(100, activation='relu'),
    #           tf.keras.layers.Dropout(dropRate),
    #           tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')]
    # cnn = tf.keras.models.Sequential(layers)
    cnn = tf.keras.models.Sequential()
    cnn.add(Conv2D(32, (3, 3), padding='same', activation='elu', input_shape=(IH, IW, IZ)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(32, (3, 3), padding='same', activation='elu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Dropout(0.2))

    cnn.add(Conv2D(64, (3, 3), padding='same', activation='elu', input_shape=(IH, IW, IZ)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, (3, 3), padding='same', activation='elu', input_shape=(IH, IW, IZ)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Dropout(0.4))

    cnn.add(Flatten())
    cnn.add(Dense(NUM_CLASSES, activation='softmax'))

    cnn.summary()
    opt = tf.optimizers.RMSprop(lr=0.001, decay=1e-6)
    cnn.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    cnn.fit(x, y, validation_split=0.1, batch_size=batchSize, epochs=eps, shuffle=True)
    return cnn

#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":        # Goal: >99% accuracy
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
    elif DATASET == "mnist_f":      # Goal: >92% accuracy
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.fashion_mnist.load_data()
    elif DATASET == "cifar_10":     # Goal: >70% accuracy
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
    elif DATASET == "cifar_100_f":  # Goal: >35% accuracy
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    elif DATASET == "cifar_100_c":  # Goal: >50% accuracy
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain, xTest = xTrain / 255.0, xTest / 255.0
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain, batchSize=BATCH_SIZE, eps=EPOCHS)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain, batchSize=BATCH_SIZE, eps=EPOCHS)
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


#=========================<Main>================================================

def parseArgs():
    global ALGORITHM, DATASET, EPOCHS, BATCH_SIZE
    ALGORITHM, DATASET = 'tf_net', 'mnist_d'
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv, 'a:d:e:b:h')
    except:
        raise ValueError('Unrecognized argument. See -h for help')

    algorithms = ['guesser', 'tf_net', 'tf_conv']
    datasets = ['mnist_d', 'mnist_f', 'cifar_10', 'cifar_100_c', 'cifar_100_f']
    for opt, arg in opts:
        if opt in ['-a']:
            arg = arg.lower()
            if arg not in algorithms:
                raise ValueError('Unrecognized algorithm. Try one of %s' % algorithms)
            ALGORITHM = arg
        elif opt in ['-d']:
            arg = arg.lower()
            if arg not in datasets:
                raise ValueError('Unrecognized algorithm. Try one of %s' % datasets)
            DATASET = arg
        elif opt in ['-e']:
            EPOCHS = int(arg)
            if EPOCHS < 1:
                raise ValueError('Number of epochs must be at least 1')
        elif opt in ['-b']:
            BATCH_SIZE = int(arg)
            if BATCH_SIZE < 1:
                raise ValueError('Batch size must be at least 1')
        elif opt in ['-h']:
            print('Usage: python lab2.py\n\
                -a <algorithm> | %s\n\
                -d <dataset> | %s\n\
                -e <epochs>\n\
                -b <batchSize>\n'
                % (algorithms, datasets))
            sys.exit()

    setDataDimensions()


def main():
    parseArgs()
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    main()
