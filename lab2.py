
import os, sys, getopt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import random
import datetime
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

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
DROP_RATE = 0.25
SAVE_PATH = None
LOAD_PATH = None
LOG_PATH  = None
KEEP_TRAINING = False
SHOULD_CROP = False

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
        IS = IH * IW * IZ
    elif DATASET == "cifar_100_f":
        NUM_CLASSES = 100
        IH = 32
        IW = 32
        IZ = 3
        IS = IH * IW * IZ
    elif DATASET == "cifar_100_c":
        NUM_CLASSES = 20
        IH = 32
        IW = 32
        IZ = 3
        IS = IH * IW * IZ


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
    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH, histogram_freq=1)
    ann.fit(x, y,
            validation_split=0.1,
            batch_size=batchSize,
            epochs=eps,
            shuffle=True,
            callbacks=[tensorboardCallback] if LOG_PATH != None else None)
    return ann


def buildTFConvNet(x, y,
                   batchSize=BATCH_SIZE, eps = EPOCHS, dropout = True, dropRate = DROP_RATE,
                   model=None):
    cnn = model
    if (cnn == None):
        cnn = tf.keras.models.Sequential()
        if SHOULD_CROP:
          global IH, IW
          IH, IW = 24, 24
          cnn.add(tf.keras.layers.experimental.preprocessing.RandomCrop(IH, IW))

        cnn.add(Conv2D(8, (2, 2), padding='same', activation='elu', input_shape=(IH, IW, IZ)))
        cnn.add(BatchNormalization())
        cnn.add(Conv2D(16, (2, 2), padding='same', activation='elu'))
        cnn.add(BatchNormalization())

        cnn.add(Conv2D(32, (3, 3), padding='same', activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPool2D((2, 2)))
        cnn.add(Dropout(0.5 * dropRate))

        cnn.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPool2D((2, 2)))
        cnn.add(Dropout(dropRate))

        cnn.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPool2D((2, 2)))
        cnn.add(Dropout(dropRate))

        cnn.add(Flatten())
        cnn.add(Dense(1024, activation='elu'))
        cnn.add(Dropout(2 * dropRate))
        cnn.add(Dense(1024, activation='elu'))
        cnn.add(Dropout(2 * dropRate))
        cnn.add(Dense(NUM_CLASSES, activation='softmax'))

        opt = tf.optimizers.RMSprop(lr=0.001, decay=1e-6)
        cnn.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH, histogram_freq=1)
    cnn.fit(x, y,
            validation_split=0.1,
            batch_size=batchSize,
            epochs=eps,
            shuffle=True,
            callbacks=[tensorboardCallback] if LOG_PATH != None else None)
    cnn.summary()
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
    if LOAD_PATH != None and KEEP_TRAINING == False:
        return tf.keras.models.load_model(LOAD_PATH)
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain, batchSize=BATCH_SIZE, eps=EPOCHS)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        if KEEP_TRAINING == False:
            return buildTFConvNet(xTrain, yTrain, batchSize=BATCH_SIZE, eps=EPOCHS)
        return buildTFConvNet(xTrain, yTrain,
                              batchSize=BATCH_SIZE, eps=EPOCHS,
                              model=tf.keras.models.load_model(LOAD_PATH))
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
    _, yTest = data
    acc = 0
    #predTotals, testTotals = list(), list()
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        #predTotals.append(np.argmax(preds[i]))
        #testTotals.append(np.argmax(preds[i]))
    accuracy = acc / preds.shape[0]
    #print(tf.math.confusion_matrix(testTotals, predTotals).numpy())
    print("Dataset: %s" % DATASET)
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    return (accuracy * 100)


def saveMetaData(model, accuracy):
    # Save algorithm and dataset for future loading
    print('saving model to %s...' % SAVE_PATH)
    model.save(SAVE_PATH)

    # Save hyperparameters, accuracy, and network architecture for easy reference
    metadata = open('%s/meta.txt' % SAVE_PATH, 'w')
    metadata.write('%s\n%s\nepochs=%s\nbatchSize=%s\ndropRate=%s\naccuracy=%f%%\n\n\n'
                    % (ALGORITHM, DATASET, EPOCHS, BATCH_SIZE, DROP_RATE, accuracy))
    model.summary(print_fn=lambda x: metadata.write(x + '\n'))
    metadata.close()


#=========================<Main>================================================

def parseArgs():
    global ALGORITHM, DATASET, EPOCHS, BATCH_SIZE, SAVE_PATH, LOAD_PATH, LOG_PATH, KEEP_TRAINING, SHOULD_CROP
    ALGORITHM, DATASET = 'tf_net', 'mnist_d'
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv, 'a:d:e:b:l:L:sch')
    except:
        raise ValueError('Unrecognized argument. See -h for help')

    shouldSave = True
    algorithms = ['guesser', 'tf_net', 'tf_conv']
    datasets = ['mnist_d', 'mnist_f', 'cifar_10', 'cifar_100_c', 'cifar_100_f']
    for opt, arg in opts:
        if opt == '-a':
            arg = arg.lower()
            if arg not in algorithms:
                raise ValueError('Unrecognized algorithm. Try one of %s' % algorithms)
            ALGORITHM = arg
        elif opt == '-d':
            arg = arg.lower()
            if arg not in datasets:
                raise ValueError('Unrecognized algorithm. Try one of %s' % datasets)
            DATASET = arg
        elif opt == '-c':
            SHOULD_CROP = True
        elif opt == '-e':
            EPOCHS = int(arg)
            if EPOCHS < 1:
                raise ValueError('Number of epochs must be at least 1')
        elif opt == '-b':
            BATCH_SIZE = int(arg)
            if BATCH_SIZE < 1:
                raise ValueError('Batch size must be at least 1')
        elif opt == '-s':
            shouldSave = False
        elif opt == '-l' or opt == '-L':
            LOAD_PATH = arg
            shouldSave = False    # If model is being loaded, no need to save it again
            if opt == '-L':
                KEEP_TRAINING = True
                shouldSave = True

            # Load in the appropriate dataset and use appropriate algorithm from saved model
            metadata = open('%s/meta.txt' % LOAD_PATH, 'r')
            meta = [line.strip() for line in metadata.readlines()]
            ALGORITHM, DATASET = meta[0], meta[1]
        elif opt == '-h':
            print('Usage: \n\
                -a <algorithm> | %s\n\
                -d <dataset> | %s\n\
                -e <epochs>\n\
                -b <batchSize>\n\
                -c toggle random cropping | default=False\n\
                -s toggle model saving | default=True\n\
                -l <path to model to load>\n\
                -L <path to model to load> (continues training)\n'
                % (algorithms, datasets))
            sys.exit()

    # Default save the model, create path
    if shouldSave:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
        SAVE_PATH = './models/%s-%s-%s' % (ALGORITHM, DATASET, timestamp)
        LOG_PATH = './logs/fit/%s-%s-%s' % (ALGORITHM, DATASET, timestamp)
        print(LOG_PATH)

    setDataDimensions()


def main():
    parseArgs()
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    accuracy = evalResults(data[1], preds)
    if ALGORITHM != 'guesser' and SAVE_PATH != None:
        saveMetaData(model, accuracy)


if __name__ == '__main__':
    main()
