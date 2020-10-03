import tensorflow as tf

(xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()

def processImage(image, label):
  print('image:', image)
  print('label:', label)
  return tf.image.resize(image, (227, 227)), label

yTrain = tf.keras.utils.to_categorical(yTrain, 10)
yTest = tf.keras.utils.to_categorical(yTest, 10)

train_ds = tf.data.Dataset.from_tensor_slices((xTrain[:3], yTrain[:3]))
test_ds = tf.data.Dataset.from_tensor_slices((xTest, yTest))

train_ds = (train_ds.map(processImage))
#test_ds = (test_ds.map(processImage))

print(train_ds.shape)
