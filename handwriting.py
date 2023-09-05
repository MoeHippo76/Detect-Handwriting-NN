import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tsne
rng = np.random.RandomState(23455)

"""
mnist_loader
~~~~~~~~~~~~

taken from Nielsen's online book:
http://neuralnetworksanddeeplearning.com/chap1.html


A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    
    global training_inputs, training_results
    global validation_inputs, validation_results
    global test_inputs, test_results
    global num_samples, numpixels, num_test_samples
    global tr_d, va_d, te_d
    tr_d, va_d, te_d = load_data()
    
    num_samples=len(tr_d[0])
    training_inputs=np.zeros([num_samples,numpixels,numpixels])
    training_results=np.zeros([num_samples,10])    
    for j in range(num_samples):
        training_inputs[j,:] = np.reshape(tr_d[0][j], (numpixels, numpixels))
        training_results[j,:] = vectorized_result(tr_d[1][j])
#    validation_inputs = [np.reshape(x, (numpixels)) for x in va_d[0]]
#    validation_results = [vectorized_result(y) for y in va_d[1]]

    num_test_samples=len(te_d[0])
    test_inputs=np.zeros([num_test_samples,numpixels, numpixels])
    test_results=np.zeros([num_test_samples,10])    
    for j in range(num_test_samples):
        test_inputs[j,:] = np.reshape(te_d[0][j], (numpixels, numpixels))
        test_results[j,:] = vectorized_result(te_d[1][j])

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10))
    e[j] = 1.0
    return e

def display_image(index):
    global test_inputs, numpixels
    if numpixels != 28:
        plt.imshow(np.reshape(te_d[0][index], (28,28)), cmap = 'binary')
    else:
        plt.imshow(np.reshape( test_inputs[index], (numpixels, numpixels)), cmap = 'binary' )
    plt.show()

def display_image_array(num_cols):
    global numpixels
    bigImage = np.zeros([num_cols*numpixels, num_cols*numpixels])

    for j in range( num_cols**2):
        x = (j%num_cols) * numpixels
        y = int(j/num_cols) * numpixels
        bigImage[x: x + numpixels,  y: y + numpixels] = np.reshape(training_inputs[j, :], [numpixels, numpixels])
    
    plt.imshow(bigImage, cmap = 'binary')
    plt.show()

def test_on(start, stop, print = False):
    global test_inputs, test_results
    global net

    prediction_prob = net.predict_on_batch(test_inputs[start:stop, :])
    predictions = np.argmax(prediction_prob, axis= 1)
    true_labels = np.argmax(test_results[start:stop, :], axis=1)

    if print:
        print("Predictions  ", predictions)
        print("True Labels", true_labels)
    
    return predictions, true_labels


def init_net():
    global net, numpixels
    
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Conv2D( filters = 7, kernel_size = [5,5], input_shape = (numpixels, numpixels, 1), activation = 'relu', padding = 'same'))
    net.add(tf.keras.layers.AveragePooling2D(pool_size = 4))
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10, activation= "softmax"))
    net.compile(loss = "categorical_crossentropy", optimizer = tf.keras.optimizers.SGD(learning_rate= 1),  metrics=['categorical_accuracy'])

def init_net_dense(PCA_dims = None):
    global net, numpixels
    
    if type(PCA_dims) is not None:
        fit_PCA(PCA_dims)
    
    
    net = tf.keras.Sequential()
    # note: batch_input_shape is (batchsize,timesteps,data_dim)
    net.add(tf.keras.layers.Dense(400, input_shape=(numpixels**2,), activation='relu'))
    net.add(tf.keras.layers.GaussianDropout(0.1))
    net.add(tf.keras.layers.Dense(100, activation='relu'))
    net.add(tf.keras.layers.Dense(10, activation='softmax'))
    net.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate = 1.0), metrics=['categorical_accuracy'])

def display_mistakes(maxnum):
    global num_test_samples

    predictions_probs=net.predict_on_batch(test_inputs[0:num_test_samples])
    predictions= np.argmax(predictions_probs,axis=1)
    true_labels= np.argmax(test_results[0:num_test_samples], axis=1)
    
    which=np.where(true_labels!=predictions)[0]
    for j in which:
        if j<maxnum:
            display_image(j)
            print("True ", true_labels[j], " - Predicted ", predictions[j], " with prob. ", predictions_probs[j,predictions[j]])
 
def apply_tsne(num_images = 2500):
    global tr_d

    X = tr_d[0][0:num_images]
    X_labels = tr_d[1][0:num_images]

    Y = tsne.tsne(X, 2, 50, 30)
    plt.scatter(Y[:, 0], Y[:, 1], 20, X_labels)
    plt.title("t-SNE on MNIST dataset")
    plt.legend()
    plt.show()
    plt.savefig("tsne_handwriting")

def fit_PCA(num_eigs_root):
    global tr_d, te_d, training_inputs, test_inputs, numpixels
    X = tr_d[0]
    psi = X - np.sum(X)/num_samples
    rho = np.dot(psi.T, psi)
    evals, evecs = np.linalg.eigh(rho)

    numpixels = num_eigs_root

    evecs = evecs.T[::-1][0: numpixels**2]


    training_inputs = np.zeros([num_samples,numpixels * numpixels])

    test_inputs=np.zeros([num_test_samples,numpixels * numpixels])

    projection = np.dot(evecs , X.T).T

    for j in range(num_samples):
        training_inputs[j, :] = np.reshape(projection[j], (numpixels * numpixels))

    projection = np.dot(te_d[0], evecs.T)
    for j in range(num_test_samples):
        test_inputs[j, :] = np.reshape(projection[j], (numpixels * numpixels) )


numpixels= 28
load_data_wrapper() # load all the MNIST images

#init_net()
init_net_dense(10)

history = net.fit(training_inputs, training_results, batch_size= 100, epochs=30, validation_split=0.1)

predictions, true = test_on(0, num_test_samples)
query = np.where(predictions != true)[0]
print(f"Error in test data = {len(query) / num_test_samples * 100} %")

display_mistakes(100)

plt.plot(history.history["categorical_accuracy"], label = "training dataset")
plt.plot(history.history["val_categorical_accuracy"], label = "validation dataset")
plt.title("Categorical Accuracy of Net")
plt.legend()
plt.show()

