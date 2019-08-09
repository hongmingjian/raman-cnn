import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.layers import Input, Layer
from keras.layers.core import Dense, Flatten, Reshape, InputSpec
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers.convolutional import Conv1D
from keras.regularizers import Regularizer, l1, l2
from keras.constraints import Constraint
from keras.initializers import Constant
import scipy.io as sio
import numpy as np

from keras import backend as K
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')


def normalize_tuple(value, n, name):
    """Transforms a single int or iterable of ints into an int tuple.

    # Arguments
        value: The value to validate and convert. Could an int, or any iterable
          of ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. "strides" or
          "kernel_size". This is only used to format error messages.

    # Returns
        A tuple of n integers.

    # Raises
        ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except ValueError:
                raise ValueError('The `' + name + '` argument must be a tuple of ' +
                                 str(n) + ' integers. Received: ' + str(value) + ' '
                                 'including element ' + str(single_value) + ' of type' +
                                 ' ' + str(type(single_value)))
    return value_tuple

class Padding1D(Layer):
    def __init__(self, padding=1, mode='CONSTANT', constant_values=0, **kwargs):
        super(type(self), self).__init__(**kwargs)
        self.padding = normalize_tuple(padding, 2, 'padding')
        self.mode = mode
        self.constant_values = constant_values
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        if input_shape[1] is not None:
            length = input_shape[1] + self.padding[0] + self.padding[1]
        else:
            length = None
        return (input_shape[0],
                length,
                input_shape[2])

    def call(self, inputs):
        if K.backend() == 'tensorflow':
            return K.tensorflow_backend.tf.pad(inputs,
                                               [[0, 0], [self.padding[0], self.padding[1]], [0, 0]],
                                               mode=self.mode,
                                               constant_values=self.constant_values)
        else:
            raise NotImplementedError(type(self).__name__+": Not implemented for backend "+K.backend())

    def get_config(self):
        config = {'padding': self.padding, 'mode': self.mode, 'constant_values': self.constant_values}
        base_config = super(type(self), self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class C1_Constraint(Constraint):
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        w /= K.sum(w, axis=self.axis, keepdims=True)
        return w

    def get_config(self):
        return {'axis': self.axis}

class C2_Constraint(Constraint):
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        w1 = np.zeros(w.shape)
        w1[w1.shape[0]//2, :, :] = 1
        w1 = K.cast(w1, K.floatx())
        w2 = (w1 - w)
        w2 *= K.cast(K.greater_equal(w2, 0.), K.floatx())
        w2 /= K.sum(w2, axis=self.axis, keepdims=True)
        w = w1 - w2
        return w

    def get_config(self):
        return {'axis': self.axis}

data=sio.loadmat("Rawdata.mat")
train_X=np.array(data["model2X"])
Y=np.array(data["model2Y"])
trainY=np.zeros((Y.size, 2))
for i in range(Y.size):
    if Y[i,0]==-1:
        trainY[i, 0] = 1
    else:
        trainY[i, 1] = 1

Y2=np.array(data["test2Y"])
test_X=np.array(data["test2X"])
testY=np.zeros((Y2.size, 2))
for i in range(Y2.size):
    if Y2[i,0]==-1:
        testY[i, 0] = 1
    else:
        testY[i, 1] = 1

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def cnn_block(x, c1, c2, xlen):
    out = Reshape((xlen, 1))(x)
    out = Padding1D(c1//2, mode='SYMMETRIC')(out)
    out = Conv1D(1, c1, name="cnn_1", kernel_constraint=C1_Constraint(), use_bias=False, input_shape=(xlen, 1))(out)
    out = Padding1D(c2//2, mode='SYMMETRIC')(out)
    out = Conv1D(1, c2, name="cnn_2", kernel_constraint=C2_Constraint(), use_bias=False)(out)
    out = Flatten()(out)
    return out

def classifier(x):
    out = Dense(300, activation="relu", kernel_regularizer=l2(2e-5))(x)
    out = Dense(150, activation="relu", kernel_regularizer=l2(2e-5))(out)
    out = Dense(2, activation="softmax")(out)
    return out

c1=5
c2=71
batch_size=32
epochs=59

inp = Input(shape=(train_X.shape[1],))
out = cnn_block(inp, c1, c2, train_X.shape[1])
out = classifier(out)
model = Model(inp, out)

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()
history = LossHistory()

model.fit(train_X, trainY, callbacks=[history], batch_size=batch_size, epochs=epochs)

loss_and_metrics1=model.evaluate(train_X,trainY)
loss_and_metrics2=model.evaluate(test_X,testY)

print(loss_and_metrics1)
print(loss_and_metrics2)

get_cnn1 = Model(inputs=model.input, outputs=model.get_layer('cnn_1').output)
cnn1_output =get_cnn1.predict(train_X)

get_cnn2 = Model(inputs=model.input, outputs=model.get_layer('cnn_2').output)
cnn2_output =get_cnn2.predict(train_X)

sample = 5

plt.figure(1)
ax = plt.subplot(111)
ax.plot(np.linspace(598, 1837, train_X.shape[1]), train_X[sample, :])
ax.plot(np.linspace(598, 1837, train_X.shape[1]), cnn1_output[sample, :, 0], 'r')
ax.plot(np.linspace(598, 1837, train_X.shape[1]), train_X[sample, :]-cnn1_output[sample, :, 0])
ax.legend(['Raw input', 'Output of C1', 'Error'])
ax.set_xlabel('Raman shift ($cm^{-1}$)')
ax.set_ylabel('Intensity(counts)')
ax.yaxis.set_ticks([], [])

plt.figure(2)
ax = plt.subplot(111)
ax.plot(np.linspace(598, 1837, train_X.shape[1]), cnn1_output[sample, :, 0])
ax.plot(np.linspace(598, 1837, train_X.shape[1]), cnn2_output[sample, :, 0], 'r')
ax.plot(np.linspace(598, 1837, train_X.shape[1]), cnn1_output[sample, :, 0]-cnn2_output[sample, :, 0])
#ax.legend(['Input of C2 (Ouput of C1)', 'Output of C2', 'Error'])
ax.legend(['x(n)', 's(n)', 'x(n)*h(n)'])  # for Fig.2
ax.set_xlabel('Raman shift ($cm^{-1}$)')
ax.set_ylabel('Intensity(counts)')
ax.yaxis.set_ticks([], [])

plt.figure(12)
ax = plt.subplot(111)
ax.plot(model.get_weights()[0][:, 0, 0])
ax.set_xlabel('Index of convolution kernel')
ax.set_ylabel('weight')

plt.figure(13)
ax = plt.subplot(111)
ax.plot(model.get_weights()[1][:, 0, 0])
ax.set_xlabel('Index of convolution kernel')
ax.set_ylabel('weight')

from scipy import signal
plt.figure(22)
ax = plt.subplot(111)
w, h = signal.freqz(model.get_weights()[0][:, 0, 0])
ax.plot(w, 20 * np.log10(abs(h)))
ax.set_ylabel('Amplitude [dB]')
ax.set_xlabel('Frequency [rad]')

plt.figure(23)
ax = plt.subplot(111)
w, h = signal.freqz(model.get_weights()[1][:, 0, 0])
ax.plot(w, 20 * np.log10(abs(h)))
ax.set_ylabel('Amplitude [dB]')
ax.set_xlabel('Frequency [rad]')

def epoch_format_func(value, tick_number):
    return r"%d" % (value//(train_X.shape[0]//batch_size)+1)

plt.figure(9)
ax = plt.subplot(111)
ax.plot(history.losses)
ax.xaxis.set_major_formatter(plt.FuncFormatter(epoch_format_func))
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')

plt.show()
