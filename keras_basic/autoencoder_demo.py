"""
https://www.cnblogs.com/yangmang/p/7530463.html
"""
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 500
input_dim = 28*28

def AutoEncoder():
    # build autoencoder model
    # this is our input placeholder
    input = Input(shape=(input_dim,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    # this model maps an input to its reconstruction
    autoencoder = Model(input, decoded)


    # # this model maps an input to its encoded representation
    # encoder = Model(input, encoded)
    # encoded_input = Input(shape=(encoding_dim,))
    # # retrieve the last layer of the autoencoder model
    # decoder_layer = autoencoder.layers[-1]
    # # create the decoder model
    # decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder

def train(autoencoder,x_train,x_test):
    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

def mnist_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0
    #print(x_train.shape)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    #print(x_train.shape)
    return x_train, x_test

if __name__ == '__main__':
    x_train,x_test = mnist_data()
    autoencoder = AutoEncoder()
    train(autoencoder=autoencoder,
          x_train=x_train,
          x_test=x_test)
