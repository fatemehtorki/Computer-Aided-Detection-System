from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D,BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras import backend as K
from keras.utils.conv_utils import convert_kernel
import tensorflow as tf
import numpy as np
from keras.models import load_model


# Model definition
def get_model(first_layer):
    # from keras import applications
    # model = applications.VGG16(include_top=False, weights='imagenet')

    model = Sequential()

    model.add(first_layer)
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))   # TODO

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))  # TODO

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    # model.summary()

    return model

def get_simp_model(first_layer):
    mask_size=7
    model = Sequential()
    model.add(first_layer)
    # Conv. 1
    # 66
    model.add(Conv2D(32, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros', input_shape=(224, 224, 3)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    # Conv. 2,3 & 4
    # 64
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    ###############self.model.add(layers.Dropout(0.2))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    # Conv. 5
    # 96
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))

    # Max Pooling
    model.add(MaxPool2D(pool_size=(2, 2)))
    #        self.model.add(layers.Dropout(0.2))

    # Conv. 6,7,8 & 9
    # 96
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    # Conv. 10
    # 144
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))

    # Max Pooling
    model.add(MaxPool2D(pool_size=(2, 2)))
    #        self.model.add(layers.Dropout(0.3))

    # Conv. 11
    # 144
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.3))

    # Conv. 12
    # 178
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.3))

    # Conv. 13
    # 216
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.3))

    # Conv. 14
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, mask_size, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    return model

def load_model_weights(model, weights_path):
    print('\nLoading model.')

    # Load pre-trained model
    model.load_weights(weights_path, by_name=True)
    model.keras_model._make_predict_function()

    # Theano to Tensoflow - depends on the version
    ops = []
    for layer in model.layers:
        if layer.__class__.__name__ in ['Conv2D']:  # Layers with pre-trained weights
            original_w = K.get_value(layer.kernel)
            converted_w = convert_kernel(original_w)
            ops.append(tf.assign(layer.kernel, converted_w).op)
    K.get_session().run(ops)

    # Prev code
    # f = h5py.File(weights_path)
    # for k in range(f.attrs['nb_layers']):
    #     if k >= len(model.layers):
    #         # we don't look at the last (fully-connected) layers in the savefile
    #         break
    #     g = f['layer_{}'.format(k)]
    #     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    #     model.layers[k].set_weights(weights)
    # f.close()

    # model.save_weights(weights_path)
    print('\nModel loaded.')
    return model


# Return output of specified layer
def get_output_layer(model, layer_name):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer.output


# Load trained model - for occlusion experiment
def load_trained_model():
    # first_layer = ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3))
    # model = get_model(first_layer) # must have FC and output layer for class prediction
    # model.load_weights(weights_path, by_name=True)

    model_address = 'a/model_dir/chest.h5'
    model = load_model(model_address)
    from keras.applications import VGG16
    # model = VGG16(weights='imagenet', input_shape=(224, 224, 3))
    #model = MobileNet(weights='imagenet')
    #model_address = '/home/atlas/PycharmProjects/SimpleNet/model/VGG16_withoutPreTraining.h5'
    #model = load_model(model_address)
    #model.summary()

    return model

# Predict probabilities for given test image using trained model
def pred_prob_list(model, test_image):
    #test_image = np.expand_dims(test_image, axis=0)
    #test_image = preprocess_input(test_image)
    print('image shape befor predict: ',test_image.shape)
    #test_image = test_image/np.max(test_image)
    #test_image = test_image * (1./255)
    # model._make_predict_function()
    # import tensorflow as tf
    # global graph
    #
    # graph = tf.get_default_graph()
    #
    # with graph.as_default():
    #     print("yessss")
    #     predictions = model.predict(test_image)
    # return predictions
    # print("yessss")
    predictions = model.predict(test_image)
    return predictions
