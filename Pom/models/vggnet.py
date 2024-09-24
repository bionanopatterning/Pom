import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam


def create_model(input_shape, output_dimensionality=1):
    inputs = Input(input_shape)

    # Block 1
    conv1 = Conv2D(512, (5, 5), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(512, (5, 5), activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2b = Conv2D(512, (5, 5), activation='relu', padding='same', dilation_rate=2)(conv2)
    conv2b = BatchNormalization()(conv2b)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2b)

    # Block 2
    conv3 = Conv2D(256, (5, 5), activation='relu', padding='same')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (5, 5), activation='relu', padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Block 3
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    # Additional block with dilated convolutions for larger receptive field
    conv7 = Conv2D(512, (5, 5), activation='relu', padding='same', dilation_rate=2)(pool3)
    conv7 = BatchNormalization()(conv7)
    conv8 = Conv2D(512, (5, 5), activation='relu', padding='same', dilation_rate=2)(conv7)
    conv8 = BatchNormalization()(conv8)

    # Upsampling and Decoding
    up1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv8)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)

    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv9)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)

    up3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv10)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)

    # Final output layer
    output = Conv2D(output_dimensionality, (1, 1), activation='sigmoid')(conv11)

    # create the model
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model
