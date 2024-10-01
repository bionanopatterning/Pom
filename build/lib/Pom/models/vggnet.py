import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam


def create_model(input_shape, output_dimensionality=1):
    inputs = Input(input_shape)

    # Block 1
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Block 2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Block 4
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    # Up Block 1
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    # Up Block 2
    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    # Up Block 3
    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    # Up Block 4
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    # Output Layer
    output = Conv2D(output_dimensionality, (1, 1), activation='sigmoid')(conv9)

    # Create the model
    model = Model(inputs=[inputs], outputs=[output])

    # Compile the model with a suitable optimizer and loss function
    model.compile(optimizer=Adam(learning_rate=0.5e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model
# def create_model(input_shape, output_dimensionality=1):
#     inputs = Input(input_shape)
#
#     # Block 1
#     conv1 = Conv2D(512, (5, 5), activation='relu', padding='same')(inputs)
#     conv1 = BatchNormalization()(conv1)
#     conv2 = Conv2D(512, (5, 5), activation='relu', padding='same')(conv1)
#     conv2 = BatchNormalization()(conv2)
#     conv2b = Conv2D(512, (5, 5), activation='relu', padding='same', dilation_rate=2)(conv2)
#     conv2b = BatchNormalization()(conv2b)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv2b)
#
#     # Block 2
#     conv3 = Conv2D(256, (5, 5), activation='relu', padding='same')(pool1)
#     conv3 = BatchNormalization()(conv3)
#     conv4 = Conv2D(256, (5, 5), activation='relu', padding='same')(conv3)
#     conv4 = BatchNormalization()(conv4)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#     # Block 3
#     conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
#     conv5 = BatchNormalization()(conv5)
#     conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
#     conv6 = BatchNormalization()(conv6)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)
#
#     # Additional block with dilated convolutions for larger receptive field
#     conv7 = Conv2D(512, (5, 5), activation='relu', padding='same', dilation_rate=2)(pool3)
#     conv7 = BatchNormalization()(conv7)
#     conv8 = Conv2D(512, (5, 5), activation='relu', padding='same', dilation_rate=2)(conv7)
#     conv8 = BatchNormalization()(conv8)
#
#     # Upsampling and Decoding
#     up1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv8)
#     conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
#
#     up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv9)
#     conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
#
#     up3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv10)
#     conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
#
#     # Final output layer
#     output = Conv2D(output_dimensionality, (1, 1), activation='sigmoid')(conv11)
#
#     # create the model
#     model = Model(inputs=[inputs], outputs=[output])
#     model.compile(optimizer=Adam(), loss='binary_crossentropy')
#
#     return model
