import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, concatenate, Activation, Multiply, add
from tensorflow.keras.optimizers import Adam


def dice_loss(y_true, y_pred, epsilon=1e-6):
    # Compute per-class Dice coefficient
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[0,1,2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[0,1,2])

    dice_coeff = (numerator + epsilon) / (denominator + epsilon)
    dice_loss = 1 - tf.reduce_mean(dice_coeff)
    return dice_loss


def combined_loss(y_true, y_pred):
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return cce + d_loss


def attention_block(x, gating, inter_channels):
    theta_x = Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(x)
    phi_g = Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(gating)
    add_xg = add([theta_x, phi_g])
    act_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(act_xg)
    sigmoid_psi = Activation('sigmoid')(psi)
    attn_out = Multiply()([x, sigmoid_psi])
    return attn_out


def create_model(input_shape, output_dimensionality):
    inputs = Input(input_shape)

    # Encoder
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

    # Decoder
    # Up Block 1
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    attn_6 = attention_block(conv4, up6, inter_channels=256)
    merge6 = concatenate([up6, attn_6], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    # Up Block 2
    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    attn_7 = attention_block(conv3, up7, inter_channels=128)
    merge7 = concatenate([up7, attn_7], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    # Up Block 3
    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    attn_8 = attention_block(conv2, up8, inter_channels=64)
    merge8 = concatenate([up8, attn_8], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    # Up Block 4
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    attn_9 = attention_block(conv1, up9, inter_channels=32)
    merge9 = concatenate([up9, attn_9], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    output = Conv2D(output_dimensionality, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(learning_rate=3e-5), loss=combined_loss, metrics=['accuracy'])

    return model
