import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam


# class attention_row_columns(tf.keras.layers.Layer):
#     def __init__(self, dmodel, nheads, k_dim, dff_dim):
#         super().__init__()
#         assert k_dim % nheads == 0
#         self.MHA_row_layer = tf.keras.layers.MultiHeadAttention(nheads, k_dim)
#         self.MHA_col_layer = tf.keras.layers.MultiHeadAttention(nheads, k_dim)
#
#         self.permute = tf.keras.layers.Permute((2, 1, 3))
#
#         self.layer_norm1 = tf.keras.layers.LayerNormalization()
#         self.layer_norm2 = tf.keras.layers.LayerNormalization()
#
#         self.linear1 = tf.keras.layers.Dense(dff_dim)
#         self.linear2 = tf.keras.layers.Dense(dmodel)
#
#
#     def call(self, x):
#         x = self.layer_norm1(x)
#         x1 = self.MHA_row_layer(x, x, x)
#         x1 = self.permute(x1)
#         x1 = self.MHA_col_layer(x1, x1, x1)
#         x = self.layer_norm2(x1 + x)
#         x = self.linear2(tf.keras.activations.relu(self.linear1(x)))
#         return x


def create_model(input_shape, output_dimensionality):
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

    # attn_layer = attention_row_columns(64, 8, 64, 1024)
    # attn1 = attn_layer(conv9)

    output = Conv2D(output_dimensionality, (1, 1), activation='softmax')(conv9)

    # Create the model
    model = Model(inputs=[inputs], outputs=[output])

    # Compile the model with a suitable optimizer and loss function
    model.compile(optimizer=Adam(learning_rate=2.5e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
