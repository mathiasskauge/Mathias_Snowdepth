import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def build_unet(input_shape=(256, 256, 7)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPool2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPool2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPool2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPool2D((2, 2))(c4)

    # Bottleneck
    bn = conv_block(p4, 1024)

    # Decoder
    u1 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(bn)
    u1 = layers.Concatenate()([u1, c4])
    c5 = conv_block(u1, 512)

    u2 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c5)
    u2 = layers.Concatenate()([u2, c3])
    c6 = conv_block(u2, 256)

    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c6)
    u3 = layers.Concatenate()([u3, c2])
    c7 = conv_block(u3, 128)

    u4 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c7)
    u4 = layers.Concatenate()([u4, c1])
    c8 = conv_block(u4, 64)

    outputs = layers.Conv2D(1, 1, activation=None)(c8)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

