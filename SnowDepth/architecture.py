from keras import layers, models

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def unet(input_shape, base_filters=32):
    """
    UNet for per-pixel regression
    - Linear 1-channel output
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPool2D()(c1)

    c2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPool2D()(c2)

    c3 = conv_block(p2, base_filters * 4)
    p3 = layers.MaxPool2D()(c3)

    c4 = conv_block(p3, base_filters * 8)
    p4 = layers.MaxPool2D()(c4)

    # Bottleneck
    bn = conv_block(p4, base_filters * 16)

    # Decoder
    u4 = layers.Conv2DTranspose(base_filters * 8, 2, strides=2, padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, base_filters * 8)

    u3 = layers.Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, base_filters * 4)

    u2 = layers.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, base_filters * 2)

    u1 = layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, base_filters)

    # Regression head
    outputs = layers.Conv2D(1, 1, activation="linear")(c8)

    model = models.Model(inputs=inputs, outputs=outputs, name="unet")
    return model
