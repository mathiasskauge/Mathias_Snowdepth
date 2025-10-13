from keras import layers as L, models
import numpy as np
import tensorflow as tf

''' UNET'''

def conv_block(x, filters):
    x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return x

def unet(input_shape, base_filters=32):
    """
    UNet for per-pixel regression
    - Linear 1-channel output
    """
    inputs = L.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, base_filters)
    p1 = L.MaxPool2D()(c1)

    c2 = conv_block(p1, base_filters * 2)
    p2 = L.MaxPool2D()(c2)

    c3 = conv_block(p2, base_filters * 4)
    p3 = L.MaxPool2D()(c3)

    c4 = conv_block(p3, base_filters * 8)
    p4 = L.MaxPool2D()(c4)

    # Bottleneck
    bn = conv_block(p4, base_filters * 16)

    # Decoder
    u4 = L.Conv2DTranspose(base_filters * 8, 2, strides=2, padding="same")(bn)
    u4 = L.Concatenate()([u4, c4])
    c5 = conv_block(u4, base_filters * 8)

    u3 = L.Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(c5)
    u3 = L.Concatenate()([u3, c3])
    c6 = conv_block(u3, base_filters * 4)

    u2 = L.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(c6)
    u2 = L.Concatenate()([u2, c2])
    c7 = conv_block(u2, base_filters * 2)

    u1 = L.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(c7)
    u1 = L.Concatenate()([u1, c1])
    c8 = conv_block(u1, base_filters)

    # Regression head
    outputs = L.Conv2D(1, 1, activation="linear")(c8)

    model = models.Model(inputs=inputs, outputs=outputs, name="unet")
    return model


''' TRANSFORMER '''

def _gelu(x):  
    return tf.nn.gelu(x)

class Patchify(L.Layer):
    def __init__(self, patch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.ps = patch_size

    def call(self, x):
        # x: (B, H, W, C)
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]
        ps = self.ps
        # check divisibility at runtime
        tf.debugging.assert_equal(H % ps, 0, message="H must be divisible by patch_size")
        tf.debugging.assert_equal(W % ps, 0, message="W must be divisible by patch_size")
        nH = H // ps
        nW = W // ps
        # (B, nH, nW, ps, ps, C)
        x = tf.reshape(x, (B, nH, ps, nW, ps, C))
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))  # (B, nH, nW, ps, ps, C)
        # flatten patches -> (B, nH*nW, ps*ps*C)
        x = tf.reshape(x, (B, nH * nW, ps * ps * C))
        return x

class Depatchify(L.Layer):
    def __init__(self, out_height, out_width, channels, patch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.H = out_height
        self.W = out_width
        self.C = channels
        self.ps = patch_size

    def call(self, tokens):
        B = tf.shape(tokens)[0]
        ps = self.ps
        H, W, C = self.H, self.W, self.C
        nH = H // ps
        nW = W // ps
        x = tf.reshape(tokens, (B, nH, nW, ps, ps, C))   
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))                
        x = tf.reshape(x, (B, H, W, C))                         
        return x

def transformer_block(x, d_model, num_heads, mlp_dim, dropout=0.0):
    y = L.LayerNormalization(epsilon=1e-6)(x)
    y = L.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)(y, y)
    y = L.Dropout(dropout)(y)
    x = L.Add()([x, y])
    # MLP
    y = L.LayerNormalization(epsilon=1e-6)(x)
    y = L.Dense(mlp_dim, activation=_gelu)(y)
    y = L.Dropout(dropout)(y)
    y = L.Dense(d_model)(y)
    y = L.Dropout(dropout)(y)
    x = L.Add()([x, y])
    return x

def transformer_seg_model(
    input_shape,      
    patch_size=16,        
    d_model=256,           
    depth=4,
    num_heads=4,               
    mlp_dim=512,
    dropout=0.0,
    conv_head_filters=64   
):
    H, W, C = input_shape
    assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size"
    n_patches = (H // patch_size) * (W // patch_size)
    patch_dim = patch_size * patch_size * C

    inp = L.Input(shape=input_shape)

    # Optional: shallow conv stem (helps stability)
    x = L.Conv2D(C, 3, padding="same")(inp)

    # Patchify -> (B, N, patch_dim)
    tokens = Patchify(patch_size=patch_size)(x)

    # Linear projection to d_model
    tokens = L.Dense(d_model)(tokens)

    # Learnable positional embeddings
    pos_emb = tf.Variable(
        initial_value=tf.random.normal([1, n_patches, d_model], stddev=0.02),
        trainable=True,
        name="pos_embedding"
    )
    tokens = tokens + pos_emb

    # Transformer encoder
    for _ in range(depth):
        tokens = transformer_block(tokens, d_model, num_heads, mlp_dim, dropout=dropout)

    # Project back to patch pixels and depatchify to (H, W, C)
    tokens = L.Dense(patch_dim)(tokens)
    feat_map = Depatchify(H, W, C, patch_size=patch_size)(tokens)

    # Light conv decoder/head -> (H, W, 1)
    y = L.Conv2D(conv_head_filters, 3, padding="same", activation=_gelu)(feat_map)
    y = L.Conv2D(conv_head_filters, 3, padding="same", activation=_gelu)(y)
    out = L.Conv2D(1, 1, padding="same", activation=None)(y)  # per-pixel regression

    return tf.keras.Model(inputs=inp, outputs=out, name="TransformerSeg")


# Helpers
def zscore_from_train(x_train, *xs, eps=1e-6):
    mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std  = x_train.std(axis=(0, 1, 2), keepdims=True) + eps
    return tuple((x - mean) / std for x in (x_train,) + xs)

def fill_nan_and_mask(y):
    # y: (N, H, W, 1)
    mask = (~np.isnan(y))[..., 0].astype("float32")      # (N, H, W)
    y_filled = np.where(np.isnan(y), 0.0, y).astype("float32")
    return y_filled, mask
