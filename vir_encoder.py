import tensorflow as tf
from models.text_encoder import TransformerBlock

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class ViTEncoder(tf.keras.Model):
    def __init__(self, image_size, patch_size, num_layers, hidden_size, mlp_dim, num_heads):
        super(ViTEncoder, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patches = Patches(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, hidden_size)
        self.encoder_blocks = [TransformerBlock(hidden_size, num_heads, mlp_dim) for _ in range(num_layers)]

    def call(self, images, training):
        x = self.patches(images)
        x = self.patch_encoder(x)
        for block in self.encoder_blocks:
            x = block(x, training=training)
        return x
