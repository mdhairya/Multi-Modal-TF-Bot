import tensorflow as tf
from models.vit_encoder import ViTEncoder
from models.text_encoder import TextEncoder

class VQAModel(tf.keras.Model):
    def __init__(self, config):
        super(VQAModel, self).__init__()
        # Visual Pathway
        self.vit = ViTEncoder(
            image_size=config.getint('MODEL_VIT', 'image_size'),
            patch_size=config.getint('MODEL_VIT', 'patch_size'),
            num_layers=config.getint('MODEL_VIT', 'num_layers'),
            hidden_size=config.getint('MODEL_VIT', 'hidden_size'),
            mlp_dim=config.getint('MODEL_VIT', 'mlp_dim'),
            num_heads=config.getint('MODEL_VIT', 'num_heads')
        )
        
        # Text Pathway
        self.text_enc = TextEncoder(
            vocab_size=config.getint('TRAINING', 'vocab_size'),
            maxlen=config.getint('TRAINING', 'max_seq_length'),
            embed_dim=config.getint('MODEL_TEXT', 'embed_dim'),
            num_heads=config.getint('MODEL_TEXT', 'num_heads'),
            num_layers=config.getint('MODEL_TEXT', 'num_layers'),
            ff_dim=config.getint('MODEL_TEXT', 'ff_dim')
        )
        
        # Cross Attention (Text queries Image)
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=config.getint('MODEL_TEXT', 'embed_dim')
        )
        
        # Output Classifier
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation="gelu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(config.getint('TRAINING', 'num_answers'), activation="softmax")
        ])

    def call(self, inputs, training=False):
        images, questions = inputs
        
        # Encode modalities
        img_features = self.vit(images, training=training)    # Shape: (batch, num_patches, hidden_size)
        text_features = self.text_enc(questions, training=training) # Shape: (batch, seq_len, embed_dim)
        
        # Cross modal fusion: text attends to image
        fused = self.cross_attention(
            query=text_features, 
            value=img_features, 
            key=img_features, 
            training=training
        )
        
        pooled = self.global_pool(fused)
        return self.classifier(pooled)
