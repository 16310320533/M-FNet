import tensorflow as tf
from tensorflow.keras import layers


class DomainAdaptiveStem(layers.Layer):
    """
    Hierarchical Visual Encoder Stem with Domain Adaptation capabilities.
    Maps raw grayscale NCCT pixel space into a 'pseudo-RGB' representation.
    """

    def __init__(self, name="domain_adaptive_stem", **kwargs):
        super(DomainAdaptiveStem, self).__init__(name=name, **kwargs)
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2 = layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv3 = layers.Conv2D(3, kernel_size=(3, 3), padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.Activation('relu')

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(inputs), training=training)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x), training=training)))
        return self.relu3(self.bn3(self.conv3(x), training=training))


class RelationAwareTabularEncoder(layers.Layer):
    """
    Transformer-Based Tabular Encoder.
    Re-conceptualizes discrete clinical indicators as a relational pseudo-sequence.
    """

    def __init__(self, num_heads=4, embed_dim=64, ff_dim=128, dropout_rate=0.1, name="tabular_encoder", **kwargs):
        super(RelationAwareTabularEncoder, self).__init__(name=name, **kwargs)
        self.feature_projection = layers.Conv1D(filters=embed_dim, kernel_size=1, activation='relu')
        self.embed_bn = layers.BatchNormalization()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.mha_dropout = layers.Dropout(dropout_rate)
        self.mha_norm = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_conv1 = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        self.ffn_conv2 = layers.Conv1D(filters=embed_dim, kernel_size=1)
        self.ffn_dropout = layers.Dropout(dropout_rate)
        self.ffn_norm = layers.LayerNormalization(epsilon=1e-6)
        self.final_dense = layers.Dense(256, activation='relu')
        self.final_bn = layers.BatchNormalization()

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = tf.expand_dims(inputs, axis=-1) if len(inputs.shape) == 2 else inputs
        x = self.embed_bn(self.feature_projection(x), training=training)

        attn_out = self.mha_dropout(self.mha(x, x), training=training)
        x1 = self.mha_norm(x + attn_out)

        ffn_out = self.ffn_dropout(self.ffn_conv2(self.ffn_conv1(x1)), training=training)
        seq_features = self.ffn_norm(x1 + ffn_out)

        aggregated = tf.concat([tf.reduce_mean(seq_features, axis=1), tf.reduce_max(seq_features, axis=1)], axis=1)
        return self.final_bn(self.final_dense(aggregated), training=training)


class TriPathwaySynergisticFusion(layers.Layer):
    """
    Concurrently executes bidirectional cross-attention, gated modality weighting,
    and direct feature preservation for robust multimodal integration.
    """

    def __init__(self, latent_dim=128, num_heads=4, name="tri_pathway_fusion", **kwargs):
        super(TriPathwaySynergisticFusion, self).__init__(name=name, **kwargs)
        self.img_proj = layers.Dense(latent_dim)
        self.tab_proj = layers.Dense(latent_dim)
        self.img_to_tab_mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=latent_dim)
        self.tab_to_img_mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=latent_dim)
        self.img_gate = layers.Dense(1, activation='sigmoid')
        self.tab_gate = layers.Dense(1, activation='sigmoid')

    def call(self, image_features: tf.Tensor, tabular_features: tf.Tensor, training: bool = False) -> tf.Tensor:
        h_i = self.img_proj(image_features)
        h_t = self.tab_proj(tabular_features)

        h_i_seq = tf.expand_dims(h_i, axis=1)
        h_t_seq = tf.expand_dims(h_t, axis=1)

        attn_i_to_t = tf.squeeze(self.img_to_tab_mha(query=h_i_seq, value=h_t_seq, key=h_t_seq), axis=1)
        attn_t_to_i = tf.squeeze(self.tab_to_img_mha(query=h_t_seq, value=h_i_seq, key=h_i_seq), axis=1)

        gated_i = h_i * self.img_gate(h_i)
        gated_t = h_t * self.tab_gate(h_t)

        return tf.concat([h_i, h_t, attn_i_to_t, attn_t_to_i, gated_i, gated_t], axis=1)