import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from .layers import DomainAdaptiveStem, RelationAwareTabularEncoder, TriPathwaySynergisticFusion


def build_m2fnet(input_shape_img=(224, 224, 3), num_tabular_features=7) -> Model:
    """Constructs the complete Multimodal Medical Fusion Network (M2FNet)."""

    # Image Stream
    img_input = layers.Input(shape=input_shape_img, name="image_input")
    pseudo_rgb = DomainAdaptiveStem()(img_input)

    base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(56, 56, 3))
    base_resnet.trainable = False

    backbone_features = base_resnet(pseudo_rgb, training=False)
    x_img = layers.GlobalAveragePooling2D(name="img_gap")(backbone_features)
    x_img = layers.Dropout(0.6)(layers.BatchNormalization()(layers.Dense(512, activation='relu')(x_img)))
    img_features = layers.Dense(256, activation='relu', name="img_features_final")(x_img)

    # Tabular Stream
    tab_input = layers.Input(shape=(num_tabular_features,), name="tabular_input")
    tab_features = RelationAwareTabularEncoder()(tab_input)

    # Fusion Module
    fused_vector = TriPathwaySynergisticFusion()(img_features, tab_features)

    # Prediction Head
    x_out = layers.Dropout(0.4)(layers.BatchNormalization()(layers.Dense(128, activation='relu')(fused_vector)))
    x_out = layers.Dropout(0.2)(layers.Dense(64, activation='relu')(x_out))
    output = layers.Dense(1, activation='sigmoid', name="binary_prediction")(x_out)

    return Model(inputs=[img_input, tab_input], outputs=output, name="M2FNet")