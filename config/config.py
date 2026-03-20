import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Hyperparameters for M2FNet architecture."""
    input_shape_img: tuple = (224, 224, 3)
    num_tabular_features: int = 7
    tabular_embed_dim: int = 64
    transformer_heads: int = 4
    fusion_latent_dim: int = 128
    dropout_rate: float = 0.4

@dataclass
class TrainingConfig:
    """Hyperparameters for the training pipeline."""
    batch_size: int = 16
    epochs: int = 50
    initial_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    seed: int = 42

@dataclass
class PathConfig:
    """Directory paths for data demo and artifacts."""
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(base_dir, 'data demo')
    output_dir: str = os.path.join(base_dir, 'checkpoints')
    log_dir: str = os.path.join(base_dir, 'logs')

# Global configuration instance
model_cfg = ModelConfig()
train_cfg = TrainingConfig()
path_cfg = PathConfig()