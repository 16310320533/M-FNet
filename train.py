import os
import argparse
import logging
import tensorflow as tf
from models.builder import build_m2fnet
from utils.metrics import get_clinical_evaluation_metrics
from config.config import train_cfg, path_cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("M2FNet_Trainer")


def main():
    parser = argparse.ArgumentParser(description="M2FNet Training Pipeline")
    parser.add_argument('--epochs', type=int, default=train_cfg.epochs)
    parser.add_argument('--batch_size', type=int, default=train_cfg.batch_size)
    args = parser.parse_args()

    os.makedirs(path_cfg.output_dir, exist_ok=True)
    logger.info("Initializing Multimodal Fusion Framework...")

    model = build_m2fnet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=train_cfg.initial_learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=get_clinical_evaluation_metrics()
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(path_cfg.output_dir, "best_m2fnet_weights.h5"),
            monitor='val_auc', mode='max', save_best_only=True, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=train_cfg.early_stopping_patience, mode='max')
    ]

    logger.info("Pipeline ready. Execute model.fit() with instantiated MultimodalDataGenerator.")
    # model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)


if __name__ == "__main__":
    main()