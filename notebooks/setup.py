from pathlib import Path

import tensorflow as tf

from notebooks.utils import SmartDataGenerator, split_dataset, prepare_model


#####
# dataset setup
#####

SEED = 24052022
base_folder = Path(r"C:\Users\ML\Desktop\seba_preprocessed_fusion\numpy_160")
train_data, validation_data, test_data = split_dataset(base_folder, train_test_split=0.95, train_val_split=0.90, seed=SEED)

size = (160, 160, 160)
train_generator = SmartDataGenerator(train_data, dim=size, batch_size=1, seed=SEED)
validation_generator = SmartDataGenerator(validation_data, dim=size, batch_size=1, seed=SEED)
test_generator = SmartDataGenerator(test_data, dim=size, batch_size=1, seed=SEED)


#####
# model setup
#####
config = {
    'sim_param': 1.0,
    'lambda_param': 4.0,
    'gamma_param': 2.0,
    "is_final": True,
    "initial_epoch": 150,
    "epochs": 300,
    "steps_per_epoch": 100
}
config["name"] = f"{config['lambda_param']}_{config['gamma_param']}{'_final' if config['is_final'] else ''}"
config["base_dir"] = Path(f"../models/test_{config['name']}")
config["log_dir"] = str(config["base_dir"] / "logs")
config["history_filepath"] = str(config["base_dir"] / "history.csv")
config["checkpoints_dir"] = config["base_dir"] / "checkpoints"
config["checkpoints_filepath"] = str(config["checkpoints_dir"] / "best_weights_{epoch:04d}.ckpt")

model = prepare_model(inshape=size, sim_param=config['sim_param'], lambda_param=config['lambda_param'], gamma_param=config['gamma_param'])
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=config['log_dir'], histogram_freq=5, write_graph=True, write_images=True),
    tf.keras.callbacks.CSVLogger(filename=config['history_filepath'], separator=';', append=True),
    tf.keras.callbacks.EarlyStopping(patience=20, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.8, patience=15, min_lr=1e-10, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=config['checkpoints_filepath'], save_weights_only=True, save_best_only=True, verbose=1)
]

latest_checkpoint = tf.train.latest_checkpoint(config["checkpoints_dir"])
