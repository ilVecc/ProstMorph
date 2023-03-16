from notebooks.setup import train_generator, validation_generator, config, model, callbacks, latest_checkpoint

if latest_checkpoint is not None:
    model.load_weights(latest_checkpoint)

config["base_dir"].mkdir(exist_ok=True)
hist = model.fit(
    train_generator, validation_data=validation_generator,
    epochs=config['epochs'], steps_per_epoch=config['steps_per_epoch'], initial_epoch=config['initial_epoch'],
    callbacks=callbacks,
    verbose=2
)

import matplotlib.pyplot as plt


def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


plot_history(hist)
