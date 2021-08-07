import tensorflow as tf
from tensorflow.keras import layers
import os
from IPython import display
import time
import PIL.Image as Pimg
from numpy import asarray
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(3, use_bias=False, input_shape=(128, 128, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(3, (5, 5), strides=(1, 1), padding='same', input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(3, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(noise, images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=False)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(train_dataset, label_dataset, seed, epochs):
    for epoch in range(epochs):
        start = time.time()
        for train_batch, label_batch in zip(train_dataset, label_dataset):
            train_step(train_batch, label_batch)
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    path = r"C:\Users\Dennis Pkemoi\Desktop\VidopsMemBank\Epoch_Images" + '/' + 'image_at_epoch_{:04d}.jpg'.format(
        epoch)
    tf.keras.preprocessing.image.save_img(path, predictions[0] * 127.5 + 127.5, data_format='channels_last')


if __name__ == "__main__":
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    checkpoint_dir = r"C:\Users\Dennis Pkemoi\Desktop\VidopsMemBank\training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator, discriminator=discriminator)
    EPOCHS = 50
    BATCH_SIZE = 32
    BUFFER_SIZE = 60000

    blurred_path = r"C:\Users\Dennis Pkemoi\Desktop\VidopsMemBank\Dataset\archive\motion_blurred"
    sharp_path = r"C:\Users\Dennis Pkemoi\Desktop\VidopsMemBank\Dataset\archive\sharp"
    train_images = [(asarray(Pimg.open(blurred_path + '/' + i).resize((128, 128)), dtype=np.float32) - 127.5) / 127.5
                    for i in os.listdir(blurred_path)]
    label_images = [(asarray(Pimg.open(sharp_path + '/' + i).resize((128, 128)), dtype=np.float32) - 127.5) / 127.5 for
                    i in os.listdir(sharp_path)]
    blurred_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    sharp_dataset = tf.data.Dataset.from_tensor_slices(label_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test = (asarray(Pimg.open(blurred_path + '/' + os.listdir(blurred_path)[0]).resize((128, 128)),
                    dtype=np.float32) - 127.5) / 127.5
    print('Test image:', os.listdir(blurred_path)[0])
    train(blurred_dataset, sharp_dataset, test.reshape(1, 128, 128, 3), EPOCHS)
