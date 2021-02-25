#!/usr/bin/env python3
# The Core of the image reconstruction model.

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Input, Reshape, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

# Some contants for the GAN model.
IMAGE_SHAPE: tuple = (32, 32, 3)
LATENT_DIMENSIONS: int = 100
(x, y), (_, _) = keras.datasets.cifar10.load_data()
x = x[y.flatten() == 8]


class Generator(Sequential):
    """The Generator class used for this GAN Model.

    Args:
        Sequential (keras.Sequential): Sequential provides training and inference features on this model.

    Returns:
        Model: Returns the generator model via the build() function.
    """
    appliers: list = [
        Dense(
            128 * 8 * 8,
            activation="relu",
            input_dim=LATENT_DIMENSIONS,
        ),
        Reshape((8, 8, 128)),
        UpSampling2D(),
        Conv2D(
            128,
            kernel_size=3,
            padding="same",
        ),
        BatchNormalization(momentum=0.78),
        Activation("relu"),
        UpSampling2D(),
        Conv2D(64, kernel_size=3, padding="same"),
        BatchNormalization(momentum=0.78),
        Activation("relu"),
        Conv2D(3, kernel_size=3, padding="same"),
        Activation("tanh"),
    ]

    def __init__(_) -> None:
        super().__init__()

    def build(self) -> Model:
        for applier in Generator.appliers:
            self.add(applier)
        noise = Input(shape=(LATENT_DIMENSIONS, ))
        image = self(noise)
        return Model(noise, image)


class Discriminator(Sequential):
    """The Discriminator class used for this GAN Model.

    Args:
        Sequential (keras.Sequential): Sequential provides training and inference features on this model.

    Returns:
        Model: Returns the discriminator model via the build() function.
    """
    appliers: list = [
        Conv2D(
            32,
            kernel_size=3,
            strides=2,
            input_shape=IMAGE_SHAPE,
            padding="same",
        ),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv2D(64, kernel_size=3, strides=2, padding="same"),
        ZeroPadding2D(padding=((0, 1), (0, 1))),
        BatchNormalization(momentum=0.82),
        LeakyReLU(alpha=0.25),
        Dropout(0.25),
        Conv2D(128, kernel_size=3, strides=2, padding="same"),
        BatchNormalization(momentum=0.82),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv2D(256, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.25),
        Dropout(0.25),
        Flatten(),
        Dense(1, activation='sigmoid'),
    ]

    def __init__(_) -> None:
        super().__init__()

    def build(self) -> Model:
        for applier in Discriminator.appliers:
            self.add(applier)
        image = Input(shape=IMAGE_SHAPE)
        validity = self(image)

        return Model(image, validity)


class GANModel(object):
    """The GAN Model class."""
    def __init__(self, x) -> None:
        super().__init__()
        # Building and compiling the discriminator
        discriminator = Discriminator().build()
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        # Making the Discriminator untrainable so that the generator can learn from fixed gradient
        discriminator.trainable = False

        generator = Generator().build()

        # Defining the input for the generator and generating the images
        z = Input(shape=(LATENT_DIMENSIONS, ))
        image = generator(z)

        # Checking the validity of the generated image
        valid = discriminator(image)

        # Defining the combined model of the Generator and the Discriminator
        combined_network = Model(z, valid)
        combined_network.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

        num_epochs = 15000
        batch_size = 32
        display_interval = 100

        # Normalizing the input
        x = (x / 127.5) - 1.

        # efining the Adversarial ground truths
        valid = np.ones((batch_size, 1))

        # Adding some noise
        valid += 0.05 * np.random.random(valid.shape)
        fake = np.zeros((batch_size, 1))
        fake += 0.05 * np.random.random(fake.shape)

        for epoch in range(num_epochs):
            print("EPOCH=", epoch)

            # Training the Discriminator
            # Sampling a random half of images
            index = np.random.randint(0, x.shape[0], batch_size)
            images = x[index]

            # Sampling noise and generating a batch of new images
            noise = np.random.normal(0, 1, (batch_size, LATENT_DIMENSIONS))
            generated_images = generator.predict(noise)

            # Training the discriminator to detect more accurately
            # whether a generated image is real or fake
            discm_loss_real = discriminator.train_on_batch(images, valid)
            discm_loss_fake = discriminator.train_on_batch(generated_images, fake)

            # discriminator loss returned here
            discm_loss = 0.5 * np.add(discm_loss_real, discm_loss_fake)
            print("discm_loss=", discm_loss)

            # Training the Generator

            # Training the generator to generate images which pass the authenticity test

            # Generator loss returned here
            genr_loss = combined_network.train_on_batch(noise, valid)
            print("genr_loss=", genr_loss)

            # Tracking the progress
            if epoch % display_interval == 0:
                self.display_images(generator)

    def render_result(generator: Generator):
        # Plotting some of the original images
        s = x[:40]
        s = 0.5 * s + 0.5
        _, ax = plt.subplots(5, 8, figsize=(16, 10))
        for i, image in enumerate(s):
            ax[i // 8, i % 8].imshow(image)
            ax[i // 8, i % 8].axis('off')

        plt.show()

        # Plotting some of the last batch of generated images
        noise = np.random.normal(size=(40, LATENT_DIMENSIONS))
        generated_images = generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5
        _, ax = plt.subplots(5, 8, figsize=(16, 10))
        for i, image in enumerate(generated_images):
            ax[i // 8, i % 8].imshow(image)
            ax[i // 8, i % 8].axis('off')

        plt.show()

    @staticmethod
    def display_images(generator: Generator) -> None:
        """Render the reconstructied image.

        Args:
            generator (Generator): Pass your generator object.
        """
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, LATENT_DIMENSIONS))
        generated_images = generator.predict(noise)

        # Scaling the generated images
        generated_images = 0.5 * generated_images + 0.5

        _, axs = plt.subplots(r, c)
        count = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(generated_images[count, :, :, ])
                axs[i, j].axis('off')
                count += 1
        plt.show()
        plt.close()


if __name__ == "__main__":
    model = GANModel(x)
