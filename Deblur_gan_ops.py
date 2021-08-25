import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import datetime
import numpy as np
import tqdm
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from keras.models import Model
from models import generator_model, discriminator_model, generator_containing_discriminator_multiple_outputs
import PIL
from tkinter import *
from tkinter import filedialog


class GAN:
    image_shape = (256, 256, 3)
    vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False

    def __init__(self, limiter=10):
        self.blurred = None
        self.sharp = None
        self.BASE_DIR = None
        self.Path_to_weights = None
        self.save_dir = None
        self.input_dir = None
        self.RESHAPE = (256, 256)
        self.limiter = limiter

    @staticmethod
    def file_Gui(file_type, directory=True, multi=False):
        base = Tk()
        base.withdraw()
        base.geometry("150x150")
        if not directory:
            if multi:
                filepath = filedialog.askopenfilenames(title="Select {} files".format(file_type))
                base.destroy()
                return list(filepath)
            else:
                filepath = filedialog.askopenfilename(title="Select {} file".format(file_type))
                base.destroy()
                return filepath
        else:
            dir_path = filedialog.askdirectory(title="Select {} directory".format(file_type))
            base.destroy()
            return dir_path

    @staticmethod
    def is_an_image_file(filename):
        IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
        for ext in IMAGE_EXTENSIONS:
            if ext in filename:
                return True
        return False

    @staticmethod
    def list_image_files(directory):
        files = os.listdir(directory)
        files.sort()
        return [os.path.join(directory, f) for f in files]

    @staticmethod
    def load_image(path):
        img = PIL.Image.open(path)
        return img

    def preprocess_image(self, cv_img):
        cv_img = cv_img.resize(self.RESHAPE)
        img = np.array(cv_img)
        img = (img - 127.5) / 127.5
        return img

    @staticmethod
    def deprocess_image(img):
        img = img * 127.5 + 127.5
        return img.astype('uint8')

    def save_image(self, np_arr):
        self.save_dir = self.file_Gui('Save')
        img = np_arr * 127.5 + 127.5
        im = PIL.Image.fromarray(img)
        im.save(self.save_dir)

    def load_images(self, limit=True):
        self.blurred = self.file_Gui('Blurred images')
        self.sharp = self.file_Gui('Sharp images')
        all_A_paths, all_B_paths = self.list_image_files(self.blurred), self.list_image_files(self.sharp)
        images_A, images_B = [], []
        images_A_paths, images_B_paths = [], []
        length = all_B_paths[0:self.limiter]
        if not limit:
            length = all_B_paths
        for i in range(len(length)):
            img_A, img_B = self.load_image(all_A_paths[i]), self.load_image(all_B_paths[i])
            images_A.append(self.preprocess_image(img_A))
            images_B.append(self.preprocess_image(img_B))
            images_A_paths.append(all_A_paths[i])
            images_B_paths.append(all_B_paths[i])
        return {
            'A': np.array(images_A),
            'A_paths': np.array(images_A_paths),
            'B': np.array(images_B),
            'B_paths': np.array(images_B_paths)
        }

    def save_all_weights(self, d, g, epoch_number, current_loss):
        now = datetime.datetime.now()
        save_dir = os.path.join(self.BASE_DIR, '{}{}'.format(now.month, now.day))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
        d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)

    @tf.function
    def perceptual_loss(self, y_true, y_pred):
        res = K.mean(K.square(GAN.loss_model(y_true) - GAN.loss_model(y_pred)))
        return res

    @tf.function
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def train(self, batch_size, epoch_num, pretrained=False, critic_updates=5):
        data = self.load_images()
        y_train, x_train = data['B'], data['A']
        g = generator_model()
        if pretrained:
            self.Path_to_weights = self.file_Gui('Pretrained model', directory=False)
            g.load_weights(self.Path_to_weights)
        d = discriminator_model()
        d_on_g = generator_containing_discriminator_multiple_outputs(g, d)
        print(d_on_g.summary())
        d_opt = Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        d_on_g_opt = Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        d.trainable = True
        d.compile(optimizer=d_opt, loss=self.wasserstein_loss)
        d.trainable = False
        d_on_g.compile(optimizer=d_on_g_opt, loss=[self.perceptual_loss, self.wasserstein_loss], loss_weights=[100, 1])
        d.trainable = True
        # Creating fakes
        output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))
        log_path = './logs'
        tensorboard_callback = TensorBoard(log_path)
        print("Starting Training:")
        for epoch in tqdm.tqdm(range(epoch_num)):
            permutated_indexes = np.random.permutation(x_train.shape[0])

            d_losses = []
            d_on_g_losses = []
            for index in range(int(x_train.shape[0] / batch_size)):
                batch_indexes = permutated_indexes[index * batch_size:(index + 1) * batch_size]
                image_blur_batch = x_train[batch_indexes]
                image_full_batch = y_train[batch_indexes]

                generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

                for _ in range(critic_updates):
                    d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                    d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                    d_losses.append(d_loss)
                d.trainable = False
                d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
                d_on_g_losses.append(d_on_g_loss)
                d.trainable = True
            print('\nEpoch: {} Discriminator loss: {} Model loss:{}\n'.format(epoch, np.mean(d_losses),
                                                                              np.mean(d_on_g_losses)))
            with open('log.txt', 'a+') as f:
                f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))
            if epoch_num - epoch == 1:
                self.save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))

    def test(self, batch_size):
        self.Path_to_weights = self.file_Gui('Pretrained model', directory=False)
        data = self.load_images()
        y_test, x_test = data['B'], data['A']
        g = generator_model()
        g.load_weights(self.Path_to_weights)
        generated_images = g.predict(x=x_test, batch_size=batch_size)
        generated = np.array([self.deprocess_image(img) for img in generated_images])
        x_test = self.deprocess_image(x_test)
        y_test = self.deprocess_image(y_test)

        for i in range(generated_images.shape[0]):
            y = y_test[i, :, :, :]
            x = x_test[i, :, :, :]
            img = generated[i, :, :, :]
            output = np.concatenate((y, x, img), axis=1)
            im = PIL.Image.fromarray(output.astype(np.uint8))
            im.save('results{}.png'.format(i))

    def deblur_dir(self):
        self.Path_to_weights = self.file_Gui('Model', directory=False)
        self.input_dir = self.file_Gui('Images to deblur')
        self.save_dir = self.file_Gui('deblurred images')
        g = generator_model()
        g.load_weights(self.Path_to_weights)
        for image_name in os.listdir(self.input_dir):
            image = np.array([self.preprocess_image(self.load_image(os.path.join(self.input_dir, image_name)))])
            x_test = image
            generated_images = g.predict(x=x_test)
            generated = np.array([self.deprocess_image(img) for img in generated_images])
            x_test = self.deprocess_image(x_test)
            for i in range(generated_images.shape[0]):
                x = x_test[i, :, :, :]
                img = generated[i, :, :, :]
                output = np.concatenate((x, img), axis=1)
                im = PIL.Image.fromarray(output.astype(np.uint8))
                im.save(os.path.join(self.save_dir, image_name))

    def deblur_image(self):
        self.Path_to_weights = self.file_Gui('Model', directory=False)
        g = generator_model()
        g.load_weights(self.Path_to_weights)
        path = self.file_Gui('Image to deblur', directory=False)
        image = np.array([self.preprocess_image(self.load_image(path))])
        prelim = g.predict(image)
        result = self.deprocess_image(prelim)
        im = PIL.Image.fromarray(result[0, :, :, :])
        im.save(os.path.join(self.file_Gui('path to save'), "deblurred{}.jpg".format(random.randint(0, 100))))
