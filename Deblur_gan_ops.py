import os
import random
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
from Video_functions import play_multiple
from tkinter import messagebox


class GAN:
    # Input shape for loss model, matches the input for the main GAN model
    image_shape = (256, 256, 3)
    # Create loss model, create outside the loss function to avoid the error: ValueError: tf.function-decorated function
    # tried to create variables on non-first call
    vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False

    def __init__(self, limiter=10):
        """
        :param limiter: This is due to lack of processing power, limits the input images to 10
        """
        self.blurred = None
        self.sharp = None
        self.BASE_DIR = None
        self.Path_to_weights = None
        self.save_dir = None
        self.input_dir = None
        self.RESHAPE = (256, 256)
        self.limiter = limiter

    @staticmethod
    def file_Gui(file_type, ext=None, directory=True, multi=False):
        """
        :param file_type: Short string describing a the desired file type i.e "Text files"
        :param ext: Extension i.e 'txt', directory must be False. Default is None(all files will be shown)
        :param directory: Select a directory instead of a single file
        :param multi: Select multiple files at ones, returns a list of strings of selected paths
        :return: Returns a Tkinter GUI for file selection, root window is hidden only dialog is open
        """
        base = Tk()
        # Hide root window
        base.withdraw()
        if not directory:
            if multi:
                filepath = filedialog.askopenfilenames(title="Select {} files".format(file_type),
                                                       filetypes=(("{} file".format(file_type), '*.{}'.format(ext)),
                                                                  ("All files", '*.*')))
                # Destroy base after selection is done
                base.destroy()
                return list(filepath)
            else:
                filepath = filedialog.askopenfilename(title="Select {} file".format(file_type),
                                                      filetypes=(("{} file".format(file_type), '*.{}'.format(ext)),
                                                                 ("All files", '*.*')))
                base.destroy()
                return filepath
        else:
            dir_path = filedialog.askdirectory(title="Select {} directory".format(file_type))
            base.destroy()
            return dir_path

    @staticmethod
    def tk_inquire(question):
        """
        :param question: Question to ask
        :return: True or False depending on selection
        """
        root = Tk()
        root.withdraw()
        ans = messagebox.askyesno(message=question)
        root.destroy()
        return ans

    @staticmethod
    def list_image_files(directory):
        """
        :param directory: Image directory
        :return: List of single image paths
        """
        files = os.listdir(directory)
        files.sort()
        return [os.path.join(directory, f) for f in files]

    @staticmethod
    def load_image(path):
        """
        :param path: Path to image usually by iterating through output of list_image_files() function
        :return: pillow image array, it is important that all inputs are in this format
        """
        img = PIL.Image.open(path)
        return img

    def preprocess_image(self, cv_img):
        """
        :param cv_img: PIL image array. If unsure, highly recommend you use PIL.Image.fromarray() function before input.
        :return: Returns normalized image array with values ranging from 0-1
        """
        cv_img = cv_img.resize(self.RESHAPE)
        img = np.array(cv_img)
        img = (img - 127.5) / 127.5
        return img

    @staticmethod
    def deprocess_image(img):
        """
        :param img: output from GAN model(Numpy array)
        :return: Return image to its original array format before normalizing
        """
        img = img * 127.5 + 127.5
        return img.astype('uint8')

    def save_image(self, np_arr):
        """
        :param np_arr: Numpy array to save as an image
        :return: Saved image
        """
        self.save_dir = self.file_Gui('Save')
        img = np_arr * 127.5 + 127.5
        im = PIL.Image.fromarray(img)
        im.save(self.save_dir)

    def load_images(self, limit=False):
        """
        :param limit: Whether to limit the number of input images from a directory. Controlled by self.limiter variable.
        :return: Returns a tuple of blurred(A) and sharp(B) images for GAN input
        """
        self.blurred = self.file_Gui('Blurred images')
        self.sharp = self.file_Gui('Sharp images')
        all_A_paths, all_B_paths = self.list_image_files(self.blurred), self.list_image_files(self.sharp)
        images_A, images_B = [], []
        images_A_paths, images_B_paths = [], []
        if limit:
            length = all_B_paths[0:self.limiter]
        else:
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
        """
        :param d: Discriminator model
        :param g: Generator model
        :param epoch_number: Number of epochs model has been trained
        :param current_loss: Final loss of the GAN model
        :return: Save model weights in .h5 file format
        """
        # ask for save directory
        self.BASE_DIR = self.file_Gui('Save Weights')
        # Get date today
        now = datetime.datetime.now()
        save_dir = os.path.join(self.BASE_DIR, '{}{}'.format(now.month, now.day))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
        d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)

    @tf.function
    def perceptual_loss(self, y_true, y_pred):
        """
        :param y_true: Sharp image array
        :param y_pred: Image array from generator
        :return: GAN loss from the class variable loss model
        """
        res = K.mean(K.square(GAN.loss_model(y_true) - GAN.loss_model(y_pred)))
        return res

    @tf.function
    def wasserstein_loss(self, y_true, y_pred):
        """
        :param y_true: Sharp image array
        :param y_pred: Generator predicted output array
        :return: Loss value
        """
        return K.mean(y_true * y_pred)

    def train(self, batch_size, epoch_num, pretrained=False, critic_updates=5):
        """
        :param batch_size: Desired batch size.
        :param epoch_num: Number of epochs to train
        :param pretrained: If you are using pretrained weights
        :param critic_updates:
        :return: Train GAN model
        """
        data = self.load_images()
        y_train, x_train = data['B'], data['A']
        g = generator_model()
        d = discriminator_model()
        if pretrained:
            self.Path_to_weights = self.file_Gui('Pretrained Generator', ext='h5', directory=False)
            g.load_weights(self.Path_to_weights)
            if self.tk_inquire("Do you have pretrained weights for the discriminator?"):
                d_weights = self.file_Gui('Pretrained Discriminator', ext='h5', directory=False)
                d.load_weights(d_weights)
        d_on_g = generator_containing_discriminator_multiple_outputs(g, d)
        print(d_on_g.summary())
        print('\n')
        d_opt = Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        d_on_g_opt = Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        d.trainable = True
        d.compile(optimizer=d_opt, loss=self.wasserstein_loss)
        d.trainable = False
        d_on_g.compile(optimizer=d_on_g_opt, loss=[self.perceptual_loss, self.wasserstein_loss], loss_weights=[100, 1])
        d.trainable = True
        # Creating fakes
        output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))
        # Create progress bar
        prog = tqdm.tqdm(total=epoch_num, desc="Starting Training")
        epoch = 1
        while epoch <= epoch_num:
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
            with open('log.txt', 'a+') as f:
                f.write('Epoch: {} Discriminator loss: {} Model loss: {}\n'.format(epoch, np.mean(d_losses),
                                                                                   np.mean(d_on_g_losses)))
            prog.update(1)

            if epoch_num == epoch:
                self.save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))
            epoch += 1
        prog.close()

    def test(self, batch_size):
        """
        :param batch_size: Batch size to to test at a time
        :return: Test your model, Generator restored images on a side to side with original for comparison
        """
        self.Path_to_weights = self.file_Gui('Pretrained model', ext='h5', directory=False)
        self.save_dir = self.file_Gui('Test Results')
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
            im.save(os.path.join(self.save_dir, 'GAN_results{}.png'.format(i)))

    def deblur_directory(self):
        """
        :return: Deblur all images in a directory
        """
        self.Path_to_weights = self.file_Gui('Model', ext='h5', directory=False)
        self.input_dir = self.file_Gui('Images to deblur')
        self.save_dir = self.file_Gui('save deblurred images')
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

    def deblur_image(self, save=False, outputArray=False):
        """
        :param save: save deblurred image
        :param show: Default, show deblurred image
        :return: GAN deblurred image
        """
        self.Path_to_weights = self.file_Gui('Model', ext='h5', directory=False)
        g = generator_model()
        g.load_weights(self.Path_to_weights)
        path = self.file_Gui('Image to deblur', ext='jpg', directory=False)
        image = np.array([self.preprocess_image(self.load_image(path))])
        prelim = g.predict(image)
        result = self.deprocess_image(prelim)
        im = PIL.Image.fromarray(result[0, :, :, :])
        if save:
            im.save(os.path.join(self.file_Gui('path to save'), "deblurred{}.jpg".format(random.randint(0, 100))))
        im2 = self.deprocess_image(image)
        disp = PIL.Image.fromarray(np.hstack((im2[0, :, :, :], result[0, :, :, :])))
        disp.show()
        if outputArray:
            return result

    def video_deblur(self, play=True):
        """
        :param play: Default True, whether to play video after deblurring. Plays side by side with original
        :return: Returns a saved deblurred video
        """
        self.Path_to_weights = self.file_Gui('Model', ext='h5', directory=False)
        mode = self.file_Gui('Blurred Video', ext='mp4', directory=False)
        path, name = os.path.split(mode)
        vid_out = os.path.join(path, "GAN_Sharpened_{}_{}.mp4".format(random.randint(0, 100), name))
        g = generator_model()
        g.load_weights(self.Path_to_weights)
        cappy = cv2.VideoCapture(mode)
        if not cappy.isOpened():
            print("Error opening video file")
            exit()
        # Get the total number of frames
        length = int(cappy.get(cv2.CAP_PROP_FRAME_COUNT))
        print('The input video has %d frames.' % length)
        # Get video speed in frames per second
        speed = int(cappy.get(cv2.CAP_PROP_FPS))
        print('The input video plays at %d fps.' % speed)
        # Read until video is completed
        frame_width = int(cappy.get(3))
        frame_height = int(cappy.get(4))
        # define codec and create VideoWriter object
        out = cv2.VideoWriter(vid_out, cv2.VideoWriter_fourcc(*'mp4v'), speed,
                              (frame_width, frame_height))
        count = 0
        prog_bar = tqdm.tqdm(total=length, desc="Deblurring")
        while cappy.isOpened():
            ret, frame = cappy.read()
            if ret:
                # Always make sure your frame is a PIL image array before processing it
                frame = PIL.Image.fromarray(frame)
                image = np.array([self.preprocess_image(frame)])
                prelim = g.predict(image)
                frame2 = self.deprocess_image(prelim)
                im = cv2.resize(frame2[0, :, :, :], (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                out.write(im)
                prog_bar.update(1)
                # press `q` to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('(Q) pressed exiting...')
                    break
            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            count += 1
            if length != -1:
                if count >= length:
                    break
        prog_bar.close()
        cappy.release()
        out.release()
        cv2.destroyAllWindows()
        if play:
            play_multiple(mode, vid_out)
