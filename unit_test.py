'''
This module implements unit tests for the RBM.
'''
import numpy as np
import random
from rbm import rbm
from mnist_pretrain import load_images
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class unit_test:
    def test_error_performance(self):
        """
        This function loads 10 random images from the MNIST data set and trains an RBM 
        on each image. Then it calculates the error rate performance by calculating 
        the average absolute error between the image from the dataset and the image reconstructed by the RBM.
        This function also plots the actual and reconstructed images for visual comparison.
        """
        # Load 1000 images and choose 10 of them at random
        n_image = 10 
        n_row, n_col, raw_images = load_images("train-images.idx3-ubyte", 1000)
        d_n = np.reshape(raw_images[:, :, random.sample(xrange(1000), n_image)], (n_row * n_col, n_image), 'F')

        # Number of hidden units
        n_h = 1000
      
        # Random generator used by RBM
        rand_seed = int(100000 * random.random())
        random_gen = np.random.RandomState(rand_seed)
        rbm_ = rbm(n_hidden=n_h, learning_rate=0.1, n_epoch=20, batch_size=1, rng=random_gen)
        
        
        # Plot images obtained by sampling the learned distribution 
        fig = plt.figure()
        fig.suptitle("Comparison of random images from the dataset with the corresponding images\n" +  
            "obtained by sampling of the distribution learned by RBM. Top row: images from the dataset.\n" + 
            "Bottom row: images reconstructed by RBM.")
        n_gibbs_iter = 20
        n_v = n_row * n_col
        for image_num in range(n_image):
        
            print "Training RBM for image number %d:" % (image_num+1)
            
            model, gradient_vec = rbm_.pretrain(d_n[:, image_num].reshape(n_v, 1))
        
            W = model['W']
            a = model['a']
            b = model['b']
            
            v = random_gen.random_sample((n_v, 1)) < 0.5
            h = rbm_.gen_h_phv(v, W, b)
            for __ in range(n_gibbs_iter):
                v = rbm_.gen_v_pvh(h, W, a)
                h = rbm_.gen_h_phv(v, W, b)
        
            reconstructed_image = np.reshape(v, (n_row, n_col), 'F')
            data_image = (np.reshape(d_n[:, image_num], (n_row, n_col), 'F') + 0.5) > 128
            
            error = np.mean(np.abs(data_image - reconstructed_image)) 
            print "\terror performance = %f\n" % error
            
            plt.subplot(2, 10, image_num + 1)
            plt.imshow(data_image * 255, cmap=cm.Greys_r)
            plt.axis('off')
            plt.subplot(2, 10, image_num + 11)
            plt.imshow(reconstructed_image * 255, cmap=cm.Greys_r)
            plt.axis('off')
            
    def plot_filters(self):
        """
        This function plots the weight filters obtained after training the RBM. 
        """
        # Load images
        n_image = 10
        n_row, n_col, raw_images = load_images("train-images.idx3-ubyte", n_image)
        d_n = np.reshape(raw_images, (n_row * n_col, n_image), 'F')

        # Number of hidden units
        n_h = 1000
      
        # Random generator used by RBM
        rand_seed = int(100000 * random.random())
        random_gen = np.random.RandomState(rand_seed)
        rbm_ = rbm(n_hidden=n_h, learning_rate=0.1, n_epoch=500, batch_size=n_image, rng=random_gen)
        
        print "Generating weight filters..."
        
        model, gradient_vec = rbm_.pretrain(d_n)
    
        W = model['W']
        a = model['a']
        b = model['b']
        
        n_gibbs_iter = 20
        n_v = n_row * n_col
        v = random_gen.random_sample((n_v, 1)) < 0.5
        h = rbm_.gen_h_phv(v, W, b)
        for __ in range(n_gibbs_iter):
            v = rbm_.gen_v_pvh(h, W, a)
            h = rbm_.gen_h_phv(v, W, b)
    
        fig = plt.figure()
        W = np.reshape(W, (28, 28 * 1000), 'F')
        reshaped_image = np.zeros((25 * 28, 40 * 28))
        for i in range(25):
            reshaped_image[28 * i:28* (i+1), :] = W[:, 40 * 28 * i: 40 * 28  * (i+1)]
        min_val = np.min(reshaped_image)
        max_val = np.max(reshaped_image)
        plt.imshow((reshaped_image - min_val)/(max_val - min_val) * 255, cmap=cm.Greys_r)
        fig.suptitle("Weight filters obtained after training the RBM")

            
if (__name__ == "__main__"):
    unit_test_ = unit_test()
    unit_test_.test_error_performance()
    unit_test_.plot_filters()
    plt.show()
