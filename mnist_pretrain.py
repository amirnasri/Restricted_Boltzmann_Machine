import struct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from rbm import rbm
import random

def load_images(filename, n_samples):
    try:
        f = open("train-images.idx3-ubyte", "rb")
    
        magic_number = struct.unpack(">i", f.read(4))[0]
        n_image = struct.unpack(">i", f.read(4))[0]
        if (n_image < n_samples):
            raise ValueError("Number of samples %d is larger than the total number of images." % n_samples)
        n_row = struct.unpack(">i", f.read(4))[0]
        n_col = struct.unpack(">i", f.read(4))[0]
        image_size = n_row * n_col
        
        raw_images = struct.unpack_from(str(image_size * n_samples) + "B", f.read(image_size * n_samples))
        raw_images = np.reshape(np.array(raw_images), (n_row, n_col, n_samples), 'F').swapaxes(0, 1)
        return n_row, n_col, raw_images
        
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
    
    finally:
        f.close()


if __name__ == "__main__":
    # Number of images to train on
    n_samples = 10
    
    # Number of hidden units
    n_h = 1000
  
    # Size of mini-batchs
    batch_size = 10
    
    # Number of epochs
    n_epoch = 100 
    
    # Learning rate
    learning_rate = 0.1
    
    # Load images  
    n_row, n_col, raw_images = load_images("train-images.idx3-ubyte", n_samples)
    d_n = np.reshape(raw_images, (n_row * n_col, n_samples), 'F')
    
    # Random generator used by RBM
    #random_gen = np.random.RandomState(1234567890)
    rand_seed = int(100000 * random.random())
    random_gen = np.random.RandomState(rand_seed)
    rbm_ = rbm(n_hidden=n_h, learning_rate=learning_rate, n_epoch=n_epoch, batch_size=batch_size, rng=random_gen, verbose=True)
    
    model, gradient_vec = rbm_.pretrain(d_n)
    
    W = model['W']
    a = model['a']
    b = model['b']
    
    print "Generating plots..."
    # Plot gradient vector obtained from RBM as a function of iteration number
    plt.figure()
    plt.plot(gradient_vec)
    plt.grid(b=True)
    plt.ylabel('magnitude of gradient vector', fontsize=14)
    plt.xlabel('number of iterations', fontsize=14)
    
    # Plot images obtained by sampling the learned distribution 
    fig = plt.figure()
    fig.suptitle('Images obtained by sampling of the learned distribution')
    n_gibbs_iter = 20
    n_v = n_row * n_col
    for gen_num in range(50):
        plt.subplot(5, 10, gen_num)
        
        v = random_gen.random_sample((n_v, 1)) < 0.5
        h = rbm_.gen_h_phv(v, W, b)
        for __ in range(n_gibbs_iter):
            v = rbm_.gen_v_pvh(h, W, a)
            h = rbm_.gen_h_phv(v, W, b)
    
        plt.imshow(np.reshape(v, (n_row, n_col), 'F') * 255, cmap=cm.Greys_r)
        plt.axis('off')
    plt.show()
        
        