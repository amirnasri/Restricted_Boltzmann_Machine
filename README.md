# Restricted Boltzmann Machine
An implementation of a Bernoulli Restricted Boltzmann Machine in Python

`rbm.py` first loads   the   image   data   from   the   MNIST   data   set   file   to
memory.   Then   it   uses   the   Contrastive   Divergence   algorithm to train the RBM on the MNIST dataset. 
The RBM class
takes the number of hidden units, learning rate, number of inner Gibbs sampling
iterations, batch size, and number of epochs as input parameters. 
It   is   also   possible   to   provide   two   function   pointers   h_phv_generator   and
v_pvh_generator,   which   are   the   algorithms   (strategies)   used   by   the   RBM
implementation to sample p(h|v) and p(v|h) distributions, respectively. As a result,
the   algorithm   can   be   extended   to   use   more   general   distributions   such   as   the
exponential family distribution instead of the default Bernoulli distribution.

"mnist_pretrain.py" trains the
RBM on a number of symbols from the dataset and plots different images obtained
by   sampling   of   the   learned   distribution.

"unit_test.py" provides a class performs two test functions on the
RBM. The first test loads ten random images from the MNIST data set and trains an
RBM on each image. Then it calculates the error rate performance by calculating
the   average   absolute   error   between   the   image   from   the   dataset   and   the   image
reconstructed  by  the  RBM. This function  also  plots the actual and  reconstructed
images for visual comparison.
The   second   function   plots   the   weight   filters   obtained   by   training   the   RBM   for
different   hidden   units.   This   function   trains   the   RBM   on   ten   images   and   then
reshapes the obtained weight matrix W to plot the weight filters.
