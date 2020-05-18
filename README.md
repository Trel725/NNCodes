## General description

This repository contains few Jupyter notebooks, sharing a common propertie of being useful for not only for machine learning experts but also for chemists. Still, we believe that the code can be useful for anyone, interested in ML.

## Files

1. backpropagation.ipynb

   This file contains implementation of the backpropagation algorithm in Julia language. The algorithm is implemented definitely not in the most efficient way, but tries to be as readable and understandable. Together with the following file it demonstrates how the NN are built from scratch.

2. nn_classifier.ipynb

   In this file backpropagation function from the previous one is used to train a simple NN on the Fisher irises dataset.

3. spectra.csv.gz
   
   This is a simple dataset of Raman spectra, used in by some of the scripts. It contains four classes of spectra, which are backgrund-subtracted and normalized to the 0...1 range. Class labels are given as the last of column of the dataset. Each spectrum consist of 2090 values in the range from 100 to 4278 reciprocal centimeters, i.e. to generate X-values following function can be used:

   ```np.linspace(100, 4278, 2090)```

4. ILSdata.csv 
   
  Taken from https://zenodo.org/record/3572359, described there.

## Notebooks

Directory notebooks contains few implementations of ML algorithm in Python with TensorFlow 2 framework. All examples are detaily commented.

1. classifier_feedforward.ipynb

   A simple feed-forward spectra classifier.

2. classifier_conv.ipynb

   This file implements 1D convolutional classifier and compares it with simple feed-forward one on the spectra dataset with background.

3. cnn_regression.ipynb

   A convolutional regressor, trained on ILS dataset.

4. siamese.ipynb

   Siamese neural network applied to the ILS dataset for both dimensionality reduction and metric learning [1].

5. variational_ae_mnist.ipynb
   
   Variational autoencoder [2] trained on MNIST dataset. Simple model, that able to generate new data.

6. variational_ae_spectra.ipynb

   Same for Raman spectra dataset.

7. cond_variational_ae_mnist.ipynb

   Conditional variational autoencoder trained on MNIST dataset. Style transfer functionality is demonstrated.

8. cond_variational_ae_spectra.ipynb

   Same for Raman spectra dataset.

9. wgan-gp_mnist.ipynb

   Wasserstein GAN, trained on MNIST dataset. Generates realistic digit images.

10. wgan-gp_spectra.ipynb 

   Same for Raman spectra dataset.

11. vae_wgan_larsen.ipynb

   Variational autoencoder - Wasserstein GAN, trained on MNIST dataset. Able to perform style transfer and generate realistic digit images. Uses penalizes hidden layer activation, according to [3].

## References

[1] Hadsell, Raia, Sumit Chopra, and Yann LeCun. "Dimensionality reduction by learning an invariant mapping." 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06). Vol. 2. IEEE, 2006.

[2] Doersch, Carl. "Tutorial on variational autoencoders." arXiv preprint arXiv:1606.05908 (2016).

[3] Larsen, Anders Boesen Lindbo, et al. "Autoencoding beyond pixels using a learned similarity metric." arXiv preprint arXiv:1512.09300 (2015).