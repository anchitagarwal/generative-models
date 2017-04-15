# Generative Models
---
This project is a collection of various flavors of Generative Adversarial Networks, popularly known as GAN. The implementation is done in TensorFlow.

## About
---
Generative Adversarial Networks, or GAN, were first introduced by [Goodfellow, et al. in a NIPS 2014 paper](http://papers.nips.cc/paper/5423-generative-adversarial-nets), and since then has sparked a lot of interest in the deep learning research community.

GAN are based on game theoretic scenario where two neural nets, called *generator* and *discriminator*, compete against each to reach a Nash equilibrium. The generator learns a mapping function from noise to the original images, while the discriminator learns to identify images that come from the real world, or from the generator.