# GAN Implementation Using XOR Gate
#### Project by Hardik Ajmani | Proposal by Mrinal Wahal  
**Date:** 17th July 2019

***

## Overview

This project demonstrates a simple implementation of a Generative Adversarial Network (**GAN**) from scratch using **NumPy** in Python.  
The GAN is trained to learn the behavior of an XOR gate extended to 3 inputs, making it a useful didactic example for introductory GAN concepts.

***

## Table of Contents

- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Discriminator](#discriminator)
  - [Generator](#generator)
- [How to Run](#how-to-run)
- [Code Components](#code-components)
- [Results](#results)

***

## Project Structure

- `xor_gan.py` — The main implementation file (provided code)
- `README.md` — This documentation

***

## How It Works

### Problem: XOR Function (3-Input)

- **Input:** 3 binary values (e.g., `0 1 1`)
- **Output:** The XOR of those inputs (e.g., `0⊕1⊕1 = 0`)

### Model Overview

This implementation consists of two neural networks:
- **Discriminator** — Learns to distinguish "real" XOR outputs from "fake" outputs produced by the Generator.
- **Generator** — Tries to generate outputs that can fool the Discriminator.

#### Discriminator

- **Input:** 3-bit vector
- **Architecture:** 3 → 6 → 1 neuron layers (with sigmoid activations)
- **Output:** 1 (real) or 0 (fake) — predicts whether input is a true XOR or not.

#### Generator

- **Input:** Random 3-dimensional noise
- **Architecture:** 3 → 6 → 3 neuron layers (with sigmoid activations)
- **Output:** 3-bit vector, simulating a distribution similar to true XOR inputs.

#### Data

- Uses every possible 3-bit vector as inputs (total of 8 combinations).

***

## How to Run

1. **Install dependencies**

   Make sure you have Python 3 and NumPy installed:

   `pip install numpy`

2. **Run the script**
   `python xor_gan.py`

***

## Code Components

**Key Functions and Classes:**

- `sigmoid(x, derivative=False)`  
Sigmoid activation function (and its derivative).

- `think(input, layer_1, layer_2, bias_1, bias_2)`  
Forward pass for a 2-layer neural network.

- `to_binary(mat)`  
Converts outputs >0.5 to 1 and <0.5 to 0.

- `disc_train(...)`  
Trains the Discriminator using gradient descent and binary cross-entropy loss.

- `gen_train(...)`  
Trains the Generator with the aim to fool the Discriminator.

***

## Results

- The discriminator quickly learns to separate true XORs from random input.
- The generator then iteratively improves, producing more "realistic" samples as judged by the discriminator.

Sample output:

Before training
[[1. 1. 1. 1. 1. 1. 1. 1.]]

Training Discriminator
After training
[[0. 1. 1. 0. 1. 0. 0. 1.]]

Before training generative network
[[0. 1. 0. 1. 1. 1. 0. 1.]]

Training Generator
At iteration: 0, Loss: 5.491045130434293
At iteration: 1000, Loss: 1.143342139020729
...
After training generative network
[[...]]

(The output will vary slightly due to random initialization.)

***

## Notes

- The current implementation is intended for educational and demonstration purposes, not for practical GAN use cases or large datasets.
- Hyperparameters (learning rate, epochs) can be tuned for different results.
- For deeper understanding, see inline comments in `xor_gan.py`.

***


