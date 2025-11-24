# LPSO-SAGAN: Automatic Generation of Lacquer Painting Design

This repository contains a MATLAB implementation of a **Self-Attention Generative Adversarial Network (SAGAN)** optimized by a **Hybrid Learning Automata and Particle Swarm Optimization (LPSO)** algorithm. 
This model is designed to generate Lacquer Painting designs for handicrafts. This code is specifically implemented for **MATLAB R2020a**. 

## Requirements
*   **MATLAB R2020a**
*   **Deep Learning Toolbox**
*   **Parallel Computing Toolbox** (Recommended for GPU acceleration)
*   A CUDA-capable NVIDIA GPU is highly recommended for training GANs.

    **Prepare the Dataset:**
    *   Create a folder named `dataset` in the same directory as the scripts.
    *   Place your Lacquer Painting images (or any art dataset) inside this folder.
    *   *Note:* The code will automatically resize images to 128x128.

## Technical Implementation Details
To bypass MATLAB R2020a's strict `dlnetwork` validation which crashes when reshaping labeled arrays inside custom layers, the Self-Attention mechanism was implemented using a **Graph connection strategy**. Therefore, instead of a single complex `SelfAttentionLayer`, we wire together standard layers via `layerGraph`.