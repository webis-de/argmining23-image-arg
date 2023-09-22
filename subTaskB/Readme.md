# Multimodal Persuasiveness Classifier

This repository contains code to train and evaluate a multimodal persuasiveness classification model using the CLIP (Contrastive Language-Image Pre-training) model. The goal of this code is to classify the persuasiveness of textual content with associated images.

## Introduction

The codebase provides the following main functionalities:

- **Training**: It trains a downstream classification model for persuasiveness using both text and image inputs.
- **Evaluation**: The trained model is evaluated on a validation dataset to assess its performance.
- **Logging**: Training and validation loss values are logged and saved in a CSV file for analysis.
- **Model Saving**: The trained downstream model can be saved for future use.
