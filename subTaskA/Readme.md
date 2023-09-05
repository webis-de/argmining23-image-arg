*Brief understanding of the training process*
- Inside the training loop, we extract the input IDs and attention masks from the batch, and the model is called with input_ids and mask to obtain the output logits.
- The logits are compared with the corresponding training labels (train_label) using the cross-entropy loss (criterion).
- The loss is backpropagated through the model, and the optimizer is used to update the model parameters.
- During training, the accuracy is computed by comparing the predicted class (argmax of the logits) with the training labels.
- The same process is repeated for the validation data to compute validation loss and accuracy.
