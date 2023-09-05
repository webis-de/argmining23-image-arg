from transformers import BertForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

def load_model(device): # taking device argument to specify where the model will be loaded (e.g., "cpu" or "cuda:0" for GPU).
    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    # Move the model to the specified device
    model = model.to(device)

    # Print the device being used
    print("Model is using device:", device)

    return model


# BertForSequenceClassification extends the base BERT model by adding a classification layer on top. 
# This additional layer is a linear transformation that takes the final hidden state of the BERT model and maps it to the specified number of output labels.
# By default, it uses a softmax activation function to produce probabilities for each class.
# BertForSequenceClassification.from_pretrained(), you get a complete model that includes the 12 transformer layers from the base BERT model
# And an additional classification layer on top, which allows you to perform sequence classification tasks.