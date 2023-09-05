import torch
from transformers import BertTokenizer
import pandas as pd
import numpy as np


def preprocess_tweets(tweets, labels=None):

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    max_len = 0
    # For every sentence...
    for tweet in tweets:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(tweet, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    input_ids = []
    attention_masks = []

    # For every tweet...
    for tweet in tweets:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            tweet,                      # Sentence to encode.
            add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
            max_length=max_len,         # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True, # Construct attn. masks.
            return_tensors='pt',        # Return PyTorch tensors.
            truncation=True,                # Activate truncation.
            truncation_strategy='longest_first'  # Truncation strategy.
        )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    if labels is not None:
        labels = torch.tensor(labels)
        return input_ids, attention_masks, labels
    else:
        return input_ids, attention_masks
