import torch
import clip
import torch.nn as nn
from typing import Tuple, List
import pandas as pd

import sys
sys.path.append('/Users/torky/Documents/project-multimodal-sequence-representations-for-feed-data/main_TaskB/src')


from dataset.TestDataset import Imageargdataset_Test


def predict(
        model: nn.Module,
        dataset: Imageargdataset_Test,
        preprocess: nn.Module,
        classification_model: nn.Module,
        device: str,
) -> List[int]:
    """
    Generate predictions for the given dataset using the pretrained model.

    Args:
        model (nn.Module): Pretrained CLIP model.
        dataset (Imageargdataset): Dataset for prediction.
        preprocess (nn.Module): Preprocessing module for images.
        classification_model (nn.Module): Classification model.
        device (str): Device to run the computations on.

    Returns:
        List[int]: List of predicted labels for the dataset.
    """
    model.eval()
    predictions_ids = []
    predicted_labels = []

    with torch.no_grad():
        for i in range(0, len(dataset)):
            image = dataset[i]["image"]
            text = dataset[i]["tweet_text"]
            tweet_id = dataset[i]["tweet_id"]
            image = preprocess(image).unsqueeze(0).to(device)

            max_context_length = model.context_length
            tokens = []
            for i in range(0, len(text), max_context_length):
                chunk = text[i: i + max_context_length]
                tokens.extend(clip.tokenize(chunk))
            text_tokens = torch.stack(tokens).to(device)

            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)
            text_features_combined = torch.sum(text_features, dim=0, keepdim=True)
            text_features_combined = text_features_combined.to(image_features.dtype)
            joint_embedding = torch.cat([image_features, text_features_combined], dim=-1)

            logits = classification_model(joint_embedding.to(classification_model.weight.dtype))
            _, predicted = torch.max(logits, 1)
            predictions_ids.append((tweet_id, predicted))

    return predictions_ids


def save_predictions_to_csv(dataset: Imageargdataset_Test, predicted_labels: List[Tuple]) -> None:
    """
    Save predictions to a CSV file according to the specified format.

    Args:
        dataset (Imageargdataset): Validation dataset.
        predicted_labels (List[int]): List of predicted labels for the validation set.

    Returns:
        None
    """
    # Create a DataFrame with tweet_id and persuasiveness columns
    df = pd.DataFrame(predicted_labels, columns=['tweet_id', 'persuasiveness'])

    # Convert predicted labels to "yes" and "no" strings
    df["persuasiveness"] = df["persuasiveness"].apply(lambda label: "yes" if label == 1 else "no")

    # Save DataFrame to CSV file
    csv_filename = "test-team.baseline.TaskB.1.csv"  # Update with your desired filename
    df.to_csv(csv_filename, index=False)

    print(f"Predictions saved to {csv_filename}")


def load_pretrained_weights(model: nn.Module, weights_path: str) -> None:
    """
    Load pretrained weights from a .pth file.

    Args:
        model (nn.Module): Model to load the weights into.
        weights_path (str): Path to the .pth file containing pretrained weights.

    Returns:
        None
    """
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def main(
        backbone_model: Tuple[nn.Module, nn.Module],
        downstream_model: nn.Module,
        weights_path: str,
        device: str,
) -> None:
    """
    Main function to load pretrained weights and run the prediction pipeline.

    Args:
        backbone_model (Tuple[nn.Module, nn.Module]): Pretrained CLIP model and preprocess module.
        downstream_model (nn.Module): Classification model.
        weights_path (str): Path to the .pth file containing pretrained weights.
        device (str): Device to run the computations on.

    Returns:
        None
    """
    model, preprocess = backbone_model
    classification_model = load_pretrained_weights(downstream_model, weights_path)  # Load pretrained weights
    test_csv = r"/Users/torky/Desktop/University Work/Feeds/ImageArg/data_test/gun_control_test_cleaned.csv"
    image_folder = r"/Users/torky/Desktop/University Work/Feeds/ImageArg/data_test/images/gun_control"
    valid_dataset = Imageargdataset_Test(test_csv, image_folder)

    # Generate predictions
    predicted_labels = predict(model, valid_dataset, preprocess, classification_model, device)

    # Save predictions to CSV file
    save_predictions_to_csv(valid_dataset, predicted_labels)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = 2
    backbone_model = clip.load("ViT-B/32", device=device)

    downstream_model = nn.Linear(1024, num_classes).to(device)

    weights_path = "/Users/torky/Desktop/feeds models adamw/cleaned_Chuncks_0.001_gunControl/downstream_model.pth"  # Update this with the actual path to the .pth file

    main(backbone_model, downstream_model, weights_path, device)
