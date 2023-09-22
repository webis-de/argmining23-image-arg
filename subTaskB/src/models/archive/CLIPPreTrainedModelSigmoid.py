import torch
import clip
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List
from CustomDataset import Imageargdataset
from utils import create_heatmap, save_val_labels


def train_and_evaluate(
    model: nn.Module,
    train_dataset: Imageargdataset,
    valid_dataset: Imageargdataset,
    preprocess: nn.Module,
    classification_model: nn.Module,
    num_epochs: int,
    device: str,
    sigmoid: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> None:
    """
    Train and evaluate the model.

    Args:
        model (nn.Module): Pretrained CLIP model.
        train_dataset (Imageargdataset): Training dataset.
        valid_dataset (Imageargdataset): Validation dataset.
        preprocess (nn.Module): Preprocessing module for images.
        classification_model (nn.Module): Classification model.
        num_epochs (int): Number of epochs.
        device (str): Device to run the computations on.
        sigmoid (nn.Module): Sigmoid activation function.
        criterion (nn.Module): Loss criterion.
        optimizer (optim.Optimizer): Optimization algorithm.

    Returns:
        None
    """
    for epoch in range(num_epochs):
        classification_model.train()
        train_loss = 0.0
        train_correct = 0
        train_predicted_labels = []
        train_true_labels = []

        for i in range(len(train_dataset)):
            image = train_dataset[i]["image"]
            text = train_dataset[i]["tweet_text"]
            persuasiveness = train_dataset[i]["persuasiveness"]
            image = preprocess(image).unsqueeze(0).to(device)

            max_context_length = model.context_length
            tokens = []
            for i in range(0, len(text), max_context_length):
                chunk = text[i : i + max_context_length]
                tokens.extend(clip.tokenize(chunk))
            text_tokens = torch.stack(tokens).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text_tokens)
                text_features_combined = torch.sum(text_features, dim=0, keepdim=True)
                text_features_combined = text_features_combined.to(image_features.dtype)
                joint_embedding = torch.cat(
                    [image_features, text_features_combined], dim=-1
                )

            logits = classification_model(
                joint_embedding.to(classification_model.weight.dtype)
            )
            probabilities = sigmoid(logits)

            loss = criterion(
                probabilities.squeeze(),
                torch.tensor(persuasiveness, dtype=torch.float).to(device),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted_labels = (probabilities > 0.4).to(torch.int).item()
            train_correct += predicted_labels == persuasiveness
            train_loss += loss.item()
            train_predicted_labels.append(predicted_labels)
            train_true_labels.append(persuasiveness)

        train_accuracy = 100 * train_correct / len(train_dataset)
        train_loss /= len(train_dataset)

        val_loss, val_correct, val_predicted_labels, val_true_labels = evaluate(
            classification_model,
            device,
            epoch,
            model,
            num_epochs,
            preprocess,
            train_accuracy,
            train_loss,
            valid_dataset,
            sigmoid,
            criterion,
        )

    save_val_labels(val_predicted_labels, val_true_labels)
    create_heatmap(val_predicted_labels)


def evaluate(
    classification_model: nn.Module,
    device: str,
    epoch: int,
    model: nn.Module,
    num_epochs: int,
    preprocess: nn.Module,
    train_accuracy: float,
    train_loss: float,
    valid_dataset: Imageargdataset,
    sigmoid: nn.Module,
    criterion: nn.Module,
) -> Tuple[float, int, List[int], List[int]]:
    """
    Evaluate the model on the validation dataset.

    Args:
        classification_model (nn.Module): Classification model.
        device (str): Device to run the computations on.
        epoch (int): Current epoch.
        model (nn.Module): Pretrained CLIP model.
        num_epochs (int): Total number of epochs.
        preprocess (nn.Module): Preprocessing module for images.
        train_accuracy (float): Training accuracy.
        train_loss (float): Training loss.
        valid_dataset (Imageargdataset): Validation dataset.
        sigmoid (nn.Module): Sigmoid activation function.
        criterion (nn.Module): Loss criterion.

    Returns:
        Tuple[float, int, List[int], List[int]]: Validation loss, validation correct count,
        predicted labels for validation set, true labels for validation set.
    """
    classification_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_predicted_labels = []
    val_true_labels = []
    with torch.no_grad():
        for i in range(0, len(valid_dataset)):
            image = valid_dataset[i]["image"]
            text = valid_dataset[i]["tweet_text"]
            persuasiveness = valid_dataset[i]["persuasiveness"]
            image = preprocess(image).unsqueeze(0).to(device)

            max_context_length = model.context_length
            tokens = []
            for i in range(0, len(text), max_context_length):
                chunk = text[i : i + max_context_length]
                tokens.extend(clip.tokenize(chunk))
            text_tokens = torch.stack(tokens).to(device)

            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)
            text_features_combined = torch.sum(text_features, dim=0, keepdim=True)
            text_features_combined = text_features_combined.to(image_features.dtype)
            joint_embedding = torch.cat(
                [image_features, text_features_combined], dim=-1
            )

            logits = classification_model(
                joint_embedding.to(classification_model.weight.dtype)
            )
            probabilities = sigmoid(logits)

            loss = criterion(
                probabilities.squeeze(),
                torch.tensor(persuasiveness, dtype=torch.float).to(device),
            )

            predicted_labels = (probabilities > 0.4).to(torch.int).item()
            val_correct += predicted_labels == persuasiveness
            val_loss += loss.item()
            val_predicted_labels.append(predicted_labels)
            val_true_labels.append(persuasiveness)
    val_accuracy = 100 * val_correct / len(valid_dataset)
    val_loss /= len(valid_dataset)
    print(
        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.2f}% - "
        f"Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.2f}%"
    )

    return val_loss, val_correct, val_predicted_labels, val_true_labels


def main(
    backbone_model: Tuple[nn.Module, nn.Module],
    downstream_model: nn.Module,
    device: str,
) -> None:
    """
    Main function to run the training and evaluation pipeline.

    Args:
        backbone_model (Tuple[nn.Module, nn.Module]): Pretrained CLIP model and preprocess module.
        downstream_model (nn.Module): Classification model.
        device (str): Device to run the computations on.

    Returns:
        None
    """
    model, preprocess = backbone_model
    train_csv = r"main_TaskB\feeds\ImageArg\data\gun_control_train.csv"
    valid_csv = r"main_TaskB\feeds\ImageArg\data\gun_control_dev.csv"
    image_folder = r"main_TaskB\feeds\ImageArg\data\images\gun_control"
    train_dataset = Imageargdataset(train_csv, image_folder)
    valid_dataset = Imageargdataset(valid_csv, image_folder)
    classification_model = downstream_model
    sigmoid = nn.Sigmoid()
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(classification_model.parameters(), lr=0.001)
    num_epochs = 10

    train_and_evaluate(
        model,
        train_dataset,
        valid_dataset,
        preprocess,
        classification_model,
        num_epochs,
        device,
        sigmoid,
        criterion,
        optimizer,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = 1
    backbone_model = clip.load("ViT-B/32", device=device)

    downstream_model = nn.Linear(1024, num_classes).to(device)
    main(backbone_model, downstream_model, device)
