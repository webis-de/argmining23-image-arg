import torch
import clip
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd

from CustomDataset import Imageargdataset

from utils import create_heatmap, save_val_labels


def train_and_evaluate(
    model: clip,
    train_dataset: Imageargdataset,
    valid_dataset: Imageargdataset,
    preprocess: nn.Module,
    classification_model: nn.Module,
    num_epochs: int,
    device: str,
    sigmoid,
    criterion,
    optimizer,
) -> None:
    """
    Train and evaluate the model.

    Args:
        model (clip): Pretrained CLIP model.
        train_dataset (Imageargdataset): Training dataset.
        valid_dataset (Imageargdataset): Validation dataset.
        preprocess (nn.Module): Preprocessing module for images.
        classification_model (nn.Module): Classification model.
        num_epochs (int): Number of epochs.
        device (str): Device to run the computations on.

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
                chunk = text[i: i + max_context_length]
                tokens.extend(clip.tokenize(chunk))
            text_tokens = torch.stack(tokens).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text_tokens)
                text_features_combined = torch.sum(text_features, dim=0, keepdim=True)
                text_features_combined = text_features_combined.to(image_features.dtype)
                cos_sim = cosine_similarity(
                    image_features, text_features_combined
                ).reshape((1, 1))

            logits = classification_model(cos_sim.to(classification_model.weight.dtype))

            probabilities = sigmoid(logits)

            loss = criterion(
                probabilities.to(device),
                torch.tensor(persuasiveness, dtype=torch.float)
                .reshape(logits.shape)
                .to(device),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_correct += probabilities.to(torch.int).item() == persuasiveness
            train_loss += loss.item()
            train_predicted_labels.append(probabilities)
            train_true_labels.append(persuasiveness)

        train_accuracy = 100 * train_correct / len(train_dataset)
        train_loss /= len(train_dataset)

        val_loss, val_correct, val_predicted_labels, val_true_labels = evaluate(classification_model, device, epoch,
                                                                                model, num_epochs, preprocess,
                                                                                train_accuracy, train_loss,
                                                                                valid_dataset, sigmoid, criterion)

    save_val_labels(val_predicted_labels, val_true_labels)
    create_heatmap(val_predicted_labels)


def evaluate(
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
):
    val_loss = 0.0
    val_correct = 0
    val_predicted_labels = []
    val_true_labels = []
    val_cos_sim_values = []
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
            cos_sim = cosine_similarity(image_features, text_features_combined).reshape((1, 1))
            val_cos_sim_values.append(cos_sim.item())

            logits = classification_model(
                cos_sim.to(classification_model.weight.dtype)
            )

            logits = classification_model(cos_sim.to(classification_model.weight.dtype))

            probabilities = sigmoid(logits)

            loss = criterion(
                probabilities.to(device),
                torch.tensor(persuasiveness, dtype=torch.float)
                .reshape(logits.shape)
                .to(device),
            )

            predicted_labels = probabilities.to(torch.int).item()
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
    if epoch == 9:
        print("Cosine Similarity\tPredicted Label\tTrue Label")
        for cos_sim, pred_label, true_label in zip(val_cos_sim_values, val_predicted_labels, val_true_labels):
            print(f"{cos_sim:.4f}\t\t\t\t\t{pred_label}\t\t\t\t\t{true_label}")

        sorted_indices = sorted(range(len(val_cos_sim_values)), key=val_cos_sim_values.__getitem__)
        sorted_cos_sim_values = [val_cos_sim_values[i] for i in sorted_indices]
        sorted_predicted_labels = [val_predicted_labels[i] for i in sorted_indices]
        sorted_true_labels = [val_true_labels[i] for i in sorted_indices]

        data = {
            'Cosine Similarity': sorted_cos_sim_values,
            'Predicted Label': sorted_predicted_labels,
            'True Label': sorted_true_labels
        }
        df = pd.DataFrame(data)
        df.to_csv('Cos_sim_val_pred_labels')

    return val_loss, val_correct, val_predicted_labels, val_true_labels


def main(backbone_model, downstream_model, device):
    model, preprocess = backbone_model
    train_csv = r""
    valid_csv = r""
    image_folder = r""
    train_dataset = Imageargdataset(train_csv, image_folder)
    valid_dataset = Imageargdataset(valid_csv, image_folder)
    classification_model = downstream_model
    sigmoid = nn.Sigmoid()
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(classification_model.parameters(), lr=0.1)
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

    downstream_model = nn.Linear(1, num_classes).to(device)
    main(backbone_model, downstream_model, device)
