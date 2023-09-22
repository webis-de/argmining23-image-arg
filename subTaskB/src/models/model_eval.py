import torch
import numpy as np
from torchvision.transforms import transforms
from sklearn.metrics import confusion_matrix

from src.dataset.CustomDataset import Imageargdataset

import seaborn as sns
import matplotlib.pyplot as plt

from transformers import CLIPTokenizer


def evaluate_model(model, dataloader, device, task, topic=None):
    CM = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images = data["image"]
            tweet_texts = data["tweet_text"]
            stance = data["stance"]
            persuasiveness = data["persuasiveness"]

            images = images.to(device)
            stance = stance.to(device)
            persuasiveness = persuasiveness.to(device)

            if task == "task_a":
                outputs = model(images, tweet_texts, topic)

            elif task == "task_b":
                outputs = model(images, tweet_texts)

            # model should return class probabilities
            preds = torch.argmax(outputs.data, 1)
            if task == "task_a":
                CM += confusion_matrix(stance.cpu(), preds.cpu(), labels=[0, 1])
            elif task == "task_b":
                CM += confusion_matrix(persuasiveness.cpu(), preds.cpu(), labels=[0, 1])
            else:
                raise ValueError(
                    "Wrong task input given, please provide either 'task_a' or 'task_b'"
                )

        tn = CM[0][0]
        tp = CM[1][1]
        fp = CM[0][1]
        fn = CM[1][0]
        acc = np.sum(np.diag(CM) / np.sum(CM))
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)

        print("\nTestset Accuracy(mean): %f %%" % (100 * acc))
        print()
        print("Confusion Matirx : ")
        print(CM)
        print("- Sensitivity : ", (tp / (tp + fn)) * 100)
        print("- Specificity : ", (tn / (tn + fp)) * 100)
        print("- Precision: ", (tp / (tp + fp)) * 100)
        print("- NPV: ", (tn / (tn + fn)) * 100)
        print(
            "- F1 : ", ((2 * sensitivity * precision) / (sensitivity + precision)) * 100
        )
        print()

    return acc, CM


def print_confus_heatmap(confus_matrix):
    normalized_cm = (
        confus_matrix.astype("float") / confus_matrix.sum(axis=1)[:, np.newaxis]
    )
    _, ax = plt.subplots(figsize=(8, 6))
    _ = sns.heatmap(normalized_cm, annot=True, cmap="Blues", fmt=".2f", cbar=False)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


if __name__ == "__main__":
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    dataset = Imageargdataset(
        csv_file=r"YOUR_TRAIN_CSV.csv",
        image_folder=r"YOUR_IMAGE_FOLDER",
        tokenizer=tokenizer,
        transforms=image_transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )

    model1 = CLIPClassifier("ViT-B/32")

    accuracy, confus_matrix = evaluate_model(model1, dataloader, device, "task_b")
    print_confus_heatmap(confus_matrix)

    print("plotting losses and accuracies")
    print(accuracy, confus_matrix)
