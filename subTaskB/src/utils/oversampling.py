import pandas as pd
import matplotlib.pyplot as plt


def check_class_imbalance(dataset):
    data = pd.read_csv(dataset)

    class_counts = data["persuasiveness"].value_counts()

    class_frequencies = class_counts / len(data)

    plt.figure(figsize=(8, 6))
    plt.bar(class_counts.index, class_counts.values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.show()

    print("Class Frequencies:")
    for class_label, frequency in class_frequencies.items():
        print(f"{class_label}: {frequency:.2%}")

    imbalance_ratio = class_counts.max() / class_counts.min()
    print("Imbalance Ratio:", imbalance_ratio)

    print("class counts:")
    print(class_counts.values)

def balance_dataset(input_csv, output_csv):
    # Read the input CSV file
    data = pd.read_csv(input_csv)

    # Separate samples with "no" and "yes" persuasiveness
    no_samples = data[data["persuasiveness"] == "no"]
    yes_samples = data[data["persuasiveness"] == "yes"]

    # Determine the number of samples in each class
    num_no_samples = len(no_samples)
    num_yes_samples = len(yes_samples)

    # Copy samples from the minority class until it matches the majority class
    if num_no_samples > num_yes_samples:
        oversampled_yes = yes_samples.sample(n=num_no_samples, replace=True)
        balanced_data = pd.concat([no_samples, oversampled_yes], ignore_index=True)
    else:
        oversampled_no = no_samples.sample(n=num_yes_samples, replace=True)
        balanced_data = pd.concat([oversampled_no, yes_samples], ignore_index=True)

    # Save the balanced dataset to the output CSV file
    balanced_data.to_csv(output_csv, index=False)


if __name__ == "__main__":
    input_csv = "INPUT_CSV.csv"
    output_csv = "OUTPUT_CSV.csv"
    balance_dataset(input_csv, output_csv)
    check_class_imbalance("YOUR_CSV.csv")
