import pandas as pd

def under_sample_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    persuasiveness_counts = dataset['persuasiveness'].value_counts()
    minority_class = persuasiveness_counts.idxmin()
    minority_indices = dataset[dataset['persuasiveness'] == minority_class].index
    majority_sample = dataset[dataset['persuasiveness'] != minority_class].sample(n=persuasiveness_counts[minority_class], random_state=42)
    under_sampled_dataset = pd.concat([dataset.loc[minority_indices], majority_sample])
    under_sampled_dataset = under_sampled_dataset.sample(frac=1, random_state=42)
    under_sampled_dataset.to_csv('under_sampled_dataset.csv', index=False)

    return under_sampled_dataset
