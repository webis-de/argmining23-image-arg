import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Imageargdataset_Test(Dataset):
    def __init__(self, csv_file: str, image_file: str, tokenizer=None, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.image_file = image_file
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tweet_id = self.data["tweet_id"][index]
        image_path = rf"{self.image_file}/{tweet_id}.jpg"
        image = Image.open(image_path)
        tweet_text = self.data["tweet_text"][index]

        if image.mode != "RGB":
            image = image.convert("RGB")

        return {
            "image": image,
            "tweet_text": tweet_text,
            "tweet_id": tweet_id,
        }
