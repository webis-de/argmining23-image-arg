import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Imageargdataset(Dataset):
    def __init__(self, csv_file: str, image_folder: str, tokenizer=None, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.image_file = image_folder
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tweet_id = self.data["tweet_id"][index]
        image_path = rf"{self.image_file}/{tweet_id}.jpg"
        image = Image.open(image_path)
        tweet_text = self.data["tweet_text"][index]
        stance = self.data["stance"][index]
        persuasiveness = self.data["persuasiveness"][index]

        if stance == "oppose":
            stance = 0
        elif stance == "support":
            stance = 1

        if persuasiveness == "no":
            persuasiveness = 0
        elif persuasiveness == "yes":
            persuasiveness = 1

        if image.mode != "RGB":
            image = image.convert("RGB")

        return {
            "image": image,
            "tweet_text": tweet_text,
            "stance": stance,
            "persuasiveness": persuasiveness,
            "image_path": image_path,
        }
