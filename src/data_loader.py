import json
import os

import nltk
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from .vocabulary import Vocabulary

def get_loader(
    transform,
    mode="train",
    batch_size=1,
    vocab_threshold=None,
    vocab_file="./vocab.pkl",
    start_word="<start>",
    end_word="<end>",
    unk_word="<unk>",
    vocab_from_file=True,
    num_workers=0,
    cocoapi_loc=None,
):
    """
    Create a DataLoader for the MS COCO image-captioning dataset.

    Args:
        transform: Image preprocessing/augmentation pipeline.
        mode: Dataset mode, either "train" or "test".
        batch_size: Number of samples per batch.
            Test mode requires batch_size=1.
        vocab_threshold: Minimum word frequency used to build the vocabulary.
        vocab_file: Path to the serialized vocabulary file.
        start_word: Start-of-sequence token.
        end_word: End-of-sequence token.
        unk_word: Unknown-word token.
        vocab_from_file: If True, load an existing vocabulary.
            If False, build a vocabulary from the training captions.
        num_workers: Number of worker processes used for data loading.
        cocoapi_loc: Root directory of the COCO dataset.

    Returns:
        torch.utils.data.DataLoader:
            Configured data loader for training or inference.
    """

    # Validate dataset location.
    if cocoapi_loc is None:
        raise ValueError(
             "cocoapi_loc must point to the COCO dataset root directory."
        )

    # Validate mode.
    if mode not in ("train", "test"):
        raise ValueError("mode must be either 'train' or 'test'.")

    # A new vocabulary can only be created from training captions.
    if not vocab_from_file and mode != "train":
        raise ValueError(
            "Vocabulary generation is only supported in training mode."
        )

    # Select dataset paths based on mode.
    if mode == "train":
        if vocab_from_file and not os.path.exists(vocab_file):
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocab_file}. "
                "Set vocab_from_file=False to create a new vocabulary."
            )

        img_folder = os.path.join(
            cocoapi_loc,
            "train2014"
        )

        annotations_file = os.path.join(
            cocoapi_loc,
            "annotations",
            "captions_train2014.json"
        )

    else:
        if batch_size != 1:
            raise ValueError(
                "Test mode requires batch_size=1."
            )

        if not vocab_from_file:
            raise ValueError(
                "Test mode requires vocab_from_file=True."
            )

        if not os.path.exists(vocab_file):
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocab_file}. "
                "Generate the vocabulary from the training data first."
            )

        img_folder = os.path.join(
            cocoapi_loc,
            "test2014"
        )

        annotations_file = os.path.join(
            cocoapi_loc,
            "annotations",
            "image_info_test2014.json"
        )

    if not os.path.isdir(img_folder):
        raise FileNotFoundError(
            f"COCO image directory not found: {img_folder}"
        )

    if not os.path.isfile(annotations_file):
        raise FileNotFoundError(
            f"COCO annotations file not found: {annotations_file}"
        )
    
    # Build the COCO dataset.
    dataset = CoCoDataset(
        transform=transform,
        mode=mode,
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_file=vocab_file,
        start_word=start_word,
        end_word=end_word,
        unk_word=unk_word,
        annotations_file=annotations_file,
        vocab_from_file=vocab_from_file,
        img_folder=img_folder,
    )

    # Training batches contain captions with the same sequence length.
    if mode == "train":
        indices = dataset.get_train_indices()

        sampler = data.sampler.SubsetRandomSampler(
            indices=indices
        )

        batch_sampler = data.sampler.BatchSampler(
            sampler=sampler,
            batch_size=dataset.batch_size,
            drop_last=False,
        )

        data_loader = data.DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
        )

    # Test mode loads one image at a time.
    else:
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=dataset.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return data_loader

class CoCoDataset(data.Dataset):
    """
    MS COCO dataset wrapper for image captioning.

    In training mode, each sample returns:
        (processed_image, caption_tensor)

    In test mode, each sample returns:
        (original_image, processed_image)
    """

    def __init__(
        self,
        transform,
        mode,
        batch_size,
        vocab_threshold,
        vocab_file,
        start_word,
        end_word,
        unk_word,
        annotations_file,
        vocab_from_file,
        img_folder,
    ):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.img_folder = img_folder

        self.vocab = Vocabulary(
            vocab_threshold,
            vocab_file,
            start_word,
            end_word,
            unk_word,
            annotations_file,
            vocab_from_file,
        )

        if self.mode == "train":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())

            print("Obtaining caption lengths...")

            all_tokens = [
                nltk.tokenize.word_tokenize(
                    str(
                        self.coco.anns[self.ids[index]]["caption"]
                    ).lower()
                )
                for index in tqdm(np.arange(len(self.ids)))
            ]

            self.caption_lengths = [
                len(tokens)
                for tokens in all_tokens
            ]

        else:
            with open(annotations_file, "r") as file:
                test_info = json.load(file)

            self.paths = [
                item["file_name"]
                for item in test_info["images"]
            ]

    def __getitem__(self, index):
        """
        Return one dataset sample.

        Training mode:
            Returns a transformed image and its tokenized caption.

        Test mode:
            Returns the original image and transformed image tensor.
        """

        if self.mode == "train":
            ann_id = self.ids[index]
            annotation = self.coco.anns[ann_id]

            caption_text = annotation["caption"]
            img_id = annotation["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            # Load and preprocess image.
            image = Image.open(
                os.path.join(self.img_folder, path)
            ).convert("RGB")

            image = self.transform(image)

            # Tokenize caption.
            tokens = nltk.tokenize.word_tokenize(
                str(caption_text).lower()
            )

            # Convert caption tokens to vocabulary IDs.
            caption_ids = [
                self.vocab(self.vocab.start_word)
            ]

            caption_ids.extend(
                self.vocab(token)
                for token in tokens
            )

            caption_ids.append(
                self.vocab(self.vocab.end_word)
            )

            caption = torch.tensor(
                caption_ids,
                dtype=torch.long,
            )

            return image, caption

        # Test mode
        path = self.paths[index]

        pil_image = Image.open(
            os.path.join(self.img_folder, path)
        ).convert("RGB")

        original_image = np.array(pil_image)
        image = self.transform(pil_image)

        return original_image, image

    def get_train_indices(self):
        """
        Sample indices whose captions have the same sequence length.

        This allows captions in a training batch to be stacked
        directly into a tensor without padding.
        """

        selected_length = np.random.choice(
            self.caption_lengths
        )

        matching_indices = np.where(
            np.array(self.caption_lengths) == selected_length
        )[0]

        indices = np.random.choice(
            matching_indices,
            size=self.batch_size,
        )

        return indices.tolist()

    def __len__(self):
        if self.mode == "train":
            return len(self.ids)

        return len(self.paths)