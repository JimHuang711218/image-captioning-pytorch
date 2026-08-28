import os
import pickle
from collections import Counter

import nltk
from pycocotools.coco import COCO


class Vocabulary:
    """
    Vocabulary manager for converting between words and token IDs.
    """

    def __init__(
        self,
        vocab_threshold,
        vocab_file="./vocab.pkl",
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        annotations_file="../cocoapi/annotations/captions_train2014.json",
        vocab_from_file=False,
    ):
        """
        Initialize or load the vocabulary.

        Args:
            vocab_threshold: Minimum word frequency required for inclusion.
            vocab_file: Path to the serialized vocabulary file.
            start_word: Start-of-sequence token.
            end_word: End-of-sequence token.
            unk_word: Unknown-word token.
            annotations_file: Path to the COCO training annotations.
            vocab_from_file: If True, load an existing vocabulary.
                If False, build a new vocabulary from training captions.
        """

        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file

        self.get_vocab()

    def get_vocab(self):
        """
        Load an existing vocabulary or build a new one.
        """

        if self.vocab_from_file:
            if not os.path.exists(self.vocab_file):
                raise FileNotFoundError(
                    f"Vocabulary file not found: {self.vocab_file}"
                )

            with open(self.vocab_file, "rb") as file:
                vocab = pickle.load(file)

            self.word2idx = vocab.word2idx
            self.idx2word = vocab.idx2word
            self.idx = len(self.word2idx)

            print(
                f"Vocabulary successfully loaded from "
                f"{self.vocab_file}."
            )

        else:
            self.build_vocab()

            with open(self.vocab_file, "wb") as file:
                pickle.dump(self, file)

            print(
                f"Vocabulary successfully saved to "
                f"{self.vocab_file}."
            )

    def build_vocab(self):
        """
        Build the vocabulary from COCO training captions.
        """

        self.init_vocab()

        # Add special tokens first.
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)

        # Add words that satisfy the frequency threshold.
        self.add_captions()

    def init_vocab(self):
        """
        Initialize mappings between words and token IDs.
        """

        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """
        Add a word to the vocabulary if it is not already present.
        """

        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """
        Tokenize COCO captions and add sufficiently frequent words.
        """

        coco = COCO(self.annotations_file)
        counter = Counter()

        annotation_ids = list(coco.anns.keys())

        for index, ann_id in enumerate(annotation_ids):
            caption = coco.anns[ann_id]["caption"]

            tokens = nltk.tokenize.word_tokenize(
                str(caption).lower()
            )

            counter.update(tokens)

            if index % 100000 == 0:
                print(
                    f"[{index}/{len(annotation_ids)}] "
                    "Tokenizing captions..."
                )

        words = [
            word
            for word, count in counter.items()
            if count >= self.vocab_threshold
        ]

        for word in words:
            self.add_word(word)

    def __call__(self, word):
        """
        Convert a word into its token ID.
        """

        return self.word2idx.get(
            word,
            self.word2idx[self.unk_word],
        )

    def __len__(self):
        """
        Return the vocabulary size.
        """

        return len(self.word2idx)