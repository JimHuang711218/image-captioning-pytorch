from dataclasses import dataclass
from pathlib import Path

from pycocotools.coco import COCO

@dataclass(frozen=True)
class EvaluationSample:
    """
    One image-captioning evaluation sample.

    Attributes:
        image_path: Path to the evaluation image.
        reference: Human-written reference captions for the image
    """
    image_path : Path
    references : list[str]
    image_id : int | None = None


class EvaluationDataset:
    """
    Dataset for evaluating image-captioning models on COCO.

    Each sample contains:
        - the path to one image
        - all human reference captions associated with that image
    """

    def __init__(
            self,
            image_dir: Path,
            annotations_file: Path
    ):
        self.image_dir = Path(image_dir)
        self.annotations_file = Path(annotations_file)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Evaluation image directory not found : {self.image_dir}"
            )
        
        if not self.annotations_file.is_file():
            raise FileNotFoundError(
                f"Evaluation annotations not found: {self.annotations_file}"
            )
        
        self.coco = COCO(str(self.annotations_file))
        self.image_ids = sorted(self.coco.getImgIds())

    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, index: int) -> EvaluationSample:
        image_id = self.image_ids[index]

        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.image_dir / image_info["file_name"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Evaluation image not found: {image_path}"
            )
        
        annotation_ids = self.coco.getAnnIds(
            imgIds=[image_id]
        )

        annotations = self.coco.loadAnns(annotation_ids)

        references = [
            annotation["caption"]
            for annotation in annotations
        ]

        if not references:
            raise ValueError(
                f"No reference captions found for Image ID {image_id}"
            )
        
        return EvaluationSample(
            image_path=image_path,
            references=references,
            image_id=image_id,
        )
        