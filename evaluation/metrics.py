from typing import Sequence

from nltk.translate.bleu_score import(
    corpus_bleu,
    sentence_bleu,
    SmoothingFunction
)

BLEU_WEIGHTS = {
    "BLEU-1": (1.0, 0.0, 0.0, 0.0),
    "BLEU-2": (0.5, 0.5, 0.0, 0.0),
    "BLEU-3": (1/3, 1/3, 1/3, 0.0),
    "BLEU-4": (0.25, 0.25, 0.25, 0.25)
}

def tokenize_caption(caption: str) -> list[str]:
    """
    Convert a caption into lowercase tokens.
    """
    return caption.lower().strip().split()

def calculate_sentence_bleu(
        prediction: str,
        references: Sequence[str],
) -> dict[str, float]:
    """
    Calculate sentence-level BLEU-1 through BLEU-4

    Args:
        prediction:
            Model-generated caption.
        
        references:
            Human-annotated reference captions.

    Returns:
        Dictionary containing BLEU-1 through BLEU-4
    """
    if not references:
        raise ValueError("At least one reference is required.")
    
    prdiction_tokens = tokenize_caption(prediction)

    reference_tokens = [
        tokenize_caption(reference)
        for reference in references
    ]

    if not prdiction_tokens:
        return {metric: 0.0 for metric in BLEU_WEIGHTS}
    
    smoothing = SmoothingFunction().method1

    scores = {}

    for metric, weights in BLEU_WEIGHTS.items():
        scores[metric] = float(
            sentence_bleu(
                reference_tokens,
                prdiction_tokens,
                weights=weights,
                smoothing_function=smoothing,
            )
        )
    
    return scores

def calculate_corpus_bleu(
        predictions: Sequence[str],
        references: Sequence[Sequence[str]],
) -> dict[str,float]:
    """
    Calculate corpus-level BLEU-1 through BLEU-4.

    Each prediction corresponds to a list of reference captions.
    """
    if len(predictions) != len(references):
        raise ValueError(
            "Predictions and references must have equal lengths."
        )
    
    if not predictions:
        raise ValueError("Evaluation corpus cannot be empty.")
    
    if any(not sample_references for sample_references in references):
        raise ValueError(
            "Every prediction must have at least one reference."
        )
    
    prediction_tokens = [
        tokenize_caption(prediction)
        for prediction in predictions
    ]

    reference_tokens = [
        [
        tokenize_caption(reference)
        for reference in sample_references
        ]
        for sample_references in references
    ]

    scores = {}

    for metric, weights in BLEU_WEIGHTS.items():
        scores[metric] = float(
            corpus_bleu(
                reference_tokens,
                prediction_tokens,
                weights=weights
            )
        )
    
    return scores
