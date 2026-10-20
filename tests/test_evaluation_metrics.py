import pytest

from evaluation.metrics import (
    tokenize_caption,
    calculate_sentence_bleu,
    calculate_corpus_bleu,
)


def test_tokenize_caption():
    caption = "A Dog Is Running"

    tokens = tokenize_caption(caption)

    assert tokens == ["a", "dog", "is", "running"]


def test_perfect_prediction():
    prediction = "a dog is running"

    references = ["a dog is running"]

    scores = calculate_sentence_bleu(
        prediction,
        references,
    )

    for score in scores.values():
        assert score == pytest.approx(1.0)


def test_different_prediction():
    prediction = "cats sleep inside"

    references = ["dogs run outside"]

    scores = calculate_sentence_bleu(
        prediction,
        references,
    )

    for score in scores.values():
        assert score == pytest.approx(0.0)


def test_multiple_references():
    prediction = "a dog is running"

    references = [
        "a cat is sleeping",
        "a dog is running",
        "a bird is flying",
    ]

    scores = calculate_sentence_bleu(
        prediction,
        references,
    )

    assert scores["BLEU-4"] == pytest.approx(1.0)


def test_empty_prediction():
    scores = calculate_sentence_bleu(
        "",
        ["a dog is running"],
    )

    assert all(score == 0.0 for score in scores.values())


def test_empty_references():
    with pytest.raises(ValueError):
        calculate_sentence_bleu(
            "a dog is running",
            [],
        )


def test_corpus_bleu():
    predictions = [
        "a dog is running",
        "a cat is sleeping",
    ]

    references = [
        ["a dog is running"],
        ["a cat is sleeping"],
    ]

    scores = calculate_corpus_bleu(
        predictions,
        references,
    )

    for score in scores.values():
        assert score == pytest.approx(1.0)


def test_corpus_length_mismatch():
    with pytest.raises(ValueError):
        calculate_corpus_bleu(
            ["a dog is running"],
            [],
        )


def test_reproducibility():
    prediction = "a dog is running"

    references = ["a dog is running"]

    result1 = calculate_sentence_bleu(
        prediction,
        references,
    )

    result2 = calculate_sentence_bleu(
        prediction,
        references,
    )

    assert result1 == result2


def test_partial_overlap_prediction():
    """
    Different sentences sharing one unigram
    should produce a nonzero BLEU-1 score.
    """
    prediction = "a cat sleeps indoors"
    references = ["a dog runs outside"]

    scores = calculate_sentence_bleu(
        prediction,
        references,
    )

    assert scores["BLEU-1"] == pytest.approx(0.25)

    for score in scores.values():
        assert 0.0 <= score <= 1.0


def test_no_overlap_prediction():
    """
    Sentences with no shared unigrams
    should produce zero cumulative BLEU scores.
    """
    prediction = "cats sleep inside"
    references = ["dogs run outside"]

    scores = calculate_sentence_bleu(
        prediction,
        references,
    )

    for score in scores.values():
        assert score == pytest.approx(0.0)