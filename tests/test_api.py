from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app, get_inference_service


class FakeInferenceService:
    def predict(self, image_path: Path) -> str:
        return "a test caption"


app.dependency_overrides[get_inference_service] = (
    lambda: FakeInferenceService()
)

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={
            "image": (
                "test.txt",
                b"not an image",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Uploaded file must be an image."
    }


def test_predict_returns_caption():
    image_path = (
        PROJECT_ROOT
        / "tests"
        / "assets"
        / "test_image.jpg"
    )

    assert image_path.exists(), (
        f"Test image not found: {image_path}"
    )

    with open(image_path, "rb") as image_file:
        response = client.post(
            "/predict",
            files={
                "image": (
                    "test_image.jpg",
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "caption": "a test caption"
    }