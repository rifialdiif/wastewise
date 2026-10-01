"""Phase 1 checks: project artifacts are present and consistent, and the app boots."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "mobilenetv2_waste_classifier.keras"
KB_PATH = ROOT / "knowledge" / "waste_knowledge_base.json"
CLASS_MAPPING_PATH = ROOT / "config" / "class_mapping.json"

EXPECTED_CLASSES = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
KB_FIELDS = {
    "category",
    "material",
    "general_treatment",
    "preparation",
    "reuse_options",
    "avoid",
    "special_notes",
}


def test_artifacts_exist():
    for path in (MODEL_PATH, KB_PATH, CLASS_MAPPING_PATH):
        assert path.is_file(), f"Missing artifact: {path}"


def test_class_mapping_matches_knowledge_base():
    mapping = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))

    assert sorted(mapping) == [str(i) for i in range(6)]
    assert set(mapping.values()) == EXPECTED_CLASSES
    assert set(kb) == EXPECTED_CLASSES


def test_knowledge_base_entries_have_required_fields():
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    for name, entry in kb.items():
        missing = KB_FIELDS - entry.keys()
        assert not missing, f"KB entry '{name}' missing fields: {missing}"


def test_env_is_gitignored():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
