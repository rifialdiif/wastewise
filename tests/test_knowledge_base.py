"""Phase 5 checks: the knowledge base loads, validates, and serves entries per class."""

import json

import pytest

from app.config import settings
from app.services.recommendation import KnowledgeBase, KnowledgeBaseError

KB_PATH = settings.resolve(settings.kb_path)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(KB_PATH, CLASS_NAMES)


def test_every_class_has_an_entry(kb):
    for name in CLASS_NAMES:
        assert kb.get(name)["material"]


def test_trash_is_marked_residual_and_requires_verification(kb):
    trash = kb.get("trash")
    assert trash["category"] == "residual_or_mixed_waste"
    assert trash["reuse_options"] == []
    assert any("Verify" in note for note in trash["special_notes"])


def test_unknown_class_raises(kb):
    with pytest.raises(KnowledgeBaseError, match="No knowledge base entry"):
        kb.get("battery")


def test_missing_class_entry_fails_validation():
    with pytest.raises(KnowledgeBaseError, match="no entry for classes"):
        KnowledgeBase(KB_PATH, CLASS_NAMES + ["battery"])


def test_malformed_entry_fails_validation(tmp_path):
    bad_kb = tmp_path / "kb.json"
    bad_kb.write_text(json.dumps({"glass": {"category": "x", "material": "Glass"}}), encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="must be a list of strings"):
        KnowledgeBase(bad_kb, ["glass"])


def test_unreadable_file_raises(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="Cannot read knowledge base"):
        KnowledgeBase(tmp_path / "missing.json", CLASS_NAMES)
