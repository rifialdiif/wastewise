import json
from pathlib import Path

LIST_FIELDS = ("general_treatment", "preparation", "reuse_options", "avoid", "special_notes")
TEXT_FIELDS = ("category", "material")


class KnowledgeBaseError(RuntimeError):
    """Raised when the knowledge base is missing, malformed, or lacks a class."""


class KnowledgeBase:
    """Waste-treatment facts per class. This is the only source of treatment rules."""

    def __init__(self, kb_path: Path, class_names: list[str]):
        self._entries = self._load(kb_path)
        self._validate(class_names)

    @staticmethod
    def _load(kb_path: Path) -> dict:
        try:
            data = json.loads(kb_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KnowledgeBaseError(f"Cannot read knowledge base at {kb_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise KnowledgeBaseError("Knowledge base must be a JSON object keyed by class name")
        return data

    def _validate(self, class_names: list[str]) -> None:
        missing = set(class_names) - self._entries.keys()
        if missing:
            raise KnowledgeBaseError(f"Knowledge base has no entry for classes: {sorted(missing)}")

        for name in class_names:
            entry = self._entries[name]
            for field in TEXT_FIELDS:
                if not isinstance(entry.get(field), str) or not entry[field]:
                    raise KnowledgeBaseError(f"KB entry '{name}': '{field}' must be a non-empty string")
            for field in LIST_FIELDS:
                value = entry.get(field)
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise KnowledgeBaseError(f"KB entry '{name}': '{field}' must be a list of strings")

    def get(self, class_name: str) -> dict:
        """Return the KB entry for a predicted class."""
        try:
            return self._entries[class_name]
        except KeyError:
            raise KnowledgeBaseError(f"No knowledge base entry for class '{class_name}'") from None
