import json

from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from app.schemas.prediction import Recommendation

SYSTEM_INSTRUCTION = """\
You write waste-handling recommendations for the WasteWise API.

You receive a JSON object with the predicted waste class, the classifier's
confidence, the prediction status (`accepted` or `uncertain`), and the knowledge
base entry for that class.

Rules:
1. Use ONLY the facts in `knowledge_base_entry`. Do not add disposal, recycling,
   composting, hazardous-waste or treatment rules that are not in it.
   Keep its conditions such as "where available" or "when practical"; never turn
   them into unconditional instructions, and do not assume a recycling bin or
   service exists.
2. Do not change, question or reinterpret `predicted_class`.
3. Rephrase the knowledge base into clear, practical guidance. Keep it concise:
   a summary of 1-2 sentences, at most 5 steps, at most 4 warnings.
4. Put safety notes ("avoid" items) and local-facility caveats in `warnings`.
   When the entry mentions that acceptance or requirements vary by local
   facility, include a warning to check with the local collection or recycling
   facility.
5. If the knowledge base entry is not enough to give a specific recommendation,
   say so in the summary and state that local verification is required.
6. For residual or mixed waste (category `residual_or_mixed_waste`), do not give
   definitive treatment advice. Tell the user to verify the material first.
7. If `prediction_status` is `uncertain`, the class is only the most likely
   guess and may be wrong. Start the summary by saying the material should be
   verified first, and phrase the guidance as applying only if the item really
   is this material. Never present an uncertain class as confirmed.
8. Respond only with JSON matching the response schema.
"""


class LLMServiceError(RuntimeError):
    """Raised when Gemini cannot produce a valid recommendation."""


class GeminiRecommender:
    """Turns a knowledge base entry into a natural-language recommendation with Gemini."""

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 15.0):
        if not api_key:
            raise LLMServiceError("GEMINI_API_KEY is not set")
        self.model = model
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_seconds * 1000),
                # One retry for transient overload; 429 (quota) is not retried.
                retry_options=types.HttpRetryOptions(
                    attempts=2, initial_delay=1.0, http_status_codes=[500, 502, 503, 504]
                ),
            ),
        )

    @staticmethod
    def build_prompt(class_name: str, confidence: float, kb_entry: dict, uncertain: bool = False) -> str:
        """Only the predicted class, its confidence, its status and its KB entry are sent to Gemini."""
        return json.dumps(
            {
                "predicted_class": class_name,
                "confidence": confidence,
                "prediction_status": "uncertain" if uncertain else "accepted",
                "knowledge_base_entry": kb_entry,
            },
            indent=2,
        )

    def recommend(
        self, class_name: str, confidence: float, kb_entry: dict, uncertain: bool = False
    ) -> Recommendation:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=self.build_prompt(class_name, confidence, kb_entry, uncertain),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=Recommendation,
                    temperature=0.2,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except errors.APIError as exc:
            raise LLMServiceError(f"Gemini API error ({exc.code}): {exc.message}") from exc
        except Exception as exc:  # network failures, timeouts
            raise LLMServiceError(f"Gemini request failed: {exc}") from exc

        try:
            recommendation = Recommendation.model_validate_json(response.text or "")
        except ValidationError as exc:
            raise LLMServiceError(f"Gemini returned an invalid recommendation: {exc}") from exc

        if not recommendation.summary.strip() or not recommendation.steps:
            raise LLMServiceError("Gemini returned an empty recommendation")
        return recommendation
