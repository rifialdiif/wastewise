from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class PredictionStatus(str, Enum):
    ACCEPTED = "accepted"
    UNCERTAIN = "uncertain"


class Prediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    predicted_class: str = Field(alias="class", description="Predicted waste class.", examples=["plastic"])
    confidence: float = Field(ge=0.0, le=1.0, description="Softmax probability of the predicted class.")
    status: PredictionStatus = Field(
        description="'accepted' when confidence >= threshold, otherwise 'uncertain'."
    )


class Candidate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    candidate_class: str = Field(alias="class", description="A possible waste class.")
    confidence: float = Field(ge=0.0, le=1.0, description="Softmax probability of this class.")


class Recommendation(BaseModel):
    """Treatment recommendation. Also used as Gemini's structured-output schema."""

    summary: str = Field(description="One or two sentences on how to handle this material.")
    steps: list[str] = Field(description="Short, practical handling steps taken from the knowledge base.")
    warnings: list[str] = Field(description="Safety notes and local-verification reminders.")


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error message.")


class PredictResponse(BaseModel):
    prediction: Prediction
    recommendation: Recommendation | None = None
    message: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Present when no recommendation is given, explaining why.",
    )
    candidates: list[Candidate] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Only for uncertain predictions: the most likely classes, highest first. "
            "These are hints for the user to verify, not reliable classifications."
        ),
    )
