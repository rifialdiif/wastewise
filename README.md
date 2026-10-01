# WasteWise

**AI-based waste classification and treatment recommendation API.**

Upload a photo of a waste item. WasteWise classifies its material with a
MobileNetV2 model and, **only when the model is confident**, returns practical
handling guidance. A curated knowledge base supplies the facts, and Gemini
writes them up as a short recommendation.

```
Image ─► MobileNetV2 ─► class + confidence ─► confidence gate (≥ 0.90?)
                                                 │                  │
                                              uncertain          accepted
                                                 │                  │
                                    top-3 candidates,      knowledge base entry
                                    no recommendation               │
                                                             Gemini (rephrase only)
                                                                    │
                                                         structured recommendation
```

| Component | Responsibility |
|---|---|
| MobileNetV2 | Classification into 6 material classes |
| Confidence gate | Uncertainty handling: no advice below the threshold |
| Knowledge base | **The only source of treatment facts** (factual guardrail) |
| Gemini | Natural-language recommendation, constrained to the knowledge base |
| FastAPI | API layer, validation, error handling |

---

## Model performance

MobileNetV2 transfer learning, 224×224 RGB input, trained in Google Colab on the
[Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
dataset (2,527 images; cardboard, glass, metal, paper, plastic, trash).

Held-out test set:

| Metric | Value |
|---|---|
| Accuracy (all predictions) | **86.81%** |
| Macro F1 | **83.90%** |
| Coverage at confidence ≥ 0.90 | **56.99%** |
| Accuracy of accepted predictions (≥ 0.90) | **98.61%** |

> **98.61% is not the model's overall accuracy.** It is the accuracy *among
> predictions that pass the 0.90 gate*, which is about 57% of test images. The
> rest are returned as `uncertain`. The gate trades coverage for reliability.

### Real-world check: phone photos

The dataset shows single items on plain white backgrounds. To see how the model
handles real conditions, four phone photos were tested. They had cluttered
desks, several objects per frame and dim lighting.

| Photo | Model's top prediction | Correct? | Status |
|---|---|---|---|
| Clear plastic blister pack | glass (0.66) | ❌ plastic was 2nd (0.28) | uncertain |
| Chocolate foil wrapper | plastic (0.86) | ✅ | uncertain |
| Paper receipt on a desk | cardboard (0.65) | ❌ paper was 3rd (0.06) | uncertain |
| Snack bags | trash (0.55) | ✅ | uncertain |

**Takeaways**

- **The model does not generalize well to cluttered real-world photos.** This is
  a domain gap between the training data and field conditions.
- **The confidence gate did its job.** Both wrong predictions were blocked, so
  no misleading treatment advice was generated.
- **Lowering the threshold would not help.** The wrong predictions (0.65 and
  0.66) scored *higher* than a correct one (0.55). A lower threshold would
  accept wrong answers with confident-sounding advice. Uncertain responses
  include the top-3 `candidates` instead, as hints for the user to verify.

---

## API

Interactive docs: `http://127.0.0.1:8000/docs`

### `POST /predict`

`multipart/form-data` with a `file` field (JPEG, PNG, WebP or BMP, ≤ 10 MB).

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@bottle.jpg"
```

**Accepted** (confidence ≥ threshold):

```json
{
  "prediction": { "class": "plastic", "confidence": 0.9919, "status": "accepted" },
  "recommendation": {
    "summary": "Plastic items should be reused if clean and suitable, or separated for recycling where appropriate collection facilities exist.",
    "steps": [
      "Empty all remaining contents from the item.",
      "Remove non-plastic components when practical.",
      "Ensure the plastic is kept reasonably clean and dry."
    ],
    "warnings": [
      "Do not burn plastic waste in open areas.",
      "Do not assume every type of plastic is accepted by every recycling facility.",
      "Check with your local collection or recycling facility, as acceptance depends on the specific plastic type."
    ]
  }
}
```

**Uncertain** (confidence < threshold; Gemini is not called):

```json
{
  "prediction": { "class": "glass", "confidence": 0.6641, "status": "uncertain" },
  "recommendation": null,
  "message": "Material classification is uncertain. Please verify the waste type before treatment.",
  "candidates": [
    { "class": "glass", "confidence": 0.6641 },
    { "class": "plastic", "confidence": 0.2769 },
    { "class": "paper", "confidence": 0.0429 }
  ]
}
```

**Errors**

| Status | When |
|---|---|
| `200` + `message`, `recommendation: null` | Accepted, but Gemini failed or no API key is configured. The classification is still returned. |
| `400` | Empty, corrupted or unsupported file, or image over 40 MP |
| `413` | Upload larger than `MAX_UPLOAD_MB` |
| `422` | No `file` field |
| `503` | Classification model failed to load |
| `500` | Unexpected error (details are logged, never returned) |

### `GET /health`

```json
{ "status": "ok", "classifier_loaded": true, "recommendations_enabled": true }
```

---

## Design decisions

**The knowledge base is the factual guardrail.** Gemini receives only three
things: the predicted class, the confidence and that class's knowledge base
entry. Its system instruction forbids adding treatment rules, changing the
class, or dropping conditions such as "where available". It must also give no
definitive advice for the heterogeneous `trash` class. Output is constrained by
a JSON schema (structured output) and validated with Pydantic.

**Uncertain means no advice.** Below the threshold, the API never calls Gemini
and never presents the class as reliable.

**Graceful degradation.** A Gemini outage, quota error, timeout or malformed
output never breaks the endpoint. The classification is always returned.
Transient 5xx errors get one retry; quota errors (429) do not.

**Preprocessing matches training.** The model contains its own `Rescaling`
layer, so inference feeds raw 0–255 pixels resized with `tf.image.resize`, the
same as Keras' `image_dataset_from_directory`. This was verified by running all
2,527 dataset images through the API's preprocessing. The confidence pattern
matched the Colab evaluation.

**Confidence is rounded down** to 4 decimals, so a reported value never
overstates the model's certainty. For example, 0.89996 is reported as 0.8999
(`uncertain`), not 0.9.

**Defensive input handling.** Only JPEG, PNG, WebP and BMP files are accepted;
phone MPO photos open as JPEG. Image dimensions are checked from the file
header before decoding, which blocks decompression bombs. Upload size is
checked from `Content-Length` and enforced with a bounded read.

---

## Getting started

Requires **Python 3.12**. TensorFlow 2.21 has no stable Windows wheel for 3.14.

```bash
git clone <repo-url> wastewise && cd wastewise
python -m venv .venv

# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env

# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Add your Gemini key to `.env`. You can get a free key from
[Google AI Studio](https://aistudio.google.com/apikey). Without a key, the API
still classifies; it only skips recommendations.

```bash
# Windows: .venv\Scripts\python.exe -m uvicorn app.main:app --reload
.venv/bin/python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs.

### Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | – | Gemini API key. Never commit it. |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Chosen for availability and latency on the free tier |
| `CONFIDENCE_THRESHOLD` | `0.90` | Gate for accepting predictions (0–1) |
| `MAX_UPLOAD_MB` | `10` | Maximum upload size |
| `CORS_ORIGINS` | empty | Comma-separated website origins allowed to call the API from a browser |
| `MODEL_PATH`, `KB_PATH`, `CLASS_MAPPING_PATH` | see `.env.example` | Artifact locations |

Invalid values, such as a threshold outside 0–1, stop the server at startup
with a clear error.

### Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest                              # 82 tests; Gemini is faked
RUN_LIVE_GEMINI=1 python -m pytest            # also calls the real Gemini API
```

The test suite covers:

- artifact consistency between the class mapping and the knowledge base;
- model loading and validation;
- preprocessing, including colour modes, invalid files and decompression bombs;
- correct classification of one real image per class;
- the confidence gate boundaries;
- knowledge base validation;
- Gemini failure modes: quota, server error, timeout, malformed and empty output;
- the full `/predict` flow, including proof that Gemini is not called for
  uncertain predictions;
- upload limits and clean 500 responses;
- CORS for allowed and unknown origins.

---

## Deployment

The API runs for free on a **Hugging Face Docker Space** (2 vCPU, 16 GB RAM, no
credit card). The free tier sleeps after about 48 hours without traffic, and the
first request after that takes about a minute while it wakes up.

```bash
python -m pip install -r requirements-dev.txt
hf auth login                                  # token with "write" scope
python scripts/deploy_hf.py <hf-username>/wastewise --set-secret --cors-origins "https://your-site.com"
```

`scripts/deploy_hf.py` creates the Space and uploads only what the
[`Dockerfile`](Dockerfile) needs: `app/`, model, knowledge base, class mapping
and runtime requirements. It never uploads `.env` or tests. `--set-secret`
copies `GEMINI_API_KEY` from your local `.env` into the Space's encrypted
secrets. Run the script again after any code change to redeploy.

### Calling the API from a website

Add your site's origin to `CORS_ORIGINS`, as a Space variable or with
`--cors-origins`. Then:

```js
const form = new FormData();
form.append("file", fileInput.files[0]);

const res = await fetch("https://<hf-username>-wastewise.hf.space/predict", {
  method: "POST",
  body: form,
});
const data = await res.json();

if (!res.ok) {
  alert(data.detail);                          // 400 / 413 / 503 ...
} else if (data.prediction.status === "uncertain") {
  console.log(data.message, data.candidates);  // no recommendation
} else {
  console.log(data.prediction, data.recommendation ?? data.message);
}
```

---

## Project structure

```
wastewise/
├── app/
│   ├── main.py                 # FastAPI app, startup loading, middleware, error handler
│   ├── config.py               # Settings from environment / .env
│   ├── api/routes.py           # POST /predict
│   ├── services/
│   │   ├── classifier.py       # Preprocessing, MobileNetV2 inference, confidence gate
│   │   ├── recommendation.py   # Knowledge base loading and validation
│   │   └── llm_service.py      # Gemini recommender (prompt, structured output, failures)
│   └── schemas/prediction.py   # Pydantic request/response models
├── models/mobilenetv2_waste_classifier.keras
├── knowledge/waste_knowledge_base.json
├── config/class_mapping.json
├── tests/                      # pytest suite + one fixture image per class
├── scripts/deploy_hf.py        # Deploy to a Hugging Face Docker Space
├── deploy/huggingface/README.md  # Space card (Hugging Face config front matter)
├── Dockerfile
├── .env.example
├── requirements.txt            # Runtime dependencies
└── requirements-dev.txt        # + tests and deployment tooling
```

---

## Limitations

- **Domain gap.** The model was trained on clean, single-object photos. Accuracy
  on cluttered, poorly lit real-world photos is much lower; see the phone-photo
  check above. Most such photos will be `uncertain`.
- **Six coarse classes.** There is no class for e-waste, batteries, organics or
  hazardous waste. Such items will be forced into one of the six classes and
  should be caught by the gate, but this is not guaranteed.
- **`trash` is heterogeneous.** It is a residual class, so its recommendations
  always ask the user to verify the material first.
- **LLM output is constrained, not proven.** Prompt rules and schema validation
  keep Gemini close to the knowledge base, and outputs were reviewed manually
  for all six classes. However, the content is not automatically fact-checked.
- **General guidance only.** Recycling rules differ between regions and
  facilities. The knowledge base is intentionally generic and always defers to
  local rules.
- **Upload limits.** Chunked uploads without `Content-Length` are received
  before the size check. In production, also enforce a body-size limit at the
  reverse proxy, for example Nginx `client_max_body_size`.
- **CPU inference.** Inference runs on CPU, both locally and on the free Space.
  It takes about 0.3 s per image.
- **Public endpoint.** There is no authentication or rate limiting, so heavy
  traffic can use up the Gemini free-tier quota. Classification keeps working
  when that happens; only recommendations stop.

## Future work

- Fine-tune on real-world photos with cluttered backgrounds, varied lighting and
  multiple objects, to reduce the domain gap.
- Add object detection or cropping, so the classifier sees a single item.
- Extend the classes and knowledge base, for example e-waste, batteries and
  organics, with region-specific entries.
- Automatically check that recommendation steps can be traced back to knowledge
  base entries.
- Add rate limiting or API keys for the public endpoint.
