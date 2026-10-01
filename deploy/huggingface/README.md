---
title: WasteWise
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Waste classification and treatment recommendation API
---

# WasteWise API

Upload a photo of a waste item to get its material class (cardboard, glass,
metal, paper, plastic, trash) from a MobileNetV2 model. When the model is
confident (≥ 0.90), the API also returns handling guidance grounded in a
curated knowledge base and written up by Gemini.

- **Interactive docs:** append `/docs` to this Space's URL
- **Main endpoint:** `POST /predict` (multipart field `file`)
- **Source code and full documentation:** https://github.com/rifialdiif/wastewise

Recommendations are general guidance; always check your local waste-handling rules.
