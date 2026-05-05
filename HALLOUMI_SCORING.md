# HallOumi Scoring: How Claim Verification Works

Reference codebase: `halloumi-demo` (https://github.com/oumi-ai/halloumi-demo)
Training/eval codebase: `oumi` (https://github.com/oumi-ai/oumi/tree/main/configs/projects/halloumi)

## Pipeline Overview

**Preprocessing → Model Inference → Probability Extraction → Platt Scaling → Score Output**

## Two Model Paths

### 1. Generative Model (`isEmbeddingModel: false`)

- Claims and context are split into sentences and annotated with special markers (`<|s1|>`, `<|r1|>`, etc.) — see `halloumi/preprocessing.ts`
- Sent to an OpenAI-compatible `/chat/completions` API with `logprobs: true`, `top_logprobs: 3`, `temperature: 0.0`
- From the response, extracts the **log-probabilities of tokens `"supported"` and `"unsupported"`** — see `halloumi/postprocessing.ts:209-221`
- Logprobs are passed through **softmax** to get probabilities summing to 1.0 — see `postprocessing.ts:179-207`
- The model also outputs structured markup with subclaims, citations, and rationale

### 2. Classifier/Embedding Model (`isEmbeddingModel: true`)

- Each claim sentence is paired with context and sent to an `/embeddings` endpoint
- The returned embedding vector is passed through **softmax** directly — see `postprocessing.ts:223-225`
- Index 0 = P(supported), Index 1 = P(unsupported)
- More computationally efficient, but no citations or rationale

## Platt Scaling (Calibration)

Both paths apply Platt scaling to calibrate raw probabilities (`halloumi/api.ts:13-17`):

```
clamp probability to [1e-6, 1-1e-6]
log_odds = log(p / (1 - p))
calibrated = sigmoid(-(a * log_odds + b))
```

Parameters `a` and `b` are model-specific, defined in `app/data.json`:
- Generative: `a = -0.5764`, `b = 0.1665`
- Classifier: `a = -0.9469`, `b = -0.0738`

`plattScaling` is optional in the `Model` type — if omitted, raw probabilities are used.

## Final Score

- `score` = calibrated P(supported), a float from 0.0 to 1.0
- Threshold: `>= 0.5` = supported, `< 0.5` = unsupported
- Color gradient: 8-bucket system from red (0.0) to green (1.0) — see `app/colors.tsx`

## Using a Different Model (e.g. gpt-oss-120b)

### Requirements for the generative path

The model must:
1. Expose an **OpenAI-compatible API** returning `choices[0].logprobs.content` and `choices[0].message.content`
2. Return **token-level logprobs** (the API sends `logprobs: true`)
3. Produce **HallOumi's structured output format** (`<|r1|>`, `<|supported|>`, `<|cite|>`, `<|explain|>`, etc.)
4. This means the model likely needs to be **fine-tuned** on the HallOumi training datasets

### Adding a model entry

Add to `app/data.json`:
```json
{
    "displayName": "GPT-OSS 120B",
    "name": "gpt-oss-120b",
    "apiUrl": "https://your-api-endpoint/chat/completions",
    "isEmbeddingModel": false
}
```

### Fitting Platt scaling for a new model

The Platt scaling parameters are **not computed anywhere in the oumi or halloumi-demo codebases** — they were fitted offline. To calibrate a new model:

1. Run the model on a labeled validation set (e.g. [oumi-groundedness-benchmark](https://huggingface.co/datasets/oumi-ai/oumi-groundedness-benchmark) test split)
2. Collect raw P(unsupported) from logprobs/softmax
3. Fit logistic regression on log-odds vs ground truth:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# raw_probs: model's P(unsupported) on validation set
# labels: ground truth (0=supported, 1=unsupported)
raw_probs = np.clip(raw_probs, 1e-6, 1 - 1e-6)
log_odds = np.log(raw_probs / (1 - raw_probs)).reshape(-1, 1)
lr = LogisticRegression()
lr.fit(log_odds, labels)

platt_a = lr.coef_[0][0]
platt_b = lr.intercept_[0]
```

4. Add to the model entry in `data.json`:
```json
"plattScaling": { "a": <platt_a>, "b": <platt_b> }
```

## Training a New HallOumi Model

Using oumi (`configs/projects/halloumi/8b_train.yaml`):

```bash
pip install oumi
oumi train -c oumi://configs/projects/halloumi/8b_train.yaml
```

Training datasets (all on HuggingFace):
- `oumi-ai/oumi-anli-subset` (CC BY-NC 4.0 — can be removed to avoid license restriction)
- `oumi-ai/oumi-c2d-d2c-subset`
- `oumi-ai/oumi-synthetic-claims`
- `oumi-ai/oumi-synthetic-document-claims`

Base model: `meta-llama/Llama-3.1-8B-Instruct`

## Key Files

| Component | File |
|-----------|------|
| API + Platt scaling | `halloumi-demo/halloumi/api.ts` |
| Preprocessing | `halloumi-demo/halloumi/preprocessing.ts` |
| Postprocessing + softmax | `halloumi-demo/halloumi/postprocessing.ts` |
| Model config | `halloumi-demo/app/data.json` |
| Types | `halloumi-demo/app/types.tsx` |
| Score display | `halloumi-demo/app/analysisBox.tsx` |
| Color thresholds | `halloumi-demo/app/colors.tsx` |
| Training config | `oumi/configs/projects/halloumi/8b_train.yaml` |
| Eval notebook | `oumi/configs/projects/halloumi/halloumi_eval_notebook.ipynb` |
| Inference notebook | `oumi/configs/projects/halloumi/halloumi_inference_notebook.ipynb` |
| Classifier notebook | `oumi/configs/projects/halloumi/halloumi_classifier_inference_notebook.ipynb` |
