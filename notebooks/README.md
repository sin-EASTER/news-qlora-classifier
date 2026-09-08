# News Category Classifier — Web Scraping to QLoRA Fine-Tuning

Scrapes NPR articles across five news categories, builds a labeled text
classification dataset, and fine-tunes `meta-llama/Llama-3.2-1B` for
5-way news category classification using **QLoRA** (4-bit quantization +
LoRA adapters) — training under 1% of the model's parameters.

```
Scrape (NPR)  →  Clean & EDA  →  Stratified Split  →  QLoRA Fine-Tune  →  Evaluate  →  Inference
```

## Why this project

Most "fine-tune an LLM" tutorials either full-fine-tune a tiny model or
skip straight to a pre-cleaned HF dataset. This project does the whole
pipeline end to end on real, messy, self-scraped web data, and uses
**parameter-efficient fine-tuning** (QLoRA) rather than unfreezing raw
layers — the same recipe used to fine-tune much larger LLMs on a single
consumer GPU.

## Key techniques

- **QLoRA fine-tuning** — 4-bit NF4 quantized base model + LoRA adapters
  (rank 16) injected into attention/MLP projections via `peft`.
  `model.print_trainable_parameters()` reports the exact trainable
  fraction at runtime.
- **Stratified train/test split** — keeps category proportions equal
  across splits instead of risking a skewed test set for rarer
  categories.
- **Class-weighted loss** — a custom `Trainer` subclass applies
  `compute_class_weight`-derived weights to cross-entropy, correcting
  for category imbalance measured during EDA.
- **Evaluation beyond accuracy** — macro/weighted F1, a confusion
  matrix, and training curves pulled from the `Trainer`'s own log
  history.
- **Lightweight artifacts** — only the LoRA adapter (a few MB) is saved
  and optionally pushed to the Hugging Face Hub, not the full
  1B-parameter base model.

## Repo structure

```
.
├── notebooks/
│   └── Web_Scraping_for_LLM_fine_Tuning_upgraded.ipynb   # full pipeline
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

## Setup

### 1. Accounts & access (do this first — approval can take a while)

| Requirement | Where |
|---|---|
| Decodo Scraper API key | [decodo.com](https://decodo.com) → dashboard → API credentials |
| Hugging Face account + token (write scope) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Access to the gated `meta-llama/Llama-3.2-1B` model | Request access at [huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) |

### 2. Clone and install

```bash
git clone <your-repo-url>
cd <repo-name>
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 3. Configure credentials

```bash
cp .env.example .env
# fill in SCRAPER_API_AUTH and HF_TOKEN in .env
export $(grep -v '^#' .env | xargs)   # load them into your shell
```

The notebook reads `SCRAPER_API_AUTH` and `HF_TOKEN` from the environment
first and only falls back to an interactive `getpass` prompt if they're
unset — so this works both locally and in Colab (where you'd instead use
Colab's Secrets manager or just paste the values when prompted).

### 4. Run

Open `notebooks/Web_Scraping_for_LLM_fine_Tuning_upgraded.ipynb` in
Jupyter, JupyterLab, VS Code, or Google Colab (**GPU runtime
recommended** — a T4 or better) and run top to bottom.

- **First run:** lower `ARTICLES_PER_CATEGORY` in the crawl section
  (defaults to 1000) to something small, like 50, to confirm the
  pipeline runs end to end before committing to a full crawl.
- **Later runs:** once `news_articles_dataset.csv` exists, you can skip
  the crawl section entirely and start from "Parameters and Reading
  Data."
- **No GPU?** The notebook detects this and trains LoRA without 4-bit
  quantization instead of failing — slower, but it still runs.

## Results

*Fill in after your own run — the notebook logs and plots
(`classification_report`, confusion matrix, training curves) give you
everything needed to complete this table.*

| Metric | Train | Test |
|---|---|---|
| Accuracy | — | — |
| Macro F1 | — | — |
| Weighted F1 | — | — |
| Trainable params (QLoRA) | — |  |

## Tech stack

Python · PyTorch · Hugging Face `transformers`, `datasets`, `evaluate`,
`peft`, `accelerate` · `bitsandbytes` · scikit-learn · BeautifulSoup ·
pandas · matplotlib / seaborn

## Acknowledgments

- News content sourced from [NPR](https://www.npr.org) via the
  [Decodo](https://decodo.com) scraper API
- Base model: [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B)

## License

MIT — see [LICENSE](LICENSE).
