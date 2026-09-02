# Extractive Question Answering on SQuAD 2.0

Fine-tuning transformer models to answer questions by predicting the start and
end token positions of the answer span inside a context paragraph. SQuAD 2.0
includes unanswerable questions, so the models also have to learn when no span
should be returned.

Built with PyTorch and Hugging Face Transformers. Models are evaluated with the
token-level F1 score.

## Notebooks

| Notebook | Model | Notes |
| --- | --- | --- |
| `distilbert_baseline.ipynb` | `distilbert-base-uncased` | Baseline. Contexts truncated to the max sequence length. |
| `roberta_base.ipynb` | `roberta-base` | Larger pretrained encoder, same training setup as the baseline. |
| `distilbert_doc_stride.ipynb` | `distilbert-base-uncased` | Adds a sliding-window document stride of 128 tokens so long contexts are split into overlapping chunks instead of being cut off. |

Each notebook covers data loading, tokenization with answer-span alignment, the
training loop, and F1 evaluation.

## Setup

Place the dataset files in this folder:

```
spoken_train-v1.1.json
spoken_test-v1.1.json
```

Then open any notebook and run it top to bottom. A GPU runtime is recommended.
