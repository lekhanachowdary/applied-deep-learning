# Applied Deep Learning

Three self-contained projects covering sequence modeling, transformer question
answering, and generative modeling. Each folder has its own README with setup and
usage details.

## Projects

### [video-captioning-seq2seq](video-captioning-seq2seq/)
An attention-based encoder-decoder that generates natural-language captions for
short video clips, trained on the MSVD dataset from pre-extracted CNN features.
Implemented from scratch in PyTorch, including the attention module, scheduled
sampling during decoding, and a greedy inference path. Output captions are scored
with BLEU.

### [extractive-qa-squad2](extractive-qa-squad2/)
Fine-tuning transformer models for extractive question answering on SQuAD 2.0,
which mixes answerable and unanswerable questions. Three variants are compared: a
DistilBERT baseline, a RoBERTa-base model, and a DistilBERT run that adds a
sliding-window document stride so long contexts are not truncated. Built with
Hugging Face Transformers and evaluated with F1.

### [gan-dcgan-vs-wgan](gan-dcgan-vs-wgan/)
A side-by-side comparison of DCGAN and WGAN image generation on CIFAR-10. Both
models are trained for 50 epochs, with per-epoch sample grids, generator and
discriminator loss curves, and Frechet Inception Distance tracked over training.
The notebook collects the saved metrics and renders the comparison plots.

## Stack

PyTorch, Hugging Face Transformers, NumPy, Matplotlib. Datasets and large trained
checkpoints are downloaded separately; see each project README for links.
