# Video Captioning with a Sequence-to-Sequence Model

An attention-based encoder-decoder that produces a natural-language caption for a
short video clip. Built from scratch in PyTorch and trained on the MSVD dataset
using pre-extracted CNN features (80 frames by 4096 dimensions per clip).

## Approach

- Encoder: a recurrent network over the frame feature sequence.
- Attention: at each decoding step the decoder hidden state is scored against all
  encoder outputs to form a context vector.
- Decoder: a recurrent network that emits one word at a time, with scheduled
  sampling so the fraction of ground-truth versus model-predicted inputs shifts
  as training proceeds.
- Vocabulary: words seen more than four times in the training captions, plus
  `<PAD>`, `<BOS>`, `<EOS>`, and `<UNK>` tokens. The mapping is saved to
  `i2w.pickle`.
- Inference: greedy decoding, with generated captions written out for BLEU
  scoring. `testing_sample.txt` holds an example run.

## Files

| File | Purpose |
| --- | --- |
| `model_train_seq2seq.py` | Data preprocessing, model definition, training loop |
| `model_test.py` | Loads a trained model and writes captions for a feature directory |
| `s2s.sh` | Convenience wrapper: runs inference on the test feature set |
| `i2w.pickle` | Index-to-word vocabulary produced during training |
| `testing_sample.txt` | Example generated captions |
| `SavedModel/` | Location for the trained checkpoint |

## Setup

1. Download `MLDS_hw2_1_data.tar.gz` from
   https://drive.google.com/file/d/1RevHMfXZ1zYjUm4fPU1CfFKAjyMJjdgJ/view and
   extract it into this folder so that `MLDS_hw2_1_data/` sits next to the
   scripts.
2. Train:
   ```bash
   python model_train_seq2seq.py
   ```
   This writes `i2w.pickle` and the trained model.
3. Generate captions for the test set:
   ```bash
   bash s2s.sh
   ```
