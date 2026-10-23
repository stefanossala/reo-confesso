# REo confesso

*AI opens the safe: AI-assisted model extraction and anonymity under the GDPR*

Extra-credit project for *Advanced Foundations of Law and Ethical Regulations* (2026).

**Thesis:** under GDPR Recital 26, whether an AI model is anonymous depends on the means an attacker is reasonably likely to use. AI reverse-engineering agents make extracting on-device models cheap, so models once considered "safe" inside apps may no longer be anonymous.

All data in this repository is **synthetic**. All apps and models were built by the authors.

## Structure

| Folder | Content |
| --- | --- |
| `data/` | Synthetic datasets (target_train, target_out, shadow_pool) |
| `models/` | Trained Keras and TFLite models |
| `notebooks/` | Exploratory notebooks |
| `src/` | Data generation, training and attack scripts |
| `app/` | Android apps (App A: plain model, App B: obfuscated model) |
| `docs/` | Report, slides, scorecard |

## Setup

```bash
conda create -n ai-safe python=3.11
conda activate ai-safe
pip install -r requirements.txt
```
