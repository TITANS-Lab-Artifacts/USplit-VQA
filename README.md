# USplit-VQA

**U-Shaped Split Learning for Visual Question Answering with Contribution-Aware Weighted Aggregation**

Md Khalid Syfullah and Alvi Ataur Khalil
Transformative Innovation for Trustworthy AI and Network Security (TITANS) Lab
Computer Science, Southern Illinois University Carbondale, USA

This repository holds the full experimental code for USplit-VQA, a U-shaped split learning framework for privacy-sensitive Visual Question Answering, together with CAWA (Contribution-Aware Weighted Aggregation), a gradient-similarity client scoring mechanism that limits the influence of malicious or low-quality client updates.

The paper (IEEE SMC 2026 submission) is the reference for all methodology and reported numbers. This README explains what each file does and how to run it.

# Reference (pre-print)
@article{syfullah2026usplitvqa,
  title   = {USPLIT-VQA: U-Shaped Split Learning for Visual Question Answering with Contribution-Aware Weighted Aggregation},
  author  = {Syfullah, Md Khalid and Khalil, Alvi Ataur},
  journal = {arXiv preprint arXiv:2609.12168},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.12168}
}

---

## 1. What the framework does

In USplit-VQA the model is cut twice, so each client keeps both ends of the network and the server only holds the middle:

| Component | Location | Role |
|---|---|---|
| Client head | Client | Encodes the raw image and question into visual and text tokens |
| Server body | Server | Visual refinement (CBAM), text refinement, bidirectional cross-modal fusion, learnable pooling |
| Client tail | Client | Classification head, cross-entropy loss against private labels, start of backpropagation |
| CAWA module | Server | Scores each client by reputation-weighted gradient cosine similarity, then scales its loss contribution |

Raw images, questions and answer labels never leave the client. The server sees only cut-layer activations and gradients.

CAWA works in four steps each round: reputation is converted to a bounded trust weight through a temperature-scaled softmax, that weight scales the client loss, a reputation-weighted average pairwise cosine similarity is computed across client gradients, and reputation is then updated with streak-amplified rewards and penalties against adaptive thresholds (mean plus or minus lambda times standard deviation) with a quadratic temporal warmup.

Two backbones are evaluated throughout:

* **Custom model**, a lightweight architecture of roughly 8.5M parameters with D = 256, a four-stage convolutional vision encoder producing 49 spatial tokens, a 2-layer text transformer, 3 CBAM blocks, 2 text refinement blocks and 4 bidirectional fusion layers.
* **BiomedCLIP**, the pretrained `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` checkpoint with D = 768, split by keeping the first ViT block and the first PubMedBERT layer on the client and the remaining blocks on the server.

---

## 2. Datasets

Four datasets are used. None of them are stored in this repository. Each `.txt` file at the root records where the data comes from, and every notebook downloads it at runtime through Hugging Face `datasets`.

| File | Dataset | Hugging Face path | Domain |
|---|---|---|---|
| `vqarad.txt` | VQA-RAD | `flaviagiammarino/vqa-rad` | Radiology |
| `slake.txt` | SLAKE (English subset) | `mdwiratathya/SLAKE-vqa-english` | Medical, multi-modality |
| `pathvqa.txt` | PathVQA | `flaviagiammarino/path-vqa` | Pathology |
| `vizwiz.txt` | VizWiz-VQA | `lmms-lab/VizWiz-VQA` | Accessibility |

`dataloader_vizwiz.py` is a standalone helper for VizWiz. It loads the validation split, resolves the ten crowd answers into a single label by majority voting, drops unanswerable and unsuitable samples, resizes images to 224x224, and builds an answer vocabulary capped at 300 classes with a minimum frequency of 3. The same logic is inlined inside the VizWiz notebooks, so the file also serves as a readable reference for that pipeline.

Answer normalization is dataset specific. Every notebook carries its own `normalize_answer_slake`, `normalize_answer_vqarad`, `normalize_answer_pathvqa` or `normalize_answer_vizwiz` function that maps yes/no variants, number words, modality and plane synonyms, anatomical terms, laterality and abnormality synonyms onto a single canonical string before the vocabulary is built. VQA is treated as classification over that vocabulary.

---

## 3. Repository layout

```
USplitVQA-main/
├── dataloader_vizwiz.py              VizWiz loading and majority-vote label helper
├── vqarad.txt, slake.txt,            Dataset source links
│   pathvqa.txt, vizwiz.txt
├── Direct_Model_Apply/               Centralized baseline (full data on one machine)
├── Federated_Learning/               FedAvg baseline (each client holds the full model)
├── Split_Learning/                   USplit-VQA with CAWA, the proposed method
├── Split_Point_Comparison/           Where to place the cut layer (V1 to V4)
├── Poisoning Attack/                 Label flipping and backdoor robustness, CAWA ablation
├── Model Inversion Attack/           Decoder-based reconstruction from cut-layer features
├── Gradient Inversion Attack/        Deep Leakage from Gradients (DLG)
├── Comparison With BiCSL/            Head-to-head against BiCSL on accuracy and four attacks
└── Results/                          Per-round and per-epoch logs as Excel workbooks
```

Every experiment is a single self-contained Jupyter notebook. There is no shared package to import, so each notebook repeats the data loading, the model blocks and the training loop. Any one of them can be opened and run on its own without touching the rest of the repository.

---

## 4. Code files, folder by folder

### 4.1 `Direct_Model_Apply/` (centralized baseline)

Trains the whole model on pooled data, which is the accuracy ceiling the distributed methods are measured against.

| File | Backbone | Dataset |
|---|---|---|
| `New_Custom_Centralized VQA RAD.ipynb` | Custom | VQA-RAD |
| `New_Custom_Centralized SLAKE.ipynb` | Custom | SLAKE |
| `New_Custom_Centralized_PATH_VQA.ipynb` | Custom | PathVQA |
| `New_Custom_Centralized_VizWiz.ipynb` | Custom | VizWiz |
| `New_BioMedClip_Centralized_VQA_RAD.ipynb` | BiomedCLIP | VQA-RAD |
| `New_BioMedClip_Centralized_SLAKE.ipynb` | BiomedCLIP | SLAKE |
| `New_BioMedClip_Centralized_PATH VQA.ipynb` | BiomedCLIP | PathVQA |
| `New_BioMedClip_Centralized_VIZWIZ.ipynb` | BiomedCLIP | VizWiz |

Custom runs use up to 30 epochs, batch size 32, learning rate 3e-4, weight decay 1e-4, early stopping with patience 16 and a learning rate reduction on plateau. BiomedCLIP runs use learning rate 2e-5 because the backbone is pretrained.

### 4.2 `Federated_Learning/` (FedAvg baseline)

Five clients, IID partition, 3 local epochs per round, 20 rounds for the Custom model and 15 for BiomedCLIP, sample-count weighted `fedavg_agg` aggregation of client state dicts. Each client holds a full copy of the model, which is the cost USplit-VQA is designed to avoid.

Files follow the same naming pattern as above: `New_Custom_Federated Learning {VQA RAD, SLAKE, PATH VQA, VizWiz}.ipynb` and `New_BioMedClip_Federated_Learning_{VQA_RAD, SLAKE, PATH VQA, VizWiz}.ipynb`.

### 4.3 `Split_Learning/` (USplit-VQA with CAWA, the proposed method)

Seven notebooks, grouped by filename prefix:

| File | Dataset |
|---|---|
| `New_Custom_Split Learning_VQA RAD` | VQA-RAD |
| `New_Custom_Split Learning_SLAKE` | SLAKE |
| `New_Custom_Split Learning_PATH VQA` | PathVQA |
| `New_Custom_Split_Learning_VizWiz.ipynb` | VizWiz |
| `New_BioMedClip_Split_Learning_VQA_RAD.ipynb` | VQA-RAD |
| `New_BioMedClip_Split_Learning_PATH_VQA.ipynb` | PathVQA |
| `New_BioMedClip_Split_Learning_VizWiz.ipynb` | VizWiz |

Each notebook is organized in the same order, so once you have read one you can navigate all of them:

1. **Setup**: pip installs, seeding at 42, device selection, `OUTPUT_DIR = "/kaggle/working/"`.
2. **Hyperparameters**: `NUM_CLIENTS`, `ROUNDS`, `BS`, `SERVER_LR`, `ENC_LR`, `WD`, `FREEZE_ROUNDS`, and the CAWA constants `CAWA_ALPHA`, `CAWA_BETA`, `CAWA_GAMMA`, `CAWA_LAMBDA`, `CAWA_TEMP`, `CAWA_PHI`. The VizWiz Custom notebook uses a fixed-threshold CAWA variant with `CAWA_REWARD`, `CAWA_PENALTY`, `CAWA_POS` and `CAWA_NEG` instead.
3. **Answer normalization and data loading**, then an IID split into `NUM_CLIENTS` shards, one `DataLoader` per client and one central test loader.
4. **Model blocks**: `TransformerBlock`, `VisionEncoder`, `TextEncoder`, `ChannelAttention`, `SpatialAttention`, `CBAMBlock`, `FusionLayer`.
5. **Partition**: `ClientEncoder` holds the vision and text encoders and supports `freeze()` and `unfreeze()`. `ServerModel` holds CBAM refinement, text refinement, question-aware visual attention, the fusion stack, learnable pooling and the classification head. The BiomedCLIP notebooks build the two sides by deep-copying the pretrained ViT and PubMedBERT stacks and pruning each copy to its share of the blocks.
6. **CAWA**: `grad_cos_sim(grads, idx, cids, current_reputations, temp)` flattens each client gradient to a unit vector and returns the reputation-weighted average of pairwise cosine similarities against all peers.
7. **Training loop**: clients are visited in sequence each round. Client encoders stay frozen for the first `FREEZE_ROUNDS` rounds so the server stabilizes first, then the encoders unfreeze and get their own AdamW optimizer. Each client loss is multiplied by its trust weight before backward, gradients are averaged over the client's batches, and gradient norms are clipped at 1.0. At round end the similarities, adaptive thresholds, streak counters and reputations are updated.
8. **Evaluation and export**: `eval_usplit` runs the full client to server to client forward pass with encoders frozen. The best state dict by test accuracy is kept, and an Excel workbook with a per-round sheet and a summary sheet is written to `OUTPUT_DIR`.

Typical Custom settings here are 5 clients, 20 rounds, batch size 16 or 32, server learning rate 2e-4, encoder learning rate 3e-4, 4 freeze rounds, with alpha 0.1, beta 0.05, gamma 0.8, lambda 0.5, temperature 1.0 and phi 2.0.

### 4.4 `Split_Point_Comparison/`

Four SLAKE runs on the Custom model that move the cut layer progressively toward the client, holding everything else fixed at 5 clients and 30 rounds. This is the study behind Table III and Figure 6 of the paper.

| File | Client keeps | Client share | Transmitted |
|---|---|---|---|
| `V1_Server98_Client2.ipynb` | Vision encoder only, text handled server side | about 2% | Visual tokens |
| `V2_Server57_Client43.ipynb` | Vision encoder plus text embeddings, no text transformer blocks | about 43% | Visual and text tokens |
| `V3_Server37_Client63.ipynb` | Vision encoder, text encoder, CBAM blocks, text refinement | about 63% | Refined tokens |
| `V4_Server20_Client80.ipynb` | Everything in V3 plus question attention and the first 2 fusion layers | about 80% | Fused tokens |

The only difference between the four files is the `ClientEncoder` class and the matching `ServerModel` entry point.

### 4.5 `Poisoning Attack/`

Byzantine robustness on SLAKE. The attack is compound: a malicious client flips 30% of its labels (`LABEL_FLIP_RATE = 0.30`) and stamps an 8x8 white trigger patch on 20% of its images while relabeling them to `BACKDOOR_TARGET = 1`. Two numbers are tracked every round, clean test accuracy and backdoor attack success rate through `compute_asr`, which measures how often triggered test images are pushed to the target class.

| File | What it runs |
|---|---|
| `CAWA Module No Poisoning` | Full CAWA, all clients honest, the reference curve |
| `CAWA Module One Malicious Client` | Full CAWA, client 0 malicious |
| `CAWA Module Two Malicious Client` | Full CAWA, clients 0 and 1 malicious |
| `CAWA Module Three Malicious Client` | Full CAWA, clients 0, 1 and 2 malicious |
| `Attack on Federated Learning.ipynb` | Same attack against FedAvg |
| `Attack on USplit without CAWA.ipynb` | U-shaped split with uniform aggregation, the CAWA ablation |
| `Attack on Federated Learning and USplit without CAWA.ipynb` | Both ablated baselines in one comparative run |

Which clients are malicious is set through `MALICIOUS_CLIENT0`, `MALICIOUS_CLIENT1` and `MALICIOUS_CLIENT2`, where a value of -1 means the slot is unused. All of these runs use 30 rounds, 4 freeze rounds, alpha 0.1, beta 0.05, gamma 0.5, lambda 0.8, temperature 1.0 and phi 2.0.

### 4.6 `Model Inversion Attack/Attack.ipynb`

Trains an `InversionDecoder`, a five-stage transposed convolution decoder that goes from 49 tokens of width 256 up to 224x224x3, and tries to reconstruct client images from intercepted cut-layer features. The victim encoders are pretrained for 5 epochs first so the features carry real signal, then the decoder is trained for 30 epochs under three setups (Centralized, FedAvg, USplit). Reported metrics are reconstruction MSE, PSNR through `compute_psnr` and an SSIM proxy through `compute_ssim_proxy`. The USplit setup models the smashed-data bottleneck by adding feature noise with standard deviation 2.0 and dropping 50% of the tokens before the decoder sees them.

### 4.7 `Gradient Inversion Attack/Attack.py`

Deep Leakage from Gradients against a `SimpleVQAModel`. For each of 20 SLAKE samples the true gradient is computed, a defense adds noise, and `dlg_attack` then optimizes a dummy image for 300 iterations at learning rate 0.1 to match the intercepted gradient. Five setups are compared: Centralized with clean gradients, FedAvg at noise 0.01, USplit at 0.1, CAWA at 0.05, and Adaptive CAWA at 0.05 scaled by the per-tensor gradient standard deviation. Metrics are reconstruction MSE and PSNR over the 20 samples.

### 4.8 `Comparison With BiCSL/Attack.ipynb`

The most self-contained file in the repository and a good starting point for reading. It builds both topologies from shared blocks and gives them identical server capacity:

* `UC` and `US`, the U-shaped client (encoders plus classification head) and server (middle only).
* `BC` and `BS`, the standard split client (encoders only) and server (middle plus classification head, so the server sees labels).

It then runs six evaluations back to back: clean accuracy, Byzantine accuracy under a 30% label flip on client 0, label inference (`atk_label_inf`), model inversion (`atk_model_inv`), membership inference (`atk_membership`) and gradient leakage (`atk_grad_leak`). Results go into a three-sheet workbook covering per-round curves, the head-to-head comparison table and attack details.

### 4.9 `Results/`

Excel workbooks with the raw training logs. Every workbook has a per-round or per-epoch sheet plus a summary sheet.

* Centralized logs: `Epoch`, train and validation and test loss and accuracy, learning rate.
* FedAvg logs: `Round`, average train loss and accuracy, test loss and accuracy, round time, then per-client loss, accuracy and time.
* USplit logs: the same round columns plus, for each client, `Reputation`, `Weight` and `Similarity`, which is what Figure 3 of the paper is drawn from.

The files are `{vqarad, slake, pathvqa}_{centralized, federated, usplit}_results.xlsx` and `vizwiz_{centralized, fedavg, usplit}_biomedclip_results.xlsx`. The filename identifies the dataset and the training paradigm for each log.

---

## 5. How to run

### 5.1 Environment

Everything was developed as Kaggle and Colab notebooks. The Custom model experiments ran on a Tesla T4 and the BiomedCLIP experiments on an NVIDIA RTX PRO 6000. A single GPU with 16 GB is enough for the Custom model. BiomedCLIP is far heavier, mainly because FedAvg needs the full 226.7M parameter model per client.

The notebooks install their own dependencies in the first cell, so no requirements file is needed:

```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "datasets", "openpyxl", "tqdm"])
```

BiomedCLIP notebooks add `open_clip_torch` and `transformers` to that list. PyTorch is assumed to be present, which is true on Kaggle and Colab. For a local setup:

```bash
python -m pip install torch torchvision
python -m pip install datasets openpyxl tqdm open_clip_torch transformers
```

### 5.2 Running an experiment

1. Pick the notebook for the paradigm, backbone and dataset you want, for example `Split_Learning/New_Custom_Split Learning_SLAKE`.
2. Open it in Jupyter, Kaggle or Colab and make sure a GPU is attached.
3. Change `OUTPUT_DIR` from `/kaggle/working/` to a directory that exists on your machine if you are not on Kaggle.
4. Run all cells. The dataset downloads from Hugging Face on the first run. Setting an `HF_TOKEN` avoids the anonymous rate limit on the larger downloads such as PathVQA.
5. The notebook prints per-round progress and writes an Excel workbook to `OUTPUT_DIR` at the end.

### 5.3 Switching datasets inside a notebook

Several notebooks keep more than one dataset call in place with the unused ones commented out:

```python
ds = load_dataset('mdwiratathya/SLAKE-vqa-english')
#ds = load_dataset('flaviagiammarino/vqa-rad')
#ds = load_dataset('flaviagiammarino/path-vqa')
```

The answer normalization function, the answer vocabulary settings and the output filename are all dataset specific, so change those together with the loader line. Starting from the notebook that already targets your dataset is the simpler route.

### 5.4 Knobs worth changing

| Variable | Meaning |
|---|---|
| `NUM_CLIENTS` | Client count. The scalability study sweeps 5, 7, 9, 11, 13, 15 |
| `ROUNDS` | Global rounds, 20 for the main runs and 30 for the security studies |
| `FREEZE_ROUNDS` | Rounds before the client encoders unfreeze |
| `SERVER_LR`, `ENC_LR` | Separate learning rates for the server body and the client encoders |
| `CAWA_ALPHA`, `CAWA_BETA` | Base reward and penalty applied to reputation |
| `CAWA_GAMMA` | Exponential streak multiplier |
| `CAWA_LAMBDA` | Threshold sensitivity in standard deviations from the mean similarity |
| `CAWA_TEMP` | Softmax temperature for turning reputation into a trust weight |
| `CAWA_PHI` | Temporal warmup exponent, 2.0 gives a quadratic ramp |
| `LABEL_FLIP_RATE`, `TRIGGER_SIZE`, `BACKDOOR_TARGET` | Poisoning strength in the attack notebooks |
| `DLG_ITERS`, `DLG_LR`, `N_ATTACK_SAMPLES` | Gradient inversion attack budget |
| `OUTPUT_DIR` | Where the Excel logs are written, `/kaggle/working/` by default |

### 5.5 Reproducing specific paper results

| Paper item | Files to run |
|---|---|
| Table IV, accuracy across datasets and paradigms | All of `Direct_Model_Apply/`, `Federated_Learning/`, `Split_Learning/` |
| Table V, resource and time comparison | `Federated_Learning/` and `Split_Learning/`, then read parameter counts and round times from the printed summary and the Excel logs |
| Table III and Figure 6a, 6c, split point study | `Split_Point_Comparison/V1` to `V4` |
| Figure 6b, 6d, client scalability | Any Custom SLAKE split learning notebook with `NUM_CLIENTS` set to 5, 7, 9, 11, 13, 15 |
| Table VI and Figure 3, poisoning and CAWA ablation | All of `Poisoning Attack/` |
| Table VII, inversion attacks | `Model Inversion Attack/Attack.ipynb` and `Gradient Inversion Attack/Attack.py` |
| Figure 5, BiCSL trade-offs | `Comparison With BiCSL/Attack.ipynb` |

---

## 6. Headline results from the paper

Test accuracy in percent, USplit-VQA against the centralized and FedAvg baselines:

| Dataset | BiomedCLIP Cent. | BiomedCLIP FL | BiomedCLIP USplit | Custom Cent. | Custom FL | Custom USplit |
|---|---|---|---|---|---|---|
| VQA-RAD | 43.68 | 44.57 | 37.92 | 39.25 | 36.36 | **43.90** |
| SLAKE | 81.24 | 83.51 | 74.18 | 78.23 | 65.88 | 67.77 |
| PathVQA | 68.42 | 69.18 | 58.25 | 46.88 | 46.60 | **48.25** |
| VizWiz | 60.89 | 62.62 | 61.01 | 62.25 | 60.02 | **62.75** |

Cost against FedAvg:

| Metric | BiomedCLIP FL | BiomedCLIP USplit | Custom FL | Custom USplit |
|---|---|---|---|---|
| Client parameters | 226.7M | 38.8M | 8.5M | 1.5M |
| Client memory | 2,720 MB | 465 MB | 102 MB | 18 MB |
| Communication per round | 907 MB | 84 MB | 34 MB | 5.6 MB |
| Time per round | 139.6 s | 99.5 s | 68.1 s | 36.3 s |

Client memory falls by 5.7x to 5.8x and communication by 6.1x to 10.8x. With one malicious client, CAWA suppresses that client's aggregation weight by 98.2% while holding 65.2% clean accuracy and 0.6% attack success rate. Protection weakens as the malicious share grows, and with two or more attackers the backdoor succeeds across all frameworks tested.

Limitations noted in the paper: the IID client partition does not reflect clinical heterogeneity, BiomedCLIP loses accuracy under the evaluated fixed split, and CAWA weakens beyond one malicious client out of five. Future work targets non-IID adaptation and improved split selection for large pretrained models.

---

## 7. Citation

```bibtex
@inproceedings{syfullah2026usplitvqa,
  title     = {USplit-VQA: U-Shaped Split Learning for Visual Question Answering
               with Contribution-Aware Weighted Aggregation},
  author    = {Syfullah, Md Khalid and Khalil, Alvi Ataur},
  booktitle = {IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  year      = {2026}
}
```

---

## 8. LLM Usage Statement

Claude was used to prepare and organize this repository, including reviewing the code files and writing this README. All experiments, model designs and results are the authors' own work.
