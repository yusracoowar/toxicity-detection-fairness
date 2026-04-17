# toxicity-detection-fairness
Fairness-aware toxicity detection pipeline comparing TF-IDF and DistilBERT with subgroup evaluation - BSc dissertation, Birmingham City University

# AI Detection of Anti-Social Language in Schools and Workplaces

**Author:** Yusra Coowar | **Student ID:** 22225755  
**Course:** BSc Computer Science with Artificial Intelligence  
**Institution:** Birmingham City University  
**Supervisor:** Hadeel Saddany  
**Year:** 2025–2026

---

## Project Overview

This repository contains the full implementation for my final-year dissertation project. The project builds a **fairness-aware toxicity detection pipeline** that compares a classical NLP baseline against a compact transformer model, with subgroup-aware evaluation designed to expose and reduce identity-linked bias in automated text moderation.

The system is designed for school and workplace moderation contexts, where false positives for identity-related comments (e.g. mentions of race, sexuality, or religion) cause direct institutional harm. The project demonstrates that strong overall accuracy can coexist with severe subgroup disparity, and evaluates whether contextual transformer modelling reduces that gap.

---

## Repository Structure

```
├── toxicity-detection-fairness-colab.ipynb       # Main notebook — full pipeline, all experiments
├── frontend/
│   └── app.py                   # Streamlit moderation interface
├── frontend_assets/
│   ├── tfidf_vectorizer.joblib  # Saved TF-IDF vectoriser
│   ├── calibrated_model.joblib  # Saved calibrated classifier
│   ├── best_threshold.joblib    # Saved best threshold
│   └── DistilBERT/              # Saved DistilBERT model and tokeniser
├── moderation_log.csv           # Audit log (generated at runtime)
└── README.md
```

---

## Models and Methods

| Component | Detail |
|---|---|
| Dataset | [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) |
| Training subset | 100,000-row stratified sample |
| Split | 80/10/10 train / validation / test (stratified) |
| Baseline | TF-IDF (unigrams + bigrams) + Logistic Regression (`class_weight="balanced"`) |
| Transformer | `distilbert-base-uncased` fine-tuned for binary sequence classification |
| Calibration | Isotonic regression (post-hoc) |
| Fairness metrics | Subgroup AUC, BPSN AUC, BNSP AUC, identity-slice FPR |
| Explainability | SHAP (TF-IDF baseline), counterfactual identity swaps |

---

## Key Results

| Metric | TF-IDF + LR | DistilBERT |
|---|---|---|
| Test ROC-AUC | 0.9113 | 0.9488 |
| Calibrated test ECE | 0.0015 | 0.0011 |
| Highest subgroup FPR (thr=0.5) | 0.4588 (black) | 0.1051 (homosexual_gay_or_lesbian) |
| Fairness gap (max − min subgroup FPR) | 0.331 | 0.087 |
| Recall at chosen threshold | 0.0374 (thr=0.93) | 0.3866 (thr=0.95) |

The overall fairness gap reduced by **74%** from baseline to DistilBERT. SHAP analysis confirmed that the TF-IDF baseline learned identity-term shortcuts - the token "black" alone had a SHAP value exceeding 1.0, more than twice any other feature. Counterfactual testing showed the largest score gap was between "gay" and "straight" substitutions (0.6279), confirming distributional shortcut learning rather than genuine toxicity detection.

---

## Running the Notebook

The notebook was developed in **Google Colab** with GPU acceleration.

### Requirements

```
torch
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
shap
joblib
streamlit
```

Install with:

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib shap joblib streamlit
```

### Dataset

Download the Jigsaw dataset from Kaggle:  
[https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification)

Place `train.csv` in your working directory before running the notebook.

### Running

Open `toxicity-detection-fairness-colab.ipynb` in Google Colab or Jupyter and run all cells top to bottom. A GPU runtime is required for the DistilBERT fine-tuning section.

---

## Running the Streamlit Interface

The moderation interface loads the saved models and allows free-text input with live prediction, threshold adjustment, reviewer feedback, and CSV audit logging.

```bash
cd frontend
streamlit run app.py
```

The saved model files must be present in `frontend_assets/` before running. These are generated during notebook execution and saved via `joblib`.

---

## Code References and Acknowledgements

The following resources directly informed the implementation:

| Reference | Use in project |
|---|---|
| [Jigsaw Unintended Bias in Toxicity Classification (Kaggle, 2019)](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) | Dataset source |
| [Dixon et al. (2018) — Measuring and Mitigating Unintended Bias](https://dl.acm.org/doi/10.1145/3278721.3278729) | Subgroup evaluation framework, identity dropout approach |
| [Borkan et al. (2019) — Nuanced Metrics for Measuring Unintended Bias](https://arxiv.org/abs/1903.04561) | Subgroup AUC, BPSN AUC, BNSP AUC metric definitions |
| [Sanh et al. (2019) — DistilBERT](https://arxiv.org/abs/1910.01108) | Transformer model architecture |
| [Devlin et al. (2019) — BERT](https://arxiv.org/abs/1810.04805) | Transformer pre-training background |
| [Hardt et al. (2016) — Equality of Opportunity](https://arxiv.org/abs/1610.02413) | Threshold fairness framework |
| [Lundberg and Lee (2017) — SHAP](https://arxiv.org/abs/1705.07874) | SHAP explainability implementation |
| [Garg et al. (2019) — Counterfactual Fairness](https://dl.acm.org/doi/10.1145/3306618.3314248) | Counterfactual identity swap evaluation |
| [Sturman et al. (2024) — Debiasing Text Safety Classifiers](https://aclanthology.org/2024.emnlp-industry.16/) | Debiasing evaluation context |
| [Mosqueira-Rey et al. (2023) — Human-in-the-loop ML](https://link.springer.com/article/10.1007/s10462-022-10246-w) | Streamlit human-in-the-loop design rationale |
| [Oversight Board (2024) — Content Moderation in a New Era](https://www.oversightboard.com/news/content-moderation-in-a-new-era-for-ai-and-automation/) | Audit logging accountability rationale |
| [Streamlit SafeGuard AI (Discuss, 2024)](https://discuss.streamlit.io/t/launched-safeguard-ai-a-bert-powered-app-to-detect-online-toxicity/116891) | Streamlit interface design inspiration |
| [thebugged/toxic-comment-check (GitHub)](https://github.com/thebugged/toxic-comment-check) | Streamlit interface design inspiration |
| [nainiayoub/text-classification-pipeline-app (GitHub)](https://github.com/nainiayoub/text-classification-pipeline-app) | Streamlit interface design inspiration |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | DistilBERT fine-tuning pipeline |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF, Logistic Regression, calibration |
| [SHAP library](https://github.com/shap/shap) | Feature importance visualisation |

---

## Ethical Considerations

- The dataset contains offensive and potentially harmful language. Explicit slurs are not reproduced in the notebook output.
- The dataset is publicly available and anonymised at source. No personal data was collected or processed.
- The system is designed as **moderation support** - human reviewers make final decisions. Automated enforcement is explicitly out of scope.
- Subgroup evaluation is central to the project because identity-linked false positives cause direct harm in safeguarding and HR contexts.

---

## Limitations

- Both models were trained on a 100,000-row stratified subset due to Google Colab compute limits.
- Subgroup results reflect only the six most prevalent identity groups in the dataset.
- No threshold simultaneously satisfied the fairness target (worst-group FPR ≤ 0.15) and recall target (≥ 0.80) for either model.
- The Streamlit interface was not evaluated with real institutional users.

---

## Licence

This repository is submitted as part of an undergraduate dissertation at Birmingham City University. The code is made available for academic review purposes. The Jigsaw dataset is subject to its own [Kaggle competition rules](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/rules).

