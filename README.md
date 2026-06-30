# Credit Card Fraud Detection

A binary classification project to detect fraudulent credit card transactions under **extreme class imbalance** (only ~0.29% of transactions are fraud). The focus is not on building a flashy model, but on **evaluating models honestly** and **choosing the right tool for the data**.

## 1. Problem Definition

- **What:** A binary classification task to detect fraudulent transactions in credit card data. The core challenge is extreme class imbalance — only about 0.29% of all transactions are fraud.
- **Why it's hard:** The imbalance breaks both *evaluation* and *training*. On the evaluation side, since 99.71% of transactions are normal, a model that blindly predicts "all normal" still scores 99.71% accuracy — so accuracy lies. On the training side, the model sees too few fraud examples and takes the lazy shortcut of labeling everything normal instead of learning the actual fraud patterns.

## 2. Data

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** ~54,000 transactions, 153 fraud cases (~0.29%)
- **Features:** V1–V28 (PCA-anonymized), Time, Amount / Target: `Class`

## 3. Why These Evaluation Metrics

### Why accuracy is useless here

This dataset is extremely imbalanced, with 99.71% normal transactions. Since accuracy = (TP + TN) / total, the huge pile of true negatives (TN) dominates the numerator. As a result, even a lazy model that predicts "normal" for every transaction reaches 99.71% accuracy — its score stays high even while missing every single fraud (exploding FN). In other words, accuracy buries the minority-class (fraud) detection ability we actually care about under the overwhelming majority class. Precision and recall, by contrast, use no TN in their calculation, so they speak honestly about fraud. I therefore discarded accuracy and adopted **precision, recall, and PR-AUC** as the core metrics.

### Why I prioritized recall over precision

In fraud detection, a false negative (missing a fraud) leads to stolen money, refunds, and loss of trust — a cost I judged to be higher than that of a false positive (blocking a legitimate customer, causing inconvenience and call-center load). I chose to accept some false alarms in exchange for not missing real fraud.

## 4. Methods & Comparison

All models are evaluated on the **same untouched test set**, with the same metrics at the default threshold (0.5):

| Method | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Baseline (Logistic Regression) | 0.8289 | 0.6429 | 0.7241 | 0.7432 |
| Random Forest | 0.9412 | 0.8163 | 0.8743 | 0.8734 |
| class_weight='balanced' | 0.0610 | 0.9184 | 0.1144 | 0.7159 |
| SMOTE | 0.0580 | 0.9184 | 0.1092 | 0.7249 |
| Neural Net (MLP) | 0.1238 | 0.8878 | 0.2172 | 0.7708 |

## 5. Results & Interpretation

### Best method for my goal, and why

Looking purely at my priority (recall), class_weight and SMOTE score highest with a recall of 0.92. But this comes at the cost of precision collapsing to ~0.06 — meaning 94% of all fraud alerts are false alarms. In practice, that false-alarm rate would overwhelm the call center and drive customers away, so it's hard to accept. Random Forest, by contrast, has a slightly lower recall (0.82) but maintains a precision of 0.94, giving it the best balance between the two. I therefore selected **Random Forest** as the best model under the criterion of "prioritize recall while keeping precision at a practically usable level." If recall absolutely had to be pushed higher, I would use class_weight but pair it with threshold tuning to manage precision as a follow-up.

### A deeper observation: recall vs PR-AUC (operating point vs ranking ability)

A notable point is that with class_weight and SMOTE, recall jumped from 0.64 to 0.92, yet PR-AUC barely moved (0.74 → 0.72). Precision/recall/F1 measure performance at a *single operating point* (the 0.5 threshold), and class_weight/SMOTE shift the decision boundary — much like lowering the threshold. PR-AUC, on the other hand, summarizes ranking ability across *all* thresholds. So the recall surge does not mean "the model became fundamentally better at separating fraud" — it's closer to "the alert criterion was loosened." The unchanged PR-AUC tells us the model's underlying ability to rank fraud above normal hardly changed.

### Neural Net vs Random Forest

The neural net (MLP) reached a PR-AUC of 0.77 — better than the baseline, but short of Random Forest's 0.87. This is not a failure; it's an expected result. This is tabular data where column order and spatial structure carry no meaning, and on such data, tree-based models that split on one feature at a time often beat neural nets that blend all features together. Neural nets also require large amounts of data to fill their many parameters, and ~54,000 rows isn't enough to play to that strength. **"Neural nets don't always win"** — I confirmed firsthand that choosing a tool suited to the problem and data matters more than reaching for the most powerful one.

## 6. What I Learned

### Data leakage (fit scaler & SMOTE on train only)

I learned that the reliability of evaluation depends on how cleanly the test set is preserved. The scaler must be fit on the training set only, and merely applied (transform) to the test set; oversampling like SMOTE must also be applied to training data only. If such processing is done on the full dataset *before* the split, test information leaks into training (leakage) and inflates scores unrealistically. The test set must imitate "unseen future transactions," so it has to stay 100% original to the very end.

### The meaning of the four-line training loop

By unfolding in PyTorch what `model.fit()` hides in a single line in sklearn, I understood the essence of learning. forward (predict) → loss (measure error) → backward (compute each weight's responsibility in reverse) → step (nudge the weights) form one cycle, and repeating it across many batches and epochs gradually refines the model's internal weights toward better values. I confirmed in code that "a model learning" ultimately means its internal numbers (weights) changing.

### Reading metrics by cost, not at face value

I learned that a high score does not equal a good model. The 99.7% accuracy was a phantom that appeared even when every fraud was missed, and a recall of 0.92 was an illusion masking a false-alarm bomb of 0.06 precision. Each metric only becomes meaningful once translated into the real cost of FN (missed fraud) and FP (false alarm). In particular, observing that recall can spike while PR-AUC stays flat trained my eye to distinguish "the metric got better" from "only the decision boundary moved."

### What I'd try next

- **Threshold optimization:** class_weight gives high recall but precision collapses. Instead of the default 0.5, I'd like to search for a cost-aware threshold to see whether recall can be maintained while pulling precision up to a practically usable level.
- **XGBoost / LightGBM:** Since Random Forest was strong on this tabular data, I'd like to check whether boosting models from the same tree-based family perform even better.
- **Cost-based evaluation metric:** I'd like to assign real monetary costs to FN and FP and re-compare the models from a "total cost minimization" perspective rather than by accuracy or F1 (an industrial-engineering approach).
- **SMOTE ratio tuning:** Instead of a full 1:1 balance (`sampling_strategy=1.0`), I'd like to experiment with more conservative ratios to see how the precision-recall balance shifts.
