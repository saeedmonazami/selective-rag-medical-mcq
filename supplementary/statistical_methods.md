# Statistical Methods

This document describes the statistical methods used in the experiments.

---

## Accuracy

Accuracy is calculated as the percentage of correctly answered questions:


accuracy = correct_answers / total_questions * 100
z = 1.96 (for 95% confidence)
p = correct / total
denominator = 1 + z^2 / total
center = (p + z^2 / (2 * total)) / denominator
margin = z * sqrt(p * (1 - p) / total + z^2 / (4 * total^2)) / denominator
CI_lower = (center - margin) * 100
CI_upper = (center + margin) * 100

### Implementation

```python
def wilson_ci(correct, total):
    if total == 0:
        return (0.0, 0.0)
    z = 1.96
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (100 * (center - margin), 100 * (center + margin))

    McNemar's Test
McNemar's test is used to compare paired binary outcomes (e.g., Local vs. RAG on the same questions). It tests whether the proportion of correct answers differs significantly between two conditions.
Contingency Table
	
RAG Correct
RAG Wrong
Local Correct
a
b
Local Wrong
c
d
b = cases where Local is correct AND RAG is wrong
c = cases where Local is wrong AND RAG is correct
Formula
chi_squared = (|b - c| - 1)^2 / (b + c)
p_value = erfc(sqrt(chi_squared / 2))
The -1 term is the continuity correction (Edwards' correction).
Interpretation
p < 0.05: Significant difference between the two conditions
p >= 0.05: No significant difference
Implementation
def mcnemar_test(b, c):
    if b + c == 0:
        return 0.0, 1.0
    chi_squared = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = math.erfc(math.sqrt(chi_squared / 2))
    return chi_squared, p_value
    Threshold Calibration
For the selective RAG strategy, we calibrate the gating threshold using a validation set that is separate from the test set.
Process
Split the data: Divide questions into validation set and test set (e.g., 100 validation, 150 test)
Compute answers: For each question, compute both Local and RAG answers
Evaluate thresholds: For each candidate threshold, simulate the selective strategy:
If retrieval score >= threshold AND context exists: use RAG answer
Otherwise: use Local answer
Select best threshold: Choose the threshold that maximizes validation accuracy
Apply to test set: Use the selected threshold on the test set (never used for calibration)
Threshold Selection Criteria
In case of ties (equal validation accuracy), we prefer the higher threshold (more conservative), which uses RAG less frequently.
# Example thresholds tested
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Selection
best_threshold = max(THRESHOLDS, key=lambda th: (validation_accuracy(th), th))
Random Seed
All experiments use SEED = 42 for reproducibility of data splits.
random.seed(42)
indices = list(range(len(dataset)))
random.shuffle(indices)
