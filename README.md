# SpamFilteringUsingNLP

## Dataset
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

## Aim
NLP for Text Classification with NLTK & Scikit-learn for classifying sms as spam or not spam.

# Result

## Classification Report
0 represents the ham class and 1 represents the spam class.

Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.

Precision = TP/TP+FP

Precision for ham = 0.96

Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class.

Recall = TP/TP+FN

F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall.

F1 Score = 2*(Recall * Precision) / (Recall + Precision)

## Confusion Matrix

TP = 1201 (sms which were actually ham and were marked ham)

FP = 53 (some sms which were marked ham but were actually spam)

FN = 9 (very less sms which are actually not spam were marked as spam)

TN = 130 (sms which were marked spam and were spam)
