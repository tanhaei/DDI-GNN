from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Example ground truth and predictions (replace with actual data)
y_true = [1, 0, 1, 0]  # Ground truth (interaction labels)
y_pred = [1, 0, 0, 0]  # Model predictions

# Calculate Precision, Recall, F1, and AUC
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"AUC: {auc}")
