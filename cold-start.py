# Example Cold-Start Evaluation (using unseen drugs in test set)
# Here, y_true contains true interactions and y_pred contains model predictions for unseen drugs
y_true_cold_start = [1, 0, 1, 0]  # Ground truth for unseen drugs
y_pred_cold_start = [1, 0, 0, 0]  # Model predictions for unseen drugs

# Calculate Cold-Start Precision, Recall, F1, and AUC
precision_cold_start = precision_score(y_true_cold_start, y_pred_cold_start)
recall_cold_start = recall_score(y_true_cold_start, y_pred_cold_start)
f1_cold_start = f1_score(y_true_cold_start, y_pred_cold_start)
auc_cold_start = roc_auc_score(y_true_cold_start, y_pred_cold_start)

print(f"Cold-Start Precision: {precision_cold_start}")
print(f"Cold-Start Recall: {recall_cold_start}")
print(f"Cold-Start F1-score: {f1_cold_start}")
print(f"Cold-Start AUC: {auc_cold_start}")
