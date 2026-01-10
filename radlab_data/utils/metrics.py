import torch.nn
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def preds_to_class_label_softmax(predictions):
    sft_max = torch.nn.Softmax(dim=0)
    sft_preds = sft_max(torch.Tensor(predictions))
    return sft_preds


def preds_to_class_label_argmax(predictions):
    preds = [torch.argmax(p) for p in predictions]
    return preds


def compute_metrics(preds, use_softmax=True, use_argmax=True):
    preds_cl = preds

    if use_softmax:
        preds_cl = preds_to_class_label_softmax(predictions=preds.predictions)

    if use_argmax:
        preds_cl = preds_to_class_label_argmax(predictions=preds_cl)

    prec_val = precision_score(
        y_true=preds.label_ids, y_pred=preds_cl, average="macro"
    )
    recall_val = recall_score(
        y_true=preds.label_ids, y_pred=preds_cl, average="macro"
    )
    f1_val_macro = f1_score(y_true=preds.label_ids, y_pred=preds_cl, average="macro")
    # conf_m_val = confusion_matrix(
    #   y_true=preds.label_ids, y_pred=preds_cl
    # )
    acc_val = accuracy_score(y_true=preds.label_ids, y_pred=preds_cl)
    return {
        "f1": f1_val_macro,
        "precision": prec_val,
        "recall": recall_val,
        "accuracy": acc_val,
        # 'confusion_matrix': conf_m_val
    }
