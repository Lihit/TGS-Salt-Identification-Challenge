import numpy as np

# Score the model and do a threshold optimization by the best IoU.

# src: https://www.kaggle.com/aglotero/another-iou-metric


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(),
                           bins=([0, 0.5, 1], [0, 0.5, 1]))
    intersection = temp1[0]
    area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(
            false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in, thred=0.5):
    #y_pred_in = np.int32(y_pred_in > thred)  # added by sgx 20180728
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(test_unaugment_null(y_true_in[batch]),  np.int32(test_unaugment_null(y_pred_in[batch]) > thred))
        metric.append(value)
    return np.mean(metric)


def my_iou_metric(label, pred):
    return iou_metric_batch(label, pred)


def GetBestThred(y_valid_ori, preds_valid):
    thresholds = np.linspace(0.3,0.6,20)
    ious = np.array([iou_metric_batch(y_valid_ori, 
        preds_valid, threshold) for threshold in thresholds])
    maxindex = np.argmax(ious)
    bestThred, bestIOU = thresholds[maxindex], ious[maxindex]
    return bestThred, bestIOU
  
def accuracy(y_valid_ori, preds_valid, thred = 0.5):
    preds_valid = (preds_valid>thred).astype(np.uint8)
    y_valid_ori.astype(np.uint8)
    return np.mean(y_valid_ori==preds_valid)