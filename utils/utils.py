import os
import numpy as np
from sklearn.metrics import jaccard_similarity_score
import pandas as pd
from config.config import DefaultConfig
from skimage.transform import resize
from sklearn.model_selection import BaseCrossValidator, train_test_split
from PIL import Image

opt = DefaultConfig()

def GenDataNameDict():
    np.random.seed(opt.seed)
    DataNameDict = {}
    AllTrainNameList = [trainname.split(
        '.')[-2] for trainname in os.listdir(os.path.join(opt.train_data_images_root))]
    np.random.shuffle(AllTrainNameList)
    AllTestNameList = [testname.split(
        '.')[-2] for testname in os.listdir(os.path.join(opt.test_data_root))]
    ValNameList = AllTrainNameList[:int(
        opt.valPercent * len(AllTrainNameList))]  # 取百分之二十作为验证集
    TrainNameList = AllTrainNameList[int(
        opt.valPercent * len(AllTrainNameList)):]
    DataNameDict['train'] = TrainNameList
    DataNameDict['val'] = ValNameList
    DataNameDict['test'] = AllTestNameList
    return DataNameDict


def CalMeanIOU(score, target):
    metric_by_threshold = []
    for threshold in np.linspace(0, 1, 11):
        val_binary_prediction = (score > threshold).astype(int)

        iou_values = []
        for i in range(score.shape[0]):
            y_mask = target[i]
            p_mask = val_binary_prediction[i]
            iou = jaccard_similarity_score(y_mask.flatten(), p_mask.flatten())
            iou_values.append(iou)
        iou_values = np.array(iou_values)

        accuracies = [
            np.mean(iou_values > iou_threshold)
            for iou_threshold in np.linspace(0.5, 0.95, 10)
        ]
        metric_by_threshold.append((np.mean(accuracies), threshold))

    best_metric, best_threshold = max(metric_by_threshold)
    return best_metric, best_threshold


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'rle_mask'])
        writer.writerows(results)


def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1):
            rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b
    return rle


def encode_rle(predictions):
    return [run_length_encoding(mask) for mask in predictions]


def create_submission(predictions_name, predictions, opt, thred=0.5):
    output = []
    for image_id, mask in zip(predictions_name, predictions):
        mask = resize(np.squeeze(mask),
                      (opt.ori_image_h, opt.ori_image_w),
                      mode='constant', preserve_range=True)
        mask = (mask > thred).astype(np.uint8)
        rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(mask))
        output.append([str(image_id), rle_encoded])

    submission = pd.DataFrame(output, columns=['id', 'rle_mask']).astype(str)
    return submission


def get_mean_rle_mask(test_df):
    pred_fold_list = test_df.filter(regex='thred_pred_fold*').columns.tolist()
    for index in test_df.index:
        mask = np.squeeze((np.sum(test_df.loc[index, pred_fold_list].values, axis=0)
                           > (len(pred_fold_list)/2)).astype(np.uint8))
        mask = resize(np.squeeze(mask),
                      (opt.ori_image_h, opt.ori_image_w),
                      mode='constant', preserve_range=True)
        test_df.loc[index, 'rle_mask'] = ' '.join(
            str(rle) for rle in run_length_encoding(mask))
    return test_df

def get_fold_rle_mask(test_df, fold_num):
    if opt.use_tta:
        pred_fold_list = test_df.filter(regex='tta_thred_pred_fold%d*'%fold_num).columns.tolist()
    else:
        pred_fold_list = test_df.filter(regex='thred_pred_fold%d*'%fold_num).columns.tolist()
    for index in test_df.index:
        mask = np.squeeze(test_df.loc[index,pred_fold_list])
        test_df.loc[index, 'rle_mask'] = ' '.join(
            str(rle) for rle in run_length_encoding(mask))
    return test_df
  
def get_col_rle_mask(test_df, col_name):
    for index in test_df.index:
        mask = np.squeeze(test_df.loc[index,col_name])
        test_df.loc[index, 'rle_mask'] = ' '.join(
            str(rle) for rle in run_length_encoding(mask))
    return test_df


def resize_image(image, target_size):
    """Resize image to target size

    Args:
        image (numpy.ndarray): Image of shape (C x H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Resized image of shape (C x H x W).

    """
    n_channels = image.shape[0]
    resized_image = resize(
        image, (n_channels, target_size[0], target_size[1]), mode='constant')
    return resized_image


def get_crop_pad_sequence(vertical, horizontal):
    top = int(vertical / 2)
    bottom = vertical - top
    right = int(horizontal / 2)
    left = horizontal - right
    return (top, right, bottom, left)


def crop_image(image, target_size):
    """Crop image to target size. Image cropped symmetrically.

    Args:
        image (numpy.ndarray): Image of shape (C x H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Cropped image of shape (C x H x W).

    """
    top_crop, right_crop, bottom_crop, left_crop = get_crop_pad_sequence(image.shape[1] - target_size[0],
                                                                         image.shape[2] - target_size[1])
    cropped_image = image[:, top_crop:image.shape[1] -
                          bottom_crop, left_crop:image.shape[2] - right_crop]
    return cropped_image


def binarize(image, threshold):
    image_binarized = (image[1, :, :] > threshold).astype(np.uint8)
    return image_binarized


def run_length_decoding(mask_rle, shape):
    """
    Based on https://www.kaggle.com/msl23518/visualize-the-stage1-test-solution and modified
    Args:
        mask_rle: run-length as string formatted (start length)
        shape: (height, width) of array to return

    Returns:
        numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[1] * shape[0], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).T


def upsample(img):
    if opt.img_size_ori == opt.img_size_target:
        return img
    return resize(img, (opt.img_size_target, opt.img_size_target), mode='constant', preserve_range=True)
    #res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    #res[:img_size_ori, :img_size_ori] = img
    # return res


def downsample(img):
    if opt.img_size_ori == opt.img_size_target:
        return img
    return resize(img, (opt.img_size_ori, opt.img_size_ori), mode='constant', preserve_range=True)


def GenTrainTest_df():
    train_df = pd.read_csv(os.path.join(
        opt.ori_data_dir, "train.csv"), index_col="id", usecols=[0])
    depths_df = pd.read_csv(os.path.join(
        opt.ori_data_dir, "depths.csv"), index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    train_df["images"] = [Image.open(os.path.join(
        opt.train_data_images_root, '%s.png' % idx)) for idx in train_df.index]
    test_df["images"] = [Image.open(os.path.join(
        opt.test_data_root, '%s.png' % idx)) for idx in test_df.index]
    train_df["masks"] = [Image.open(os.path.join(
        opt.train_data_masks_root, '%s.png' % idx)) for idx in train_df.index]
    train_df["coverage"] = train_df.masks.apply(lambda x:np.sum(np.array(x)>128)/(opt.ori_image_h**2))

    def cov_to_class(val):
        for i in range(0, 11):
            if val * 10 <= i:
                return i
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    return test_df, train_df

def GenTrainTest_df_cv2():
    train_df = pd.read_csv(os.path.join(
        opt.ori_data_dir, "train.csv"), index_col="id", usecols=[0])
    depths_df = pd.read_csv(os.path.join(
        opt.ori_data_dir, "depths.csv"), index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    train_df["images"] = [cv2.imread(os.path.join(
        opt.train_data_images_root, '%s.png' % idx),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255 for idx in train_df.index]
    test_df["images"] = [cv2.imread(os.path.join(
        opt.test_data_root, '%s.png' % idx),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255 for idx in test_df.index]
    train_df["masks"] = [cv2.imread(os.path.join(
        opt.train_data_masks_root, '%s.png' % idx),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255 for idx in train_df.index]
    train_df["coverage"] = train_df.masks.apply(lambda x:np.sum(x>0.5)/(x.shape[0]*x.shape[1]))

    def cov_to_class(val):
        for i in range(0, 11):
            if val * 10 <= i:
                return i
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    return test_df, train_df
