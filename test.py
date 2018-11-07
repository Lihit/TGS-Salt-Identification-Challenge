# coding:utf8
import torch
from sklearn.model_selection import KFold,StratifiedKFold
from torch.utils.data import DataLoader
from torch.autograd import Variable
import models
from config.config import DefaultConfig
from utils.dataset import TGSSaltDataSet
from utils.utils import *
from utils.metrics import *
from utils.visualize import *
from utils.loss import *
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


opt = DefaultConfig()
import logging
# 通过下面的方式进行简单配置输出方式与日志级别
logging.basicConfig(filename=os.path.join('./result',opt.logging_file), level=logging.INFO)



def test(bestModel, test_dataloader):
    if opt.use_tta:
        print('use tta...')
        test_df['tta_pred_fold%d' %n_fold] = np.nan
        test_df['tta_pred_fold%d' %n_fold] = test_df['tta_pred_fold%d' %n_fold].astype(object)
        test_unaugment = test_unaugment_flip
    else:
        print('not use tta...')
        test_df['pred_fold%d' %n_fold] = np.nan
        test_df['pred_fold%d' %n_fold] = test_df['pred_fold%d' %n_fold].astype(object)
        test_unaugment = test_unaugment_null
    if opt.use_gpu:
        bestModel.cuda()
    bestModel.eval()
    with torch.no_grad():
        print('fold testing...')
        for ii, (data, img_ids) in tqdm(enumerate(test_dataloader)):
            input = torch.autograd.Variable(data)
            if opt.use_gpu:
                input = input.cuda()
            score, empty_pro = model(input)
            score = torch.squeeze(F.sigmoid(score))
            if opt.use_tta:
                test_df.loc[list(img_ids), 'tta_pred_fold%d' %
                            n_fold] = list(score.cpu().data.numpy())
            else:
                test_df.loc[list(img_ids), 'pred_fold%d' %
                            n_fold] = list(score.cpu().data.numpy())
        if opt.use_tta:
            test_df['tta_pred_fold%d' % n_fold] = test_df['tta_pred_fold%d' % n_fold].apply(
                lambda x:test_unaugment(np.squeeze(x)))
        else:
            test_df['pred_fold%d' % n_fold] = test_df['pred_fold%d' % n_fold].apply(
                lambda x:test_unaugment(np.squeeze(x)))
    bestModel.train()
    

    

if __name__ == '__main__':
    ## 1.load data
    test_df, train_df = GenTrainTest_df_cv2()
    # loading data
    test_data = TestDataSet(test_df.index.tolist(), opt)
    test_dataloader = DataLoader(
        test_data, batch_size=opt.batch_size, shuffle=False)
    ## 2.load model
    print('loading model...')
    model = UNetResNet34(pretrained=False)
    print('successfully loading model...')
    if opt.load_model_path:#that should be your best model path
        print('loading pretrained model:%s'%opt.load_model_path)
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    #开始测试,数据会保存在test_df里
    opt.use_tta = True
    test(model, test_dataloader)
    #开始测试,数据会保存在test_df里
    opt.use_tta = False
    test(model, test_dataloader)
    test_df['mean_pred_fold'] = (test_df['tta_pred_fold%d' %n_fold].values+test_df['pred_fold%d' %n_fold].values)/2
    #please get the best thred by the model name ,0.5 may not be the best
    test_df['mean_thred_pred_fold'] = test_df['mean_pred_fold'].apply(
                lambda x:(x> 0.5).astype(np.int8))
    test_df = get_col_rle_mask(test_df, 'mean_thred_pred_fold')
    test_df.reset_index(inplace=True)
    submission = test_df[['id','rle_mask']]
    submission = submission.astype(str)
    submission.to_csv(os.path.join('./result','submission_tta_mean.csv'), index=False)