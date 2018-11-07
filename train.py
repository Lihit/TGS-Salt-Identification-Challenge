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
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


opt = DefaultConfig()
import logging
# 通过下面的方式进行简单配置输出方式与日志级别
logging.basicConfig(filename=os.path.join('./result',opt.logging_file), level=logging.INFO)


def train(n_fold, train_idx, valid_idx):
    BestModel = None
    BestThred = 0.5
    BestIou = -1
    LrDecayCount = 0
    previous_loss = 1e100
    previous_val_meaniou = -1
    lr = opt.lr
    logging.info('Fold:%d...' % n_fold)
    train_data = TrainDataSet(train_df.iloc[train_idx].index.tolist(), opt)
    val_data = ValDataSet(train_df.iloc[valid_idx].index.tolist(), opt)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                    shuffle=True)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                    shuffle=False)
    criterion1 = nn.CrossEntropyLoss()
    #criterion1 = torch.nn.BCELoss()
    #criterion1 = torch.nn.BCEWithLogitsLoss()
    #criterion = mixed_dice_bce_loss(opt.dice_weight, opt.bce_weight)
    #criterion = RobustFocalLoss2d()
    criterion2 = lovasz_hinge
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #Rateschduler = CyclicScheduler(base_lr=0.005, max_lr=0.015)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=lr, momentum=0.9, weight_decay=0.0001)
    # train
    train_loss = []
    val_loss = []
    val_iou = []
    val_bestthred_list = []
    val_acc_list = []
    train_acc_list = []
    for epoch in range(opt.max_epoch):
        logging.info('Epoch:%d...' % epoch)
        train_loss_batch = []
        train_acc_batch = []
        for ii, (data, mask, label, label_empty) in tqdm(enumerate(train_dataloader)):
            # learning rate schduler 
#             if Rateschduler is not None:
#                 lr = Rateschduler.get_rate()
#                 if lr<0 : break
#                 adjust_learning_rate(optimizer, lr)
            #lr = cosine_anneal_schedule(epoch)
            #adjust_learning_rate(optimizer, lr)
            # train model
            input = Variable(data.cuda())
            target = Variable(mask.cuda())
            label = Variable(label.long().cuda())
            label_empty = Variable(label_empty.float().cuda())
            #print(input.size())
            #print(target.size())
            #model = model.cuda()
            #input = data.cuda()
            #target = label.cuda()
    
            optimizer.zero_grad()
            score, empty_pro = model(input)
            #print(score.size())
            loss1 = criterion1(empty_pro,label)
            loss2 = criterion2(score.squeeze(), target.squeeze())
            loss = 0.05*loss1+loss2
            loss.backward()
            optimizer.step()
            if ii % opt.print_freq == opt.print_freq - 1:
                logging.info('Fold:%d\t Epoch:%d\t Batch:%d\t loss1:%f, loss2:%f, loss:%f' %
                        (n_fold, epoch, ii, loss1.data[0], loss2.data[0], loss.data[0]))
                print('Fold:%d\t Epoch:%d\t Batch:%d\t loss1:%f, loss2:%f, loss:%f' %
                        (n_fold, epoch, ii, loss1.data[0], loss2.data[0], loss.data[0]))
            train_acc_batch_ = accuracy(target.squeeze().cpu().data.numpy(), score.squeeze().cpu().data.numpy())
            train_loss_batch.append(loss.data[0])
            train_acc_batch.append(train_acc_batch_)
            # validate and visualize
            
        ## validation
        model.eval()
        with torch.no_grad():
            vallosslist = []
            meanioulist = []
            val_predictions = []
            val_masks = []
            for ii, data in tqdm(enumerate(val_dataloader)):
                val_input, val_target, val_label, label_empty = data
                val_input = Variable(val_input.cuda())
                val_target = Variable(val_target).cuda()
                val_label = Variable(val_label.long()).cuda()
                label_empty = Variable(label_empty.float().cuda())
                score, empty_pro = model(val_input)
                val_predictions.append(torch.squeeze(F.sigmoid(score)).cpu().data.numpy())
                val_masks.append(val_target.squeeze().cpu().data.numpy())
                valloss1 = criterion1(empty_pro,val_label)
                valloss2 = criterion2(score.squeeze(), val_target.squeeze())
                valloss = 0.05*valloss1+valloss2
                vallosslist.append(valloss.cpu().data[0])
            val_predictions = np.vstack(val_predictions)
            val_masks = np.vstack(val_masks)
            bestThred, bestIOU = GetBestThred(val_masks, val_predictions)
            val_acc = accuracy(val_masks, val_predictions, thred = bestThred)
            val_meanloss, val_acc, val_meaniou, val_bestthred = np.mean(vallosslist), val_acc, bestIOU, bestThred
        model.train()
        
        print('Fold:%d\t Epoch:%d\t meanTrainLoss:%f\t meanTrainAcc:%f\t meanValLoss:%f\t meanValAcc:%f\t meanValIOU:%f\t BestValThred:%f' %
                (n_fold, epoch, np.mean(train_loss_batch), np.mean(train_acc_batch), val_meanloss, val_acc, val_meaniou, val_bestthred))
        logging.info('Fold:%d\t Epoch:%d\t meanTrainLoss:%f\t meanTrainAcc:%f\t meanValLoss:%f\t meanValAcc:%f\t meanValIOU:%f\t BestValThred:%f' %
                (n_fold, epoch, np.mean(train_loss_batch), np.mean(train_acc_batch), val_meanloss, val_acc, val_meaniou, val_bestthred))
        train_acc_list.append(np.mean(train_acc_batch))
        train_loss.append(np.mean(train_loss_batch))
        val_loss.append(val_meanloss)
        val_iou.append(val_meaniou)
        val_bestthred_list.append(val_bestthred)
        val_acc_list.append(val_acc)
        # update learning rate
        if val_meaniou < previous_val_meaniou:
            LrDecayCount += 1
        else:
            LrDecayCount = 0
            previous_val_meaniou = val_meaniou
        if LrDecayCount == 5:
            lr = lr * opt.lr_decay
            #第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            LrDecayCount = 0
        print('current lr:',lr)
        #save the best model
        if val_meaniou>BestIou:
            print('update best model')
            BestIou = val_meaniou
            BestThred = val_bestthred
            #保存模型
            BestModel = model
            model.save(os.path.join(ExpPath,'epoch%d_fold%d_iou%f_thred%f.pth'%(epoch,n_fold,BestIou,BestThred)))
    return BestModel,BestIou,BestThred,train_loss,train_acc_list,val_loss,val_acc_list,val_iou,val_bestthred_list

def plot_result(train_loss,train_acc_list,val_loss,val_acc_list,val_iou,val_bestthred_list):
    plt.figure(1)
    plt.title('train loss vs val loss')
    plt.plot(train_loss,label='train loss')
    plt.plot(val_loss,label='val loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(ExpPath,'train-loss-vs-val-loss.png'))
    plt.figure(2)
    plt.title('val mean iou')
    plt.plot(val_iou)
    plt.ylabel('mean iou')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(ExpPath,'val-mean-iou.png'))
    plt.figure(3)
    plt.title('best thred')
    plt.plot(val_bestthred_list)
    plt.ylabel('val_bestthred')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(ExpPath,'best-thred.png'))
    plt.figure(4)
    plt.title('train acc vs val acc')
    plt.plot(train_acc_list,label='train acc')
    plt.plot(val_acc_list,label='val acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(ExpPath,'train-acc-vs-val-acc.png'))
    plt.show()

if __name__ == '__main__':
    ## 1. load data
    test_df, train_df = GenTrainTest_df_cv2()
    ## 2. load model
    print('loading model...')
    model = UNetResNet34(pretrained=False)
    print('successfully loading model...')
    if opt.load_model_path:
        print('loading pretrained model:%s'%opt.load_model_path)
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    ## 3.划分数据集和验证集
    if opt.stratified:
        print('------>StratifiedKFold:%d...' % opt.num_folds)
        logging.info('------>StratifiedKFold:%d...' % opt.num_folds)
        folds = StratifiedKFold(
            n_splits=opt.num_folds, shuffle=True, random_state=opt.seed)
    else:
        print('------>KFold%d...' % opt.num_folds)
        logging.info('------>KFold%d...' % opt.num_folds)
        folds = KFold(n_splits=opt.num_folds,
                    shuffle=True, random_state=opt.seed)
    fold_ = folds.split(train_df['images'], train_df['coverage_class'])
    n_fold = opt.fold_iter_num
    print('This is the %dth fold'%n_fold)
    logging.info('This is the %dth fold'%n_fold)
    for i in range(n_fold):
        (train_idx, valid_idx) = next(fold_)
    BestModel,BestIou,BestThred,train_loss,train_acc_list,val_loss,val_acc_list,val_iou,val_bestthred_list = train(n_fold,train_idx, valid_idx)

    #画出训练过程
    plot_result(train_loss,train_acc_list,val_loss,val_acc_list,val_iou,val_bestthred_list)
    #保存最好的模型
    BestModel.save(os.path.join('./checkpoints','Best_fold%d_iou%f_thred%f.pth'%(n_fold,BestIou,BestThred)))