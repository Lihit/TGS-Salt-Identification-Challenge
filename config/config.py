# coding:utf8
import warnings


class DefaultConfig(object):
    version = 3
    env = 'UNetResNet'  # visdom 环境
    model = 'UNetResNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    ori_data_dir = './data'
    train_data_images_root = './data/train/images'  # 训练集存放路径
    train_data_masks_root = './data/train/masks'  # 训练集mask存放路径
    test_data_root = './data/test/images'  # 测试集存放路径
    load_model_path = 'drive/My Drive/TGS_Result/version3/fold1/experiment2/epoch53_fold1_iou0.846750_thred0.505263.pth'  # 加载预训练的模型的路径，为None代表不加载
    
    logging_file = 'logger.log'
    # 原始图片大小
    ori_image_h = 101
    ori_image_w = 101
    ori_image_channels = 3
    # 图片的输入
    image_h = 128
    image_w = 128
    image_channels = 3

    img_size_ori = 101
    img_size_target = 128

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 1  # how many workers for loading data
    print_freq = 20  # print info every N batch
    pretrained = True
    useValBestThred = True
    BestThred = 0.5
    valPercent = 0.2

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 100
    lr = 0.00125 # initial learning rate
    lr_decay = 0.8  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数

    seed = 1234

    # Loss
    dice_weight =  2.0
    bce_weight = 1.0

    num_folds = 10
    stratified = False
    
    SIZE = 202
    PAD  = 27
    
    fold_iter_num = 1
    
    use_tta = True
    
    experiment_num = 3
    circle_num = 1

def parse(self, kwargs):
    """
    根据字典kwargs 更新 config参数
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))

DefaultConfig.parse = parse