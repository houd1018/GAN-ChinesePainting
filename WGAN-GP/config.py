'''
config类中定义超参数
'''
import os.path
import torchvision as tv

class Config(object):

    def __init__(self):
        """
        定义一个配置类
        """
        # 0.参数调整
        self.data_path = 'dataset/'
        self.virs = "result"
        self.num_workers = 4  # 多线程
        self.img_size = 96  # 剪切图片的像素大小
        self.batch_size = 256  # 批处理数量
        self.max_epoch = 30000  # 最大轮次
        self.lr1 = 2e-4  # 生成器学习率
        self.lr2 = 2e-4  # 判别器学习率
        self.beta1 = 0.5  # 正则化系数，Adam优化器参数
        self.gpu = True  # 是否使用GPU运算（建议使用）
        self.nz = 100  # 噪声维度
        self.ngf = 64  # 生成器的卷积核个数
        self.ndf = 64  # 判别器的卷积核个数
        self.c_lambda = 10

        # 1.模型保存路径
        self.save_path = 'imgs1/'  # opt.netg_path生成图片的保存路径
        # 判别模型的更新频率要高于生成模型
        self.d_every = 1  # 每一个batch 训练一次判别器
        self.g_every = 3  # 每1个batch训练一次生成模型
        self.save_every = 5  # 每save_every次保存一次模型
        self.netd_path = ''
        self.netg_path = ''

        # 测试数据
        self.gen_img = "result.png"
        # 选择保存的照片
        # 一次生成保存64张图片
        self.gen_num = 64
        self.gen_search_num = 512
        self.gen_mean = 0  # 生成模型的噪声均值
        self.gen_std = 1  # 噪声方差

        # Transform 自定义
        self.transforms = tv.transforms.Compose([
        # 3*96*96
        tv.transforms.Resize(self.img_size),   # 缩放到 img_size* img_size
        # 中心裁剪成96*96的图片。因为本实验数据已满足96*96尺寸，可省略
        tv.transforms.CenterCrop(self.img_size),

        # 随机裁剪为96*96的图片
        # tv.transforms.RandomResizedCrop(size=opt.img_size),

        # 高斯模糊，加入一定噪音，kernel-size应该会决定噪音的数量
        # tv.transforms.GaussianBlur(kernel_size=3),
        # 20%的机会，将图片转为灰度图片
        # tv.transforms.RandomGrayscale(p=0.1),
        # 50%的机会，将图片水平翻转
        # tv.transforms.RandomHorizontalFlip(p=0.5),

        # ToTensor 和 Normalize 搭配使用
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
