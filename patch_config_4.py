import numpy as np
import torch
from torch import optim
import torch.nn as nn
from load_data_4 import MeanProbExtractor_yolov5, \
    MeanProbExtractor_yolov2, MeanProbExtractor_yolov8, MaxProbExtractor_yolov5
from models.common import DetectMultiBackend

dota_v1_5 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
             'ship', 'tennis-court',
             'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
             'helicopter', 'container-crane']
sandtable = ['military_vehicle', 'tank', 'warship', 'fighter_aircraft', 'carrier-based_aircraft', 'civil_aircraft',
             'barracks']
dota_sandtable = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
                  'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter',
                  'roundabout', 'soccer-ball-field', 'swimming-pool', 'military_vehicle', 'tank', 'warship',
                  'fighter_aircraft', 'carrier-based_aircraft', 'civil_aircraft', 'barrackss']

class BaseConfig:
    """
    Default parameters for all config files.
    """

    def __init__(self):

        # 是否使用启发式算法初始化
        self.use_heuristic_init = True
        self.heuristic_method = 'PSO'  # 使用粒子群优化
        self.heuristic_max_iter = 100  # PSO 最大迭代次数
        self.pso_inertia_max = 0.9  # 最大惯性权重
        self.pso_inertia_min = 0.4  # 最小惯性权重
        self.pso_cognitive_max = 2.0
        self.pso_cognitive_min = 0.5
        self.pso_social_max = 2.0
        self.pso_social_min = 0.5
        #self.num_particles = 30  # 粒子数量
        self.num_particles = 50  # 增加粒子数量
        self.pso_inertia = 0.5  # 惯性权重
        self.pso_cognitive = 1.5  # 认知因子
        self.pso_social = 1.5  # 社会因子
        self.pso_interval = 20 # 每隔多少轮切换到 PSO

        """
        Set the defaults.
        """
        self.img_dir = "dataset/sandtable/train"
        self.val_img_dir = "dataset/sandtable/val"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 50 #补丁分辨率
        self.start_learning_rate = 1e-3
        #self.start_learning_rate = 0.0005
        self.seed = 1176426343
        self.img_size = 416
        self.imgsz = (416, 416)
        self.num_classes = 80
        self.max_det = 300
        self.cls_id = 0
        self.class_names = dota_v1_5
        self.scale = 0.15
        self.minangle = -15
        self.maxangle = 15
        self.min_brightness = -0.2
        self.max_brightness = 0.2
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.noise_factor = 0.1
        self.offsetx = 0.02
        self.offsety = 0.05
        self.by_rect = True
        self.rand_loc = False

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

        self.patch_name = 'base'
        self.device = torch.device('cuda:0') #使用索引为1的GPU进行计算。
        self.dtype = torch.float32 # 使用32位浮点数计算。

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50, verbose=True)
        self.max_tv = 0

        self.batch_size = 4

        self.loss_target = lambda obj, cls: obj * cls  # self.loss_target(obj, cls) return obj * cls

        self.generator = torch.Generator(self.device).manual_seed(self.seed)

        self.init_num_inference_steps = 50
        self.num_inference_steps = 4
        self.start_time_step = 601
        self.end_time_step = 1
        self.do_classifier_free_guidance = True
        self.guidance_scale = 7

class yolov2(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolo.cfg"
        self.weights = "weights/yolov2.weights"
        self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov2"
        self.mode = 'yolov2'
        self.patch_name = 'yolov2'
        self.max_tv = 0.165
        self.batch_size = 16

        self.loss_target = lambda obj, cls: obj
        from darknetv2 import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov2(self.cls_id, self.model.num_classes,
                                                       self.model.num_anchors,
                                                       self.model.anchors,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)

class yolov3(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolov3.cfg"
        self.weights = "weights/yolov3.weights"
        self.img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/images'
        self.lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/labels"
        self.val_img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/images'
        self.val_lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/labels"
        # self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov3"
        # self.val_lab_dir = "dataset/inria/Test/pos/yolo-labels_yolov3"
        self.mode = 'yolov5'  # 3、4、5模型预测都一样，所以统一用V5表示。
        self.patch_name = 'yolov3'
        self.max_tv = 0.165
        self.batch_size = 8  # 单批次处理图片数量。

        self.loss_target = lambda obj, cls: obj
        from pytorchyolo.models import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_darknet_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image

        self.prob_extractor = MaxProbExtractor_yolov5(self.cls_id, self.num_classes, self.loss_target)
        # self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
        #                                                self.loss_target, self.conf_thres,
        #                                                self.iou_thres,
        #                                                self.max_det)

class yolov3_dota(BaseConfig):
    def __init__(self):
        super().__init__()
        self.weights = "weights/yolov3_dotasp.pt"
        self.img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/images/'
        self.lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/labels/"
        self.val_img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/images/'
        self.val_lab_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/labels/'
        self.mode = 'yolov5'
        self.patch_name = 'yolov3_dota'
        self.max_tv = 0.165
        self.batch_size = 1
        self.loss_target = lambda obj, cls: obj
        self.num_classes = 16
        self.imgsz = (1024, 1024)
        self.img_size = 1024
        self.scale = 0.15
        self.cls_id = 0
        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        # 使用 DataParallel 在多个设备上并行训练
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])  # 使用 GPU 0 和 GPU 1
        self.model = torch.nn.DataParallel(self.model, device_ids=[0])
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)
class yolov3tiny(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolov3-tiny.cfg"
        self.weights = "weights/yolov3-tiny.weights"
        self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov3tiny"
        self.mode = 'yolov5'
        self.patch_name = 'yolov3tiny'
        self.max_tv = 0.165
        self.batch_size = 24

        self.loss_target = lambda obj, cls: obj
        from pytorchyolo.models import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_darknet_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)

class yolov3tiny_mpii(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolov3-tiny.cfg"
        self.weights = "weights/yolov3-tiny.weights"
        self.img_dir = 'dataset/mpii/train'
        self.lab_dir = "dataset/mpii/train/labels_yolov3tiny"
        self.val_img_dir = 'dataset/mpii/test'
        self.mode = 'yolov5'
        self.patch_name = 'yolov3tiny-mpii'
        self.max_tv = 0.165
        self.batch_size = 24

        self.loss_target = lambda obj, cls: obj
        from pytorchyolo.models import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_darknet_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.6  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)

class yolov3tiny_mix(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolov3-tiny.cfg"
        self.weights = "weights/yolov3-tiny.weights"
        self.img_dir = 'dataset/mix/train'
        self.lab_dir = "dataset/mix/train/labels_yolov3tiny"
        self.val_img_dir = 'dataset/mix/test'
        self.mode = 'yolov5'
        self.patch_name = 'yolov3tiny-mix'
        self.max_tv = 0.165
        self.batch_size = 24

        self.loss_target = lambda obj, cls: obj
        from pytorchyolo.models import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_darknet_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)

class yolov4(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolov4.cfg"
        self.weights = "weights/yolov4.weights"
        self.img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/images'
        self.lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/labels"
        self.val_img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/images'
        self.val_lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/labels"

        # self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov4"
        self.mode = 'yolov5'
        self.patch_name = 'yolov4'
        self.max_tv = 0.165
        self.batch_size = 16

        self.loss_target = lambda obj, cls: obj
        from pytorchyolo.models import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_darknet_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)
class yolov4tiny(BaseConfig):
    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolov4-tiny.cfg"
        self.weights = "weights/yolov4-tiny.weights"
        self.img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/images'
        self.lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/labels"
        self.val_img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/images'
        self.val_lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/labels"

        # self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov4tiny"
        self.mode = 'yolov5'
        self.patch_name = 'yolov4tiny'
        self.max_tv = 0.165
        self.batch_size = 16

        self.loss_target = lambda obj, cls: obj
        from pytorchyolo.models import Darknet

        self.model = Darknet(self.cfgfile)
        self.model.load_darknet_weights(self.weights)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 300  # maximum detections per image
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)

class yolov5s(BaseConfig):
    def __init__(self):
        super().__init__()

        self.patch_name = 'yolov5s'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.weights = 'weights/yolov5s-416.pt'
        # 训练集和验证集
        self.img_dir = 'dataset/sandtable/train/images'
        self.lab_dir = "dataset/sandtable/train/labels_yolov5s_st"
        self.val_img_dir = 'dataset/sandtable/val/images'
        self.val_lab_dir = "dataset/sandtable/val/labels_yolov5s_st"

        self.mode = 'yolov5'
        self.imgsz = (416, 416)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False).eval()

        # self.model = DetectMultiBackend(self.weights,
        #                                 device=self.device,
        #                                 dnn=False).eval()
        self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
                                                       self.loss_target, self.conf_thres,
                                                       self.iou_thres,
                                                       self.max_det)

class yolov5s_st(BaseConfig): # 相关通用解释
    def __init__(self):
        super().__init__()

        # 补丁名字
        self.patch_name = 'yolov5s_st'
        # 平滑损失最大值
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj
        # 模型权重路径
        self.weights = 'weights/yolov5s_st.pt'
        # 训练集和验证集
        self.img_dir = 'dataset/fushi/train/images'
        self.lab_dir = "dataset/fushi/train/labels_yolov5s_st"
        self.val_img_dir = 'dataset/fushi/val/images'
        self.val_lab_dir = "dataset/fushi/val/labels_yolov5s_st"
        # self.weights = 'weights/yolov5s-416.pt'
        # # 训练集和验证集
        # self.img_dir = 'dataset/sandtable/train/images'
        # self.lab_dir = "dataset/sandtable/train/labels_yolov5s_st"
        # self.val_img_dir = 'dataset/sandtable/val/images'
        # self.val_lab_dir = "dataset/sandtable/val/labels_yolov5s_st"

        # 补丁相对检测框比例
        self.scale = 0.2
        # 补丁增强相关
        # 旋转角度
        self.minangle = -45
        self.maxangle = 45
        # 亮度
        self.min_brightness = -0.3
        self.max_brightness = 0.3
        # 对比度
        self.min_contrast = 0.7
        self.max_contrast = 1.3
        # 噪声比例
        self.noise_factor = 0.25
        # 随机位移
        self.offsetx = 0.05
        self.offsety = 0.05
        # 是否启用随机偏移
        self.rand_loc = True  # 补丁的位置是否随机。
        # 是否适应矩形
        self.by_rect = False  # 根据检测框的形状改变补丁的形状。

        # # Stable Diffusion相关参数
        # self.num_inference_steps = 3
        # self.guidance_scale = 5
        # 模型支持的类别名称
        self.class_names = sandtable  # 训练模型里类的类名。
        self.mode = 'yolov5'  # 后处理操作的类别：
        # 类别数量
        self.num_classes = 7
        self.cls_id = 3  # 希望攻击的类别。
        self.batch_size = 16
        # 模型输入图像大小
        self.imgsz = (640, 640)
        self.img_size = 640
        # NMS置信度和IOU阈值
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
         #  模型加载（ultralytics yolov3/v5/v8
        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        # 损失函数选择
        # 定义损失的损失函数：置信度最高的框损失情况作为损失函数。
        self.prob_extractor = MaxProbExtractor_yolov5(self.cls_id, self.num_classes, self.loss_target)
        # 定义损失的损失函数：所有框置信度损失的平均值。
        # self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
        #                                                self.loss_target, self.conf_thres,
        #                                                self.iou_thres,
        #                                                self.max_det)

class yolov5s_dotast(BaseConfig):
    def __init__(self):
        super().__init__()

        self.patch_name = 'yolov5s_dotast'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.weights = 'weights/yolov5s_dotast.pt'
        self.img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/images'
        self.lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/labels"
        self.val_img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/images'
        self.val_lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/labels"
        self.scale = 0.2
        self.minangle = -45
        self.maxangle = 45
        self.min_brightness = -0.3
        self.max_brightness = 0.3
        self.min_contrast = 0.7
        self.max_contrast = 1.3
        self.noise_factor = 0.25
        self.offsetx = 0.05
        self.offsety = 0.05
        self.rand_loc = True
        self.by_rect = False

        # self.num_inference_steps = 3
        # self.guidance_scale = 5
        self.class_names = dota_v1_5
        self.mode = 'yolov5'
        self.num_classes = 16
        self.cls_id = 0
        self.batch_size = 1
        self.imgsz = (1024, 1024)
        self.img_size = 1024
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold

        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        self.prob_extractor = MaxProbExtractor_yolov5(self.cls_id, self.num_classes, self.loss_target)
        # self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
        #                                                self.loss_target, self.conf_thres,
        #                                                self.iou_thres,
        #                                                self.max_det)

class yolov8s(BaseConfig):
    def __init__(self):
        super().__init__()

        self.patch_name = 'yolov8s_dota'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.weights = 'weights/yolov8forDotas.pt'
        self.img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/images'
        self.lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/labels_yolov8s_dota"
        self.val_img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/images'
        self.val_lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/labels_yolov8s_dota"
        self.scale = 0.2
        self.minangle = -45
        self.maxangle = 45
        self.min_brightness = -0.3
        self.max_brightness = 0.3
        self.min_contrast = 0.7
        self.max_contrast = 1.3
        self.noise_factor = 0.25
        self.offsetx = 0.05
        self.offsety = 0.05
        self.rand_loc = True
        self.by_rect = False

        # self.num_inference_steps = 3
        # self.guidance_scale = 5
        self.class_names = dota_v1_5
        self.mode = 'yolov8'
        self.num_classes = 16
        self.cls_id = 0
        self.batch_size = 1
        self.imgsz = (1024, 1024)
        self.img_size = 1024
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold

        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        self.prob_extractor = MaxProbExtractor_yolov5(self.cls_id, self.num_classes, self.loss_target)

        # self.patch_name = 'yolov8s'
        # self.max_tv = 0.165
        #
        # self.loss_target = lambda obj, cls: obj
        #
        # self.weights = 'weights/yolov8forDotas.pt'
        # self.img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/images'
        # self.lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/labels"
        # self.val_img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/images'
        # self.val_lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/labels"
        # # self.lab_dir = "dataset/inria/Train/pos/yolo-labels_yolov8s"
        #
        # self.mode = 'yolov8'
        # self.imgsz = (1024, 1024)
        # self.conf_thres = 0.4  # confidence threshold
        # self.iou_thres = 0.45  # NMS IOU threshold
        #
        #
        #
        # self.model = DetectMultiBackend(self.weights,
        #                                 device=self.device,
        #                                 dnn=False).eval()
        # self.prob_extractor = MeanProbExtractor_yolov8(self.cls_id, self.num_classes,
        #                                                self.loss_target, self.conf_thres,
        #                                                self.iou_thres,
        #                                                self.max_det)

class yolo11s(BaseConfig):
    def __init__(self):
        super().__init__()

        self.patch_name = 'yolo11s_dota'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.weights = 'weights/yolo11_dotast.pt'
        self.img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP_new/train/images'
        self.lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/labels"
        self.val_img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/images'
        self.val_lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/labels_yolo11s_dota"
        self.scale = 0.2
        self.minangle = -45
        self.maxangle = 45
        self.min_brightness = -0.3
        self.max_brightness = 0.3
        self.min_contrast = 0.7
        self.max_contrast = 1.3
        self.noise_factor = 0.25
        self.offsetx = 0.05
        self.offsety = 0.05
        self.rand_loc = True
        self.by_rect = False

        # self.num_inference_steps = 3
        # self.guidance_scale = 5
        self.class_names = dota_v1_5
        self.mode = 'yolov11'
        self.num_classes = 16
        self.cls_id = 0
        self.batch_size = 1
        self.imgsz = (1024, 1024)
        self.img_size = 1024
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold

        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        self.prob_extractor = MaxProbExtractor_yolov5(self.cls_id, self.num_classes, self.loss_target)

class yolov5n(BaseConfig):
    def __init__(self):
        super().__init__()

        self.patch_name = 'yolov5n'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.weights = '/home/fyq/APPA/AP-PA/yolov5/runs/train/dota_yolov5n/weights/best.pt'  # 假设权重文件名为 yolov5n.pt
        self.img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/images'
        self.lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/labels_yolov5n"
        self.val_img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/images'
        self.val_lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/labels_yolov5n"
        self.scale = 0.2
        self.minangle = -45
        self.maxangle = 45
        self.min_brightness = -0.3
        self.max_brightness = 0.3
        self.min_contrast = 0.7
        self.max_contrast = 1.3
        self.noise_factor = 0.25
        self.offsetx = 0.05
        self.offsety = 0.05
        self.rand_loc = True
        self.by_rect = False

        # self.num_inference_steps = 3
        # self.guidance_scale = 5
        self.class_names = dota_v1_5
        self.mode = 'yolov5'
        self.num_classes = 16
        self.cls_id = 0
        self.batch_size = 1
        self.imgsz = (1024, 1024)
        self.img_size = 1024
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.7  # NMS IOU threshold

        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        self.prob_extractor = MaxProbExtractor_yolov5(self.cls_id, self.num_classes, self.loss_target)
        # self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
        #                                                self.loss_target, self.conf_thres,
        #                                                self.iou_thres,
        #                                                self.max_det)

class yolov5m(BaseConfig):
    def __init__(self):
        super().__init__()

        self.patch_name = 'yolov5m'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.weights = '/home/fyq/APPA/AP-PA/yolov5/runs/train/dota_yolov5m2/weights/best.pt'  # 假设权重文件名为 yolov5m.pt
        self.img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/images'
        self.lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/labels_yolov5m"
        self.val_img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/images'
        self.val_lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/labels_yolov5m"
        self.scale = 0.2
        self.minangle = -45
        self.maxangle = 45
        self.min_brightness = -0.3
        self.max_brightness = 0.3
        self.min_contrast = 0.7
        self.max_contrast = 1.3
        self.noise_factor = 0.25
        self.offsetx = 0.05
        self.offsety = 0.05
        self.rand_loc = True
        self.by_rect = False

        # self.num_inference_steps = 3
        # self.guidance_scale = 5
        self.class_names = dota_v1_5
        self.mode = 'yolov5'
        self.num_classes = 16
        self.cls_id = 0
        self.batch_size = 1
        self.imgsz = (1024, 1024)
        self.img_size = 1024
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold

        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        self.prob_extractor = MaxProbExtractor_yolov5(self.cls_id, self.num_classes, self.loss_target)
        # self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
        #                                                self.loss_target, self.conf_thres,
        #                                                self.iou_thres,
        #                                                self.max_det)

class yolov5l(BaseConfig):
    def __init__(self):
        super().__init__()

        self.patch_name = 'yolov5l'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.weights = '/home/fyq/APPA/AP-PA/yolov5/runs/train/dota_yolov5l/weights/best.pt'  # 假设权重文件名为 yolov5l.pt
        self.img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/images'
        self.lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/labels_yolov5l"
        self.val_img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/images'
        self.val_lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/labels_yolov5l"
        self.scale = 0.2
        self.minangle = -45
        self.maxangle = 45
        self.min_brightness = -0.3
        self.max_brightness = 0.3
        self.min_contrast = 0.7
        self.max_contrast = 1.3
        self.noise_factor = 0.25
        self.offsetx = 0.05
        self.offsety = 0.05
        self.rand_loc = True
        self.by_rect = False

        # self.num_inference_steps = 3
        # self.guidance_scale = 5
        self.class_names = dota_v1_5
        self.mode = 'yolov5'
        self.num_classes = 16
        self.cls_id = 0
        self.batch_size = 1
        self.imgsz = (1024, 1024)
        self.img_size = 1024
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold

        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        self.prob_extractor = MaxProbExtractor_yolov5(self.cls_id, self.num_classes, self.loss_target)
        # self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
        #                                                self.loss_target, self.conf_thres,
        #                                                self.iou_thres,
        #                                                self.max_det)

class yolov5x(BaseConfig):
    def __init__(self):
        super().__init__()

        self.patch_name = 'yolov5x'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.weights = '/home/fyq/APPA/AP-PA/yolov5/runs/train/dota_yolov5x10/weights/best.pt'  # 假设权重文件名为 yolov5x.pt
        self.img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/images'
        self.lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/train/labels_yolov5x"
        self.val_img_dir = '/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/images'
        self.val_lab_dir = "/home/datadisk/ShellE/Project/Sample/DOTA_SP/val/labels_yolov5x"
        self.scale = 0.2
        self.minangle = -45
        self.maxangle = 45
        self.min_brightness = -0.3
        self.max_brightness = 0.3
        self.min_contrast = 0.7
        self.max_contrast = 1.3
        self.noise_factor = 0.25
        self.offsetx = 0.05
        self.offsety = 0.05
        self.rand_loc = True
        self.by_rect = False

        # self.num_inference_steps = 3
        # self.guidance_scale = 5
        self.class_names = dota_v1_5
        self.mode = 'yolov5'
        self.num_classes = 16
        self.cls_id = 0
        self.batch_size = 1
        self.imgsz = (1024, 1024)
        self.img_size = 1024
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold

        self.model = DetectMultiBackend(self.weights,
                                        device=self.device,
                                        dnn=False).eval()
        self.prob_extractor = MaxProbExtractor_yolov5(self.cls_id, self.num_classes, self.loss_target)
        # self.prob_extractor = MeanProbExtractor_yolov5(self.cls_id, self.num_classes,
        #                                                self.loss_target, self.conf_thres,
        #                                                self.iou_thres,
        #                                                self.max_det)


patch_configs = {
    "yolov2": yolov2,
    "yolov3": yolov3,
    "yolov3_dota": yolov3_dota,
    "yolov3tiny": yolov3tiny,
    "yolov3tiny-mpii": yolov3tiny_mpii,
    "yolov3tiny-mix": yolov3tiny_mix,
    "yolov4": yolov4,
    "yolov4tiny": yolov4tiny,
    "yolov5s": yolov5s,
    "yolov5s_st": yolov5s_st,
    "yolov5s_dotast": yolov5s_dotast,
    "yolov8s": yolov8s,
    "yolo11s": yolo11s,
    "yolov5n": yolov5n,
    "yolov5m": yolov5m,
    "yolov5l": yolov5l,
    "yolov5x": yolov5x,
}