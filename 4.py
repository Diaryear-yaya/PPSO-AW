import shutil
import time
import warnings
import torch
import matplotlib.pyplot as pl
import torch.nn.functional as F
import pandas as pd
import psutil
from torch import optim
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image, ImageDraw
import post_util
import wandb
from load_data_4 import *
from post_util import plot_boxes
from tools import map_cal
from utils.general import xywh2xyxy
import patch_config_4
import torchvision.transforms.functional as TF
from utils.segment.general import process_mask_native
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
warnings.filterwarnings("ignore")
#优化 GPU 内存的使用，防止因单次分配过大而导致内存碎片化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
# Expand to show
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)

class PatchTrainer:
    def __init__(self, mode):
        self.epoch_length = 0
        if isinstance(mode, patch_config_4.BaseConfig):
            self.config = mode
        else:
            self.config = patch_config_4.patch_configs[mode]()
        self.model = self.config.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.prob_extractor = self.config.prob_extractor.cuda()
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer(scale=self.config.scale,
                                                  minangle=self.config.minangle,
                                                  maxangle=self.config.maxangle,
                                                  min_brightness=self.config.min_brightness,
                                                  max_brightness=self.config.max_brightness,
                                                  offsetx=self.config.offsetx,
                                                  offsety=self.config.offsety,
                                                  min_contrast=self.config.min_contrast,
                                                  max_contrast=self.config.max_contrast,
                                                  noise_factor=self.config.noise_factor).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        self.train_loader = torch.utils.data.DataLoader(
            InriaDataset(self.config.img_dir, self.config.lab_dir, self.config.img_size,
                         cls_ids=self.config.cls_id, pad=True),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=16,
            collate_fn=collate_fn,
        )
        self.val_loader = torch.utils.data.DataLoader(
            InriaDataset(self.config.val_img_dir, self.config.val_lab_dir, self.config.img_size,
                         cls_ids=self.config.cls_id, pad=False),
            batch_size=int(self.config.batch_size * 1.5),
            shuffle=True,
            num_workers=16,
            collate_fn=collate_fn)
        self.process = psutil.Process()

    def generate_patch_with_pso(self):
        """
        使用粒子群优化 (PSO) 生成初始对抗补丁。
        """
        cfg = self.config
        dim = 3 * cfg.patch_size * cfg.patch_size  # 补丁参数维度

        # 初始化粒子位置和速度
        positions = np.random.uniform(0, 1, (cfg.num_particles, dim))
        velocities = np.random.uniform(-0.1, 0.1, (cfg.num_particles, dim))

        # 记录粒子个人和全局最优位置
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(cfg.num_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        # 定义适应度函数
        def fitness_func(patch_params):
            patch = torch.tensor(patch_params, dtype=torch.float32).view(3, cfg.patch_size, cfg.patch_size).cuda()
            patch = torch.clamp(patch, 0, 1)  # 限制像素值在 [0, 1]
            attack_effect = self.evaluate_patch(patch)
            stealth_score = self.total_variation(patch)
            return -(attack_effect - 0.1 * stealth_score.item())

        # PSO 主循环
        for iteration in range(cfg.heuristic_max_iter):
            fitness_values = np.array([fitness_func(pos) for pos in positions])

            # 更新个人最佳和全局最佳
            for i in range(cfg.num_particles):
                if fitness_values[i] < personal_best_scores[i]:
                    personal_best_scores[i] = fitness_values[i]
                    personal_best_positions[i] = positions[i]
                if fitness_values[i] < global_best_score:
                    global_best_score = fitness_values[i]
                    global_best_position = positions[i]

            # 更新粒子速度和位置
            r1, r2 = np.random.rand(cfg.num_particles, dim), np.random.rand(cfg.num_particles, dim)
            velocities = (cfg.pso_inertia * velocities +
                          cfg.pso_cognitive * r1 * (personal_best_positions - positions) +
                          cfg.pso_social * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, 0, 1)

        # 返回全局最佳补丁
        best_patch = torch.tensor(global_best_position, dtype=torch.float32).view(3, cfg.patch_size,
                                                                                  cfg.patch_size).cuda()
        best_patch_image = T.ToPILImage()(best_patch.cpu())
        best_patch_image.save("best_patch.png")  # 保存最佳初始化补丁为图片
        return best_patch
    def train(self, start_epoch=0):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        cfg = self.config
        n_epochs = 100
        epoch_length = len(self.train_loader)
        time_str = time.strftime("%Y%m%d-%H%M%S")

        # 使用 PSO 生成初始补丁
        if cfg.use_heuristic_init and cfg.heuristic_method == 'PSO':
            print("使用 PSO 生成初始补丁...")
            adv_patch = self.generate_patch_with_pso()  # 调用 PSO 方法生成补丁
        else:
            print("使用随机初始化生成初始补丁...")
            adv_patch = self.read_image("patches/APPA-yolov5s.png")

        adv_patch.requires_grad_(True)  # 启用梯度计算

        # 初始化优化器
        optimizer = optim.AdamW([adv_patch], lr=cfg.start_learning_rate, weight_decay=1e-4)
        #optimizer = optim.Adam([adv_patch], lr=cfg.start_learning_rate, amsgrad=True)

        # 配置学习率调度器
        scheduler = self.config.scheduler_factory(optimizer)

        wandb.watch_called = False  # Re-run the model without restarting the runtime
        wandb.watch(self.model, log="all")

        et0 = time.time()

        for epoch in range(start_epoch, n_epochs):
            torch.cuda.empty_cache()

            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()

            desc = 'Running epoch %d, loss %f'
            pbar = tqdm(enumerate(self.train_loader),
                        desc=desc % (epoch, 0),
                        total=epoch_length)

            for i_batch, (img_batch, lab_batch, idx_batch, _) in pbar:
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                idx_batch = idx_batch.cuda()

                # Apply patch transformation (augmentations, rotations, etc.)
                adv_batch_t, _, _ = self.patch_transformer(adv_patch, lab_batch, cfg.img_size,
                                                           rand_loc=cfg.rand_loc,
                                                           by_rectangle=cfg.by_rect)
                p_img_batch = self.patch_applier(img_batch, adv_batch_t, idx_batch)
                p_img_batch = F.interpolate(p_img_batch, cfg.imgsz)

                output = self.model(p_img_batch)

                # Extract probabilities, compute NPS and TV losses
                extracted_prob = self.prob_extractor(output)
                nps = self.nps_calculator(adv_patch)
                tv = self.total_variation(adv_patch)

                # 动态调整权重
                if epoch < 5:  # 冷启动：前 5 个 epoch 忽略 NPS 和 TV
                    nps_weight, tv_weight = 0, 0
                else:
                    nps_weight = max(0.01, 0.1 - 0.001 * (epoch - 5))
                    tv_weight = min(2.5, 0.5 + 0.01 * (epoch - 5))

                # Losses
                nps_loss = nps * nps_weight
                tv_loss = tv * tv_weight
                det_loss = torch.mean(extracted_prob)
                loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                ep_det_loss += det_loss.item()
                ep_nps_loss += nps_loss.item()
                ep_tv_loss += tv_loss.item()
                ep_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_description(desc % (epoch, loss))

            et1 = time.time()
            ep_det_loss = ep_det_loss / epoch_length
            ep_nps_loss = ep_nps_loss / epoch_length
            ep_tv_loss = ep_tv_loss / epoch_length
            ep_loss = ep_loss / epoch_length

            scheduler.step(ep_loss)

            torch.cuda.empty_cache()

            with torch.no_grad():
                boxes = preds2boxes(cfg, output)
                train_imgs = post_util.grid_images(
                    [plot_boxes(p_img_batch[i], boxes[i], class_names=cfg.class_names) for i in
                     range(len(p_img_batch))])
                map50, img_preds, patch_img_preds = self.val(adv_patch)

                # Log weights and losses
                wandb.log({
                    "train/patch_img_pred4": wandb.Image(train_imgs),
                    "train/Patches": wandb.Image(adv_patch),
                    "train/tv_loss": ep_tv_loss,
                    "train/nps_loss": ep_nps_loss,
                    "train/det_loss": ep_det_loss,
                    "train/total_loss": ep_loss,
                    "train/time": et1 - et0,
                    "train/nps_weight": nps_weight,  # 动态记录 NPS 权重
                    "train/tv_weight": tv_weight,  # 动态记录 TV 权重
                    'train/step': epoch,
                    'val/map': map50,
                    'val/img_pred4': wandb.Image(img_preds),
                    'val/patch_img_pred4': wandb.Image(patch_img_preds),
                    "val/step": epoch,
                })
        # for epoch in range(start_epoch, n_epochs):
        #     torch.cuda.empty_cache()
        #
        #     ep_det_loss = 0
        #     ep_nps_loss = 0
        #     ep_tv_loss = 0
        #     ep_loss = 0
        #     bt0 = time.time()
        #
        #     desc = 'Running epoch %d, loss %f'
        #     pbar = tqdm(enumerate(self.train_loader),
        #                 desc=desc % (epoch, 0),
        #                 total=epoch_length)
        #
        #     for i_batch, (img_batch, lab_batch, idx_batch, _) in pbar:
        #         img_batch = img_batch.cuda()
        #         lab_batch = lab_batch.cuda()
        #         idx_batch = idx_batch.cuda()
        #
        #         # Apply patch transformation (augmentations, rotations, etc.)
        #         adv_batch_t, _, _ = self.patch_transformer(adv_patch, lab_batch, cfg.img_size,
        #                                                    rand_loc=cfg.rand_loc,
        #                                                    by_rectangle=cfg.by_rect)
        #         p_img_batch = self.patch_applier(img_batch, adv_batch_t, idx_batch)
        #         p_img_batch = F.interpolate(p_img_batch, cfg.imgsz)
        #
        #         output = self.model(p_img_batch)
        #
        #         # Extract probabilities, compute NPS and TV losses
        #         extracted_prob = self.prob_extractor(output)
        #         nps = self.nps_calculator(adv_patch)
        #         tv = self.total_variation(adv_patch)
        #
        #         # Losses
        #         nps_loss = nps * 0.01
        #         tv_loss = tv * 2.5
        #         det_loss = torch.mean(extracted_prob)
        #         loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
        #
        #         ep_det_loss += det_loss.item()
        #         ep_nps_loss += nps_loss.item()
        #         ep_tv_loss += tv_loss.item()
        #         ep_loss += loss.item()
        #
        #         loss.backward()
        #         optimizer.step()
        #         optimizer.zero_grad()
        #         pbar.set_description(desc % (epoch, loss))
        #
        #     et1 = time.time()
        #     ep_det_loss = ep_det_loss / epoch_length
        #     ep_nps_loss = ep_nps_loss / epoch_length
        #     ep_tv_loss = ep_tv_loss / epoch_length
        #     ep_loss = ep_loss / epoch_length
        #
        #     scheduler.step(ep_loss)
        #
        #     torch.cuda.empty_cache()
        #
        #     with torch.no_grad():
        #         boxes = preds2boxes(cfg, output)
        #         train_imgs = post_util.grid_images(
        #             [plot_boxes(p_img_batch[i], boxes[i], class_names=cfg.class_names) for i in
        #              range(len(p_img_batch))])
        #         map50, img_preds, patch_img_preds = self.val(adv_patch)
        #
        #         wandb.log({
        #             "train/patch_img_pred3": wandb.Image(train_imgs),
        #             "train/Patches": wandb.Image(adv_patch),
        #             "train/tv_loss": ep_tv_loss,
        #             "train/nps_loss": ep_nps_loss,
        #             "train/det_loss": ep_det_loss,
        #             "train/total_loss": ep_loss,
        #             "train/time": et1 - et0,
        #             'train/step': epoch,
        #             'val/map': map50,
        #             'val/img_pred3': wandb.Image(img_preds),
        #             'val/patch_img_pred3': wandb.Image(patch_img_preds),
        #             "val/step": epoch,
        #         })
    # def train(self, start_epoch=0):
    #     """
    #     Optimize a patch to generate an adversarial example.
    #     :return: Nothing
    #     """
    #
    #     cfg = self.config
    #     n_epochs =500
    #     epoch_length = len(self.train_loader)
    #     time_str = time.strftime("%Y%m%d-%H%M%S")
    #     # Initialize random patch
    #     adv_patch = self.read_image("patches/APPA-yolov5s.png")
    #     #adv_patch.requires_grad_(True)
    #     # Set up optimizer for the patch
    #     adv_patch.requires_grad_(True)
    #     # 初始化优化器
    #     optimizer = optim.Adam([adv_patch], lr=cfg.start_learning_rate, amsgrad=True)
    #     # 配置学习率调度器
    #     scheduler = self.config.scheduler_factory(optimizer)
    #
    #     wandb.watch_called = False  # Re-run the model without restarting the runtime
    #     wandb.watch(self.model, log="all")
    #     et0 = time.time()
    #     for epoch in range(start_epoch, n_epochs):
    #         torch.cuda.empty_cache()
    #
    #         ep_det_loss = 0
    #         ep_nps_loss = 0
    #         ep_tv_loss = 0
    #         ep_loss = 0
    #         bt0 = time.time()
    #         desc = 'Running epoch %d, loss %f'
    #         pbar = tqdm(enumerate(self.train_loader),
    #                     desc=desc % (epoch, 0),
    #                     total=epoch_length)
    #         for i_batch, (img_batch, lab_batch, idx_batch, _) in pbar:
    #             img_batch = img_batch.cuda()
    #             lab_batch = lab_batch.cuda()
    #             idx_batch = idx_batch.cuda()
    #
    #             # Generate random patch
    #             # adv_patch = self.generate_random_patch()
    #             # Apply patch transformation (augmentations, rotations, etc.)
    #             adv_batch_t, _, _ = self.patch_transformer(adv_patch, lab_batch, cfg.img_size,
    #                                                        rand_loc=cfg.rand_loc,
    #                                                        by_rectangle=cfg.by_rect)
    #             p_img_batch = self.patch_applier(img_batch, adv_batch_t, idx_batch)
    #             p_img_batch = F.interpolate(p_img_batch, cfg.imgsz)
    #             img = p_img_batch
    #             single_image = img[0].detach().cpu()  # 去除梯度，移到 CPU
    #
    #             # 转换为 NumPy 数组，并调整维度为 [H, W, C] (注意通道顺序)
    #             image_np = single_image.permute(1, 2, 0).numpy()
    #
    #             # 将图像的值范围从 [0, 1] 映射到 [0, 255]，并确保数据类型是 uint8
    #             image_np = (image_np * 255).astype(np.uint8)
    #
    #             # 将 NumPy 数组转换为 PIL 图像
    #             image_pil = Image.fromarray(image_np)
    #
    #             # 保存图像
    #             image_pil.save("AAA.jpg")
    #
    #             output = self.model(p_img_batch)
    #
    #             # Extract probabilities, compute NPS and TV losses
    #             extracted_prob = self.prob_extractor(output)
    #             nps = self.nps_calculator(adv_patch)
    #             tv = self.total_variation(adv_patch)
    #
    #             # Losses
    #             nps_loss = nps * 0.01
    #             tv_loss = tv * 2.5
    #             det_loss = torch.mean(extracted_prob)
    #             loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
    #
    #             ep_det_loss += det_loss.item()
    #             ep_nps_loss += nps_loss.item()
    #             ep_tv_loss += tv_loss.item()
    #             ep_loss += loss.item()
    #
    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             pbar.set_description(desc % (epoch, loss))
    #
    #         et1 = time.time()
    #         ep_det_loss = ep_det_loss / epoch_length
    #         ep_nps_loss = ep_nps_loss / epoch_length
    #         ep_tv_loss = ep_tv_loss / epoch_length
    #         ep_loss = ep_loss / epoch_length
    #
    #         scheduler.step(ep_loss)
    #
    #         et0 = time.time()
    #
    #         torch.cuda.empty_cache()
    #
    #         with torch.no_grad():
    #             boxes = preds2boxes(cfg, output)
    #             train_imgs = post_util.grid_images(
    #                 [plot_boxes(p_img_batch[i], boxes[i], class_names=cfg.class_names) for i in
    #                  range(len(p_img_batch))])
    #             map50, img_preds, patch_img_preds = self.val(adv_patch)
    #             wandb.log({
    #                 "train/patch_img_pred3": wandb.Image(train_imgs),
    #                 "train/Patches": wandb.Image(adv_patch),
    #                 "train/tv_loss": ep_tv_loss,
    #                 "train/nps_loss": ep_nps_loss,
    #                 "train/det_loss": ep_det_loss,
    #                 "train/total_loss": ep_loss,
    #                 "train/time": et1 - et0,
    #                 'train/step': epoch,
    #                 'val/map': map50,
    #                 'val/img_pred3': wandb.Image(img_preds),
    #                 'val/patch_img_pred3': wandb.Image(patch_img_preds),
    #                 "val/step": epoch,
    #             })

    def val(self, adv_patch, cal_map=True):
        cfg = self.config
        gt_path = os.path.join('temp', 'gt')
        dr_path = os.path.join('temp', 'dr')
        if os.path.exists(gt_path):
            shutil.rmtree(gt_path)
        if os.path.exists(dr_path):
            shutil.rmtree(dr_path)
        os.makedirs(dr_path)
        os.makedirs(gt_path)

        bn = len(self.val_loader)
        bidx = np.random.randint(0, bn)
        cnt = 0

        for bi, (img_batch, lab_batch, idx_batch, paths) in tqdm(enumerate(self.val_loader),
                                                                 desc="Validation: ",
                                                                 total=len(self.val_loader)):
            img_batch = img_batch.cuda()
            lab_batch = lab_batch.cuda()
            idx_batch = idx_batch.cuda()

            with torch.no_grad():
                output = self.model(F.interpolate(img_batch, cfg.imgsz))
                output = preds2boxes(cfg, output)
                adv_batch_t, _, _ = self.patch_transformer(adv_patch, lab_batch, cfg.img_size,
                                                           do_blur=False,
                                                           do_rotate=False,
                                                           rand_loc=False,
                                                           do_aug=False,
                                                           by_rectangle=cfg.by_rect)
                p_img_batch = self.patch_applier(img_batch.clone(), adv_batch_t, idx_batch)
                patch_output = self.model(F.interpolate(p_img_batch, cfg.imgsz))
                patch_output = preds2boxes(cfg, patch_output)

            img_preds = post_util.grid_images(
                [plot_boxes(img_batch[i], output[i], class_names=cfg.class_names) for i in
                 range(len(img_batch))])
            patch_img_preds = post_util.grid_images(
                [plot_boxes(p_img_batch[i], patch_output[i], class_names=cfg.class_names) for i in
                 range(len(p_img_batch))])
            img_preds.save(f'pred4/{cnt}.jpg')
            patch_img_preds.save(f'pred4/patch_{cnt}.jpg')
            cnt += 1
            if not cal_map:
                continue
            for bi, (img_batch, lab_batch, idx_batch, paths) in tqdm(enumerate(self.val_loader),
                                                                     desc="Validation: ",
                                                                     total=len(self.val_loader)):
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                idx_batch = idx_batch.cuda()

                with torch.no_grad():
                    output = self.model(F.interpolate(img_batch, cfg.imgsz))
                    output = preds2boxes(cfg, output)
                    adv_batch_t, _, _ = self.patch_transformer(adv_patch, lab_batch, cfg.img_size,
                                                               do_blur=False,
                                                               do_rotate=False,
                                                               rand_loc=False,
                                                               do_aug=False,
                                                               by_rectangle=cfg.by_rect)
                    p_img_batch = self.patch_applier(img_batch.clone(), adv_batch_t, idx_batch)
                    patch_output = self.model(F.interpolate(p_img_batch, cfg.imgsz))
                    patch_output = preds2boxes(cfg, patch_output)

                # if bi == bidx:
                if True:
                    img_preds = post_util.grid_images(
                        [plot_boxes(img_batch[i], output[i], class_names=cfg.class_names) for i in
                         range(len(img_batch))])
                    patch_img_preds = post_util.grid_images(
                        [plot_boxes(p_img_batch[i], patch_output[i], class_names=cfg.class_names) for i in
                         range(len(p_img_batch))])
                    # save_imgs
                    img_preds.save(f'pred4/{cnt}.jpg')
                    patch_img_preds.save(f'pred4/patch_{cnt}.jpg')
                    cnt += 1
                if not cal_map:
                    continue
                for i, (gt_pred, dr_pred, img_path) in enumerate(zip(output, patch_output, paths[0])):
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    gt_box = gt_pred[:, :4]
                    dr_box = dr_pred[:, :4]
                    dr_confs = dr_pred[:, 4]
                    gt_box = gt_box.cpu().numpy()
                    dr_box = dr_box.cpu().numpy()
                    gt_box = xywh2xyxy(gt_box)
                    dr_box = xywh2xyxy(dr_box)
                    gt_box = np.round(gt_box * cfg.img_size)
                    dr_box = np.round(dr_box * cfg.img_size)
                    with open(os.path.join(gt_path, f'{img_name}.txt'), 'w') as f:
                        for i, box in enumerate(gt_box):
                            coords = ' '.join(map(str, box))
                            f.write(f'{int(gt_pred[i, -1].item())} {coords}\n')
                        f.close()
                    with open(os.path.join(dr_path, f'{img_name}.txt'), 'w') as f:
                        for i, (box, dr_conf) in enumerate(zip(dr_box, dr_confs)):
                            coords = ' '.join(map(str, box))
                            f.write(f'{int(dr_pred[i, -1].item())} {dr_conf} {coords}' + '\n')
                        f.close()
            map50 = map_cal.count(path_ground_truth=gt_path,
                                  path_detection_results=dr_path) if cal_map else None
            return map50, img_preds, patch_img_preds
    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch
    def read_image(self, path):
        patch_img = Image.open(path).convert('RGB')
        transforms = T.Compose([T.Resize((self.config.patch_size, self.config.patch_size)), T.ToTensor()])
        return transforms(patch_img).cuda()

    def evaluate_patch(self, patch):
        """
        评估补丁的攻击效果。
        """
        cfg = self.config  # 获取配置信息

        with torch.no_grad():
            # 遍历数据加载器中的训练数据
            for img_batch, lab_batch, idx_batch, _ in self.train_loader:
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                idx_batch = idx_batch.cuda()

                # 应用补丁增强
                adv_batch_t, _, _ = self.patch_transformer(
                    patch, lab_batch, cfg.img_size,
                    rand_loc=cfg.rand_loc,
                    by_rectangle=cfg.by_rect
                )
                # 将补丁应用到图像上
                p_img_batch = self.patch_applier(img_batch, adv_batch_t, idx_batch)

                # 模型预测
                output = self.model(F.interpolate(p_img_batch, cfg.imgsz))
                # 提取目标类别的概率
                extracted_prob = self.prob_extractor(output)
                return extracted_prob.mean().item()  # 返回平均置信度

def main():

    cfg = patch_config_4.patch_configs['yolov3_dota']()
    # cfg.batch_size = 1
    trainer = PatchTrainer(cfg)
    wandb.init(project="Adversarial-attack", config=dict(
        name=trainer.config.patch_name,
        batch_size=trainer.config.batch_size,
        scale=trainer.config.scale,
        lab_dir=trainer.config.lab_dir,
        seed=trainer.config.seed,
        lr=trainer.config.start_learning_rate,
        cls_id=trainer.config.cls_id,
        patch_size=trainer.config.patch_size,
        img_size=trainer.config.img_size,
        imgsz=trainer.config.imgsz,
        conf_thres=trainer.config.img_size,
        iou_thres=trainer.config.img_size,
        max_det=trainer.config.max_det,
        guidance_scale=trainer.config.guidance_scale,
        init_num_inference_steps=trainer.config.init_num_inference_steps,
        num_inference_steps=trainer.config.num_inference_steps,
        start_time_step=trainer.config.start_time_step,
        end_time_step=trainer.config.end_time_step,
        do_classifier_free_guidance=trainer.config.do_classifier_free_guidance,
    ))
    trainer.train()
    wandb.finish()

if __name__ == '__main__':
    main()