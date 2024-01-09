# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import time
import paddle
import paddle.nn.functional as F

from paddleseg.utils import TimeAverager
from common import Metrics, save_numpy_as_csv

np.set_printoptions(suppress=True)


def evaluate(model, eval_dataset, args=None):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A semantic segmentation model.
        eval_dataset (paddle.io.woyo): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        precision (str, optional): Use AMP if precision='fp16'. If precision='fp32', the evaluation is normal.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        auc_roc(bool, optional): whether add auc_roc metric

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
    # batch_sampler = paddle.io.BatchSampler(
    #     eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    #
    # loader = paddle.io.DataLoader(
    #     eval_dataset,
    #     batch_sampler=batch_sampler,
    #     num_workers=num_workers,
    #     return_list=True)

    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    evaluator = Metrics(num_class=args.num_classes)

    with paddle.no_grad():
        for _, data in enumerate(eval_dataset):
            reader_cost_averager.record(time.time() - batch_start)

            label = data['label'].astype('int64').cuda()
            
            if args.img_ab_concat:
                images = data['img'].cuda()
                pred = model(images)
                
            else:
                img1 = data['img1'].cuda()
                img2 = data['img1'].cuda()
                pred = model(img1, img2)

            
            if hasattr(model, "predict"):
                pred = model.predict(pred)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[args.pred_idx]

            evaluator.add_batch(pred, label)

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    evaluator.calc()
    miou = evaluator.Mean_Intersection_over_Union()
    acc = evaluator.Pixel_Accuracy()
    class_iou = evaluator.Intersection_over_Union()
    class_precision = evaluator.Class_Precision()
    kappa = evaluator.Kappa()
    recall = evaluator.Mean_Recall()
    macro_f1 = evaluator.Macro_F1()
    class_recall = evaluator.Recall()
    class_dice = evaluator.Dice()
    # print(batch_cost, reader_cost)
    if args.logger != None:
        infor = "[EVAL] Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(len(eval_dataset), batch_cost, reader_cost)
        args.logger.info(infor)
        # infor = "[EVAL] mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} mdice: {:.4f} Macro_F1: {:.4f}".format(
        #      miou, acc, kappa, mdice, macro_f1)
        # args.logger.info(infor)
        args.logger.info("[METRICS] Acc:{:.4},mIoU:{:.4}, recall:{:.4},kappa:{:.4},Macro_f1:{:.4}".format(
            acc,miou,recall,kappa,macro_f1))
        
        args.logger.info("[METRICS] Class IoU: " + str(np.round(class_iou, 4)))
        args.logger.info("[METRICS] Class Precision: " + str(np.round(class_precision, 4)))
        args.logger.info("[METRICS] Class Recall: " + str(np.round(class_recall, 4)))
        args.logger.info("[METRICS] Class Dice: " + str(np.round(class_dice, 4)))
    save_numpy_as_csv(args.metric_path, np.array([args.epoch, args.loss, miou, acc, kappa, recall, macro_f1]))
    return miou
