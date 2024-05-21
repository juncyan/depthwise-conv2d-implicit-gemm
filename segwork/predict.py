import os

import cv2
import numpy as np
import time
import paddle
import datetime

from paddleseg.utils import TimeAverager, op_flops_funs
from common import Metrics
from common.logger import load_logger
from work.count_params import flops


def predict(model, dataset, weight_path=None, data_name="test", num_classes=2, save_dir=r"/mnt/data/Results/"):

    if weight_path:
        layer_state_dict = paddle.load(f"{weight_path}")
        model.set_state_dict(layer_state_dict)
    else:
        exit()

    img_ab_concat = dataset.en_concat

    time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")
    model_name = model.__str__().split("(")[0]

    img_dir = f"{save_dir}/{data_name}/{model_name}_{time_flag}"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    color_label = dataset.label_info.values
    color_label = np.transpose(color_label)

    logger = load_logger(f"{img_dir}/prediction.log")
    logger.info(f"test {model_name} on {data_name}")
    
    batch_sampler = paddle.io.BatchSampler(
        dataset, batch_size=8, shuffle=False, drop_last=False)
    
    loader = paddle.io.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        return_list=True)

    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    evaluator = Metrics(num_class=num_classes)
    with paddle.no_grad():
        for _, data in enumerate(loader):

            reader_cost_averager.record(time.time() - batch_start)

            name = data['name']
            label = data['label'].astype('int64')

            images = data['img'].cuda()
            pred = model(images)
            
            if hasattr(model, "predict"):
                pred = model.predict(pred)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[0]

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()


            pred = paddle.argmax(pred, axis=1)
            pred = pred.squeeze().cpu()

            evaluator.add_batch(pred, label)

            for idx, ipred in enumerate(pred):
                ipred = ipred.numpy()
                # print(np.array([[0,0,0],[255,255,255]]).shape)
                # print(color_label.shape)
                if (np.max(ipred) != np.min(ipred)):
                    img = color_label[ipred]
                    cv2.imwrite(f"{img_dir}/{name[idx]}", img)

    evaluator.calc()
    miou = evaluator.Mean_Intersection_over_Union()
    acc = evaluator.Pixel_Accuracy()
    class_iou = evaluator.Intersection_over_Union()
    class_precision = evaluator.Class_Precision()
    kappa = evaluator.Kappa()
    m_dice = evaluator.Mean_Dice()
    f1 = evaluator.F1_score()
    macro_f1 = evaluator.Macro_F1()
    class_recall = evaluator.Recall()

    infor = "[PREDICT] #Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(len(dataset), batch_cost, reader_cost)
    logger.info(infor)
    infor = "[METRICS] mIoU: {:.4f}, Acc: {:.4f}, Kappa: {:.4f}, mDice: {:.4f}, Macro_F1: {:.4f}".format(
            miou, acc, kappa, m_dice, macro_f1)
    logger.info(infor)

    logger.info("[METRICS] Class IoU: " + str(np.round(class_iou, 4)))
    logger.info("[METRICS] Class Precision: " + str(np.round(class_precision, 4)))
    logger.info("[METRICS] Class Recall: " + str(np.round(class_recall, 4)))
    logger.info("[METRICS] Class F1: " + str(np.round(f1, 4)))

    if img_ab_concat:
        images = data['img'].cuda()
        _, c, h, w = images.shape
        flop_p = flops(
        model, [1, c, h, w],
        custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})
            
    else:
        img1 = data['img1'].cuda()
        _, c, h, w = img1.shape
        flop_p = flops(
        model, [1, c, h, w], 2,
        custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})
    logger.info(r"[PREDICT] model total flops is: {}, params is {}".format(flop_p["total_ops"],flop_p["total_params"]))       


def test(model, dataset, args):
 
    if args.best_model_path:
        layer_state_dict = paddle.load(f"{args.best_model_path}")
        model.set_state_dict(layer_state_dict)
    else:
        exit()

    test_batch_sampler = paddle.io.BatchSampler(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    test_data_loader = paddle.io.DataLoader(
        dataset,
        batch_sampler=test_batch_sampler,
        num_workers=0,
        return_list=True)

    time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")
    img_dir = f"/mnt/data/Results/{args.data_name}/{args.model_name}_{time_flag}"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    color_label = dataset.label_info.values # np.array([[0,0,0],[255,255,255]])
    color_label = np.transpose(color_label)

    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    evaluator = Metrics(num_class=args.num_classes)
    with paddle.no_grad():
        for _, data in enumerate(test_data_loader):

            reader_cost_averager.record(time.time() - batch_start)

            name = data['name']
            label = data['label'].astype('int64')

            images = data['img'].cuda()
            pred = model(images)
                
            if hasattr(model, "predict"):
                pred = model.predict(pred)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[args.pred_idx]

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

            pred = paddle.argmax(pred, axis=1)
            pred = pred.squeeze().cpu()
            evaluator.add_batch(pred, label)

            for idx, ipred in enumerate(pred):
                ipred = ipred.numpy()
                if (np.max(ipred) != np.min(ipred)):
                    img = color_label[ipred]
                    cv2.imwrite(f"{img_dir}/{name[idx]}", img)

    evaluator.calc()
    miou = evaluator.Mean_Intersection_over_Union()
    acc = evaluator.Pixel_Accuracy()
    class_iou = evaluator.Intersection_over_Union()
    class_precision = evaluator.Class_Precision()
    kappa = evaluator.Kappa()
    m_dice = evaluator.Mean_Dice()
    f1 = evaluator.F1_score()
    macro_f1 = evaluator.Macro_F1()
    class_recall = evaluator.Recall()

    infor = "[PREDICT] #Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(len(test_data_loader), batch_cost, reader_cost)
    args.logger.info(infor)
    infor = "[METRICS] mIoU: {:.4f}, Acc: {:.4f}, Kappa: {:.4f}, mDice: {:.4f}, Macro_F1: {:.4f}".format(
            miou, acc, kappa, m_dice, macro_f1)
    args.logger.info(infor)

    args.logger.info("[METRICS] Class IoU: " + str(np.round(class_iou, 4)))
    args.logger.info("[METRICS] Class Precision: " + str(np.round(class_precision, 4)))
    args.logger.info("[METRICS] Class Recall: " + str(np.round(class_recall, 4)))
    args.logger.info("[METRICS] Class F1: " + str(np.round(f1, 4)))
    
    img1 = data['img'].cuda()
    _, c, h, w = img1.shape
    flop_p = flops(
    model, [1, c, h, w], 2,
    custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})
    args.logger.info(r"[PREDICT] model total flops is: {}, params is {}".format(flop_p["total_ops"],flop_p["total_params"]))       


