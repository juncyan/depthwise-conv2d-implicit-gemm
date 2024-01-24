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
import time
from collections import deque
import shutil
from copy import deepcopy

import numpy as np
import paddle
import paddle.nn.functional as F

from work.count_params import flops
from paddleseg.utils import worker_init_fn, op_flops_funs
from work.val import evaluate
from work.predict import test, test_last


def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def Loss(logits,labels):
    # return paddle.nn.BCEWithLogitsLoss()(logits, labels)

    if logits.shape == labels.shape:
        labels = paddle.argmax(labels,axis=1)
    elif len(labels.shape) == 3:
        labels = labels
    else:
        assert "pred.shape not match label.shape"
    #logits = F.softmax(logits,dim=1)
    return paddle.nn.CrossEntropyLoss(axis=1)(logits,labels)

def train(model,
          train_dataset,
          val_dataset=None,
          test_data=None,
          optimizer=None,
          args=None,
          iters=10000,
          save_interval=1000,
          num_workers=0):
    """
    Launch training.

    Args:
        model（nn.Layer): A semantic segmentation model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
        optimizer (paddle.optimizer.Optimizer): The optimizer.
    """

    model = model.to('gpu')

    model.train()
    # if not os.path.isdir(args.save_dir):
    #     if os.path.exists(args.save_dir):
    #         os.remove(args.save_dir)
    #     os.makedirs(args.save_dir, exist_ok=True)

    # use amp
    batch_sampler = paddle.io.BatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
        worker_init_fn=worker_init_fn, )

    val_batch_sampler = paddle.io.BatchSampler(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    val_loader = paddle.io.DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        num_workers=num_workers,
        return_list=True)

    test_batch_sampler = paddle.io.BatchSampler(
        test_data, batch_size=args.batch_size, shuffle=False, drop_last=False)

    test_data_loader = paddle.io.DataLoader(
        test_data,
        batch_sampler=test_batch_sampler,
        num_workers=0,
        return_list=True)

    if args.logger != None:
        args.logger.info("start train")

    best_mean_iou = -1.0
    best_model_iter = -1

    batch_start = time.time()

    for _epoch in range(iters):
        avg_loss_list = []
        epoch = _epoch + 1
        model.train()

        for data in loader:
            labels = data['label'].astype('int64').cuda()
            # l_split = (z.chunk(2, axis=2) for z in labels.chunk(2, axis=3))
            # l_merge = paddle.concat(tuple(paddle.concat((x2, x1), 2) for (x1, x2) in l_split), 3)
            # labels = paddle.concat([labels, l_merge], 0)

            if args.img_ab_concat:
                images = data['img'].cuda()
                # i_split = (z.chunk(2, axis=2) for z in images.chunk(2, axis=3))
                # i_merge = paddle.concat(tuple(paddle.concat((x2, x1), 2) for (x1, x2) in i_split), 3)
                # images = paddle.concat([images, i_merge], 0)
                # print(images.shape)
                pred = model(images)
                
            else:
                img1 = data['img1'].cuda()
                img2 = data['img1'].cuda()
                pred = model(img1, img2)
            
            if hasattr(model, "loss"):
                loss_list = model.loss(pred, labels)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[args.pred_idx]
                # print(pred.shape, labels.shape)
                loss_list = Loss(pred, labels)

            loss = loss_list#sum(loss_list)
            
            loss.backward()
            optimizer.step()
            
            lr = optimizer.get_lr()

            # update lr
            lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                if isinstance(lr_sche, paddle.optimizer.lr.ReduceOnPlateau):
                    lr_sche.step(loss)
                else:
                    lr_sche.step()

            model.clear_gradients()
            
            # 
            avg_loss = np.array(loss.cpu())
            avg_loss_list.append(avg_loss)
            # for i in range(len(loss_list)):
            #     avg_loss_list[i] += loss_list[i].numpy()
        batch_cost_averager = time.time() - batch_start
        avg_loss = np.mean(avg_loss_list)

        if args.logger != None:
            args.logger.info(
                "[TRAIN] iter: {}/{}, loss: {:.4f}, lr: {:.6}, batch_cost: {:.2f}, ips: {:.4f} samples/sec".format(
                    epoch, iters, avg_loss, lr, batch_cost_averager, batch_cost_averager / len(train_dataset)))

        if epoch % save_interval == 0 or epoch == iters:
            paddle.save(model.state_dict(),
                        os.path.join(args.save_dir, f'epoch_{epoch}_model.pdparams'))
            
        # if (epoch) % save_interval == 0:

        args.epoch = epoch
        args.loss = avg_loss

        mean_iou = evaluate(model,val_loader,args = args)

        if mean_iou > best_mean_iou:
            # predict(model, test_data_loader, args)
            best_mean_iou = mean_iou
            best_model_iter = epoch
            paddle.save(model.state_dict(), args.best_model_path)

        if args.logger !=  None:
            # args.logger.info(
            #     '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
            #     .format(best_mean_iou, best_model_iter))
            args.logger.info("[TRAIN] best iter {}, max mIoU {:.4f}".format(best_model_iter, best_mean_iou))
        batch_start = time.time()
        # if (epoch >=200) and (epoch - best_model_iter >= 50):
        #     break 
    # Calculate flops.
    # if not "precision" == 'fp16':
    test(model, test_data_loader, args)
    lsp = os.path.join(args.save_dir, f'epoch_{iters}_model.pdparams')
    test_last(model, test_data_loader, args, lsp)
    
    # Sleep for a second to let dataloader release resources.
    time.sleep(1)
