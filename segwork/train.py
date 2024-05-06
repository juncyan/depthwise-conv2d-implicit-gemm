import os
import time

import numpy as np
import paddle
from paddleseg.utils import worker_init_fn
from segwork.val import evaluate
from segwork.predict import test

def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def Loss(logits,labels):
   
    if logits.shape == labels.shape:
        labels = paddle.argmax(labels,axis=1)
    elif len(labels.shape) == 3:
        labels = labels
    else:
        assert "pred.shape not match label.shape"
    return paddle.nn.CrossEntropyLoss(axis=1)(logits,labels)

def train(model,
          train_dataset,
          val_dataset=None,
          test_data=None,
          optimizer=None,
          args=None,
          iters=10000,
          num_workers=0):

    model = model.to(args.device)

    model.train()
  
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
            
            images = data['img'].cuda()
            pred = model(images)
                
            if hasattr(model, "loss"):
                loss_list = model.loss(pred, labels)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[args.pred_idx]
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
            

            avg_loss = np.array(loss.cpu())
            avg_loss_list.append(avg_loss)

        batch_cost_averager = time.time() - batch_start
        avg_loss = np.mean(avg_loss_list)

        if args.logger != None:
            args.logger.info(
                "[TRAIN] iter: {}/{}, loss: {:.4f}, lr: {:.6}, batch_cost: {:.2f}, ips: {:.4f} samples/sec".format(
                    epoch, iters, avg_loss, lr, batch_cost_averager, batch_cost_averager / len(train_dataset)))


        args.epoch = epoch
        args.loss = avg_loss

        mean_iou = evaluate(model,val_loader,args = args)

        if epoch == iters:
            paddle.save(model.state_dict(),
                        os.path.join(args.save_dir, 'last_epoch_model.pdparams'))

        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_model_iter = epoch
            paddle.save(model.state_dict(),args.best_model_path)

        if args.logger !=  None:
            args.logger.info("[TRAIN] best iter {}, max mIoU {:.4f}".format(best_model_iter, best_mean_iou))
        batch_start = time.time()
      
    test(model, test_data, args)
    

