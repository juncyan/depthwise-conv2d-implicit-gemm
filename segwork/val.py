import os

import numpy as np
import time
import paddle
import paddle.nn.functional as F
import paddle.tensor
import pandas as pd

from paddleseg.utils import TimeAverager
from common import Metrics, save_numpy_as_csv

np.set_printoptions(suppress=True)


def evaluate(model, eval_dataset, args=None):
    model.eval()

    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    evaluator = Metrics(num_class=args.num_classes)

    with paddle.no_grad():
        for _, data in enumerate(eval_dataset):
            reader_cost_averager.record(time.time() - batch_start)

            label = data['label'].astype('int64').cuda()
            images = data['img'].cuda()
            pred = model(images)
                
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

    metrics = evaluator.Get_Metric()
    pa = metrics["pa"]
    miou = metrics["miou"]
    mf1 = metrics["mf1"]
    kappa = metrics["kappa"]

    if args.logger != None:
        infor = "[EVAL] Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(len(eval_dataset), batch_cost, reader_cost)
        args.logger.info(infor)
        args.logger.info("[METRICS] PA:{:.4},mIoU:{:.4},kappa:{:.4},Macro_f1:{:.4}".format(pa,miou,kappa,mf1))
        
    d = pd.DataFrame([metrics])
    if os.path.exists(args.metric_path):
        d.to_csv(args.metric_path,mode='a', index=False, header=False,float_format="%.4f")
    else:
        d.to_csv(args.metric_path, index=False,float_format="%.4f")
    return miou
