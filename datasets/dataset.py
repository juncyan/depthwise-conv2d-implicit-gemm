import copy
import os
import random

import numpy as np
from collections.abc import Sequence
from pycocotools.coco import COCO

from paddle.io import Dataset


class COCODataSet(Dataset):
    """
    Load dataset with COCO format.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): coco annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth.
            False as default
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total
            record's, if empty_ratio is out of [0. ,1.), do not sample the
            records and use all the empty entries. 1. as default
        use_default_label (bool): whether to load default label list.
    """
    def __init__(self, dataset_dir = None, image_dir = None, annotations_path = None,
                 data_fields = ['image'], sample_num = -1, load_crowed = False,
                 allow_empty = False, empty_ratio = 1.0, use_default_label = None):
        super(COCODataSet, self).__init__()
        self.dataset_dir = dataset_dir
        self.anno_path = annotations_path
        self.img_dir = image_dir
        self.data_fields = data_fields
        self.sample_num = sample_num
        self.load_crowed = load_crowed
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio
        self.use_default_label = use_default_label
        self.load_img_only = False
        self.load_semantic = False
        self._epoch = 0
        self._curr_iter = 0
        self.roidbs = None
        self.mixup_epoch = 0
        self.cutmix_epoch = 0
        self.mosaic_epoch = 0
        self.transform = None

    def __len__(self):
        return len(self.roidbs)

    def __getitem__(self, idx):
        roidb = copy.deepcopy(self.roidbs[idx])
        if self.mixup_epoch == 0 or self._epoch < self.mixup_epoch:
            n = len(self.roidbs)
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self .cutmix_epoch == 0 or self._epoch < self.cutmix_epoch:
            n = len(self.roidbs)
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.mosaic_epoch == 0 or self._epoch < self.mosaic_epoch:
            n = len(self.roidbs)
            roidb = [roidb, ] + [
                copy.deepcopy(self.roidbs[np.random.randint(n)])
                for _ in range(3)
            ]

        if isinstance(roidb, Sequence):
            for r in roidb:
                r['curr_iter'] = self._curr_iter
        else:
            roidb['curr_iter'] = self._curr_iter
        self._curr_iter += 1

        return self.transform(roidb)

    def _sample_empty(self, records, num):
        # if empty_ratio is out of [0, 1), do not sample the records
        if self.empty_ratio < 0. or self.empty_ratio >= 1:
            return records
        sample_num = min(int(num * self.empty_ratio / (1 - self.empty_ratio)),
                         len(records))
        records = random.sample(records, sample_num)

        return records

    def parse_dataset(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        img_dir = os.path.join(self.dataset_dir, self.img_dir)

        assert anno_path.endswith('.json'), \
        'invalid coco annotation file: ' + anno_path
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []
        empty_records = []
        ct = 0

        self.catid2clsid = dict({catid : i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name'] : clsid
            for catid, clsid in self.catid2clsid.items()})

        if 'annotations' not in coco.dataset:
            self.load_img_only = True
            print('Annotation file: {} does not contains ground truth '
                  'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])
            im_path  = os.path.join(img_dir, im_fname) if img_dir else im_fname
            is_empty = False

            if not os.path.exists(im_path):
                print('illegal width: {} or height: {} in annotation,'
                      'and im_id: {} will be ignored'.format(im_w, im_h, img_id))
                continue
            coco_rec = {
                'im_file' : im_path,
                'im_id' : np.array([img_id]),
                'h' : im_h,
                'w' : im_w
            } if 'image' in self.data_fields else {}

            if not self.load_img_only:
                ins_anno_ids = coco.getAnnIds(
                    imgIds = [img_id], iscrowd = None if self.load_crowed else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                is_rbox_anno = False
                for inst in instances:
                    #check gt bbox
                    if inst.get('ignore', False):
                        continue
                    if 'bbox' not in inst.keys():
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            continue

                    is_rbox_anno = True if len(inst['bbox']) == 5 else False
                    xc, yc, box_w, box_h, angle = 0., 0., 0., 0., 0.
                    if is_rbox_anno:
                        xc, yc, box_w, box_h, angle = inst['bbox']
                        x1 = xc - box_w / 2.0
                        y1 = yc - box_h / 2.0
                    else:
                        x1, y1, box_w, box_y = inst['bbox']

                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 1e-5

                    if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                        inst['clean_bbox'] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]]
                        if is_rbox_anno:
                            inst['clean_rbox'] = [xc, yc, box_w, box_h, angle]
                        bboxes.append(inst)
                    else:
                        print("Found an invalid bbox in annotations: im_id: {}, "
                              "area: {} x1: {}, y1: {}, x2: {}, y2: {}.".format(
                            img_id, float(inst['area']), x1, y1, x2, y2))

                num_bbox = len(bboxes)
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype = np.float32)
                gt_rbox = None
                if is_rbox_anno:
                    gt_rbox = np.zeros((num_bbox, 5), dtype = np.float32)
                gt_theta = np.zeros((num_bbox, 1), dtype = np.int32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox

                has_segmentation = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    if is_rbox_anno:
                        gt_rbox[i, :] = box['clean_rbox']
                    is_crowd[i][0] = box['iscrowd']

                    if 'segmentation' in box and box['iscrowd'] == 1:
                        gt_poly[i] = [[0., 0., 0., 0., 0., 0.]]
                    elif 'segmentation' in box and box['segmentation']:
                        if not np.array(box['segmentation']).size > 0 and not self.allow_empty:
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box['segmentation']
                        has_segmentation = True

                if has_segmentation and not any(gt_poly) and not self.allow_empty:
                    continue

                if is_rbox_anno:
                    gt_rec = {
                        'is_crowd': is_crowd,
                        'gt_class': gt_class,
                        'gt_bbox': gt_bbox,
                        'gt_rbox': gt_rbox,
                        'gt_poly': gt_poly
                    }
                else:
                    gt_rec = {
                        'is_crowd': is_crowd,
                        'gt_class': gt_class,
                        'gt_bbox': gt_bbox,
                        'gt_poly': gt_poly
                    }

                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

                if self.load_semantic and 'semantic' in self.data_fields:
                    seg_path = os.path.join(self.dataset_dir, 'stuffthingmaps',
                                            'train2017', im_fname[:-3] + 'png')
                    coco_rec.update({'semantic':seg_path})

            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s'%(anno_path)
        if len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records

    def set_kwargs(self, **kwargs):
        self.mixup_epoch = kwargs.get('mixup_epoch', -1)
        self.cutmix_epoch = kwargs.get('cutmix_epoch', -1)
        self.mosaic_epoch = kwargs.get('mosaic_epoch', -1)

    def set_transform(self, transform):
        self.transform = transform

    def set_epoch(self, epoch_id):
        self._epoch = epoch_id

    def get_anno(self):
        if self.anno_path is None:
            return
        return os.path.join(self.dataset_dir, self.anno_path)


if __name__ == "__main__":
    print('dataset')





