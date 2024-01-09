import copy
import random
import uuid
import cv2
import paddle
import shapely
import functools

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import numpy as np
from collections.abc import Sequence
from logger import logger
import pycocotools.mask as mask_util

from compose import Compose


class BaseOperator():
    def __int__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def apply(self, sample, context=None):
        '''
        :param sample(dict): {'image':xx, 'label': yy}
        :param context: a dict, info about this sample processing
        :return: a dict ,a processed sample
        '''
        #if context is None:
        return sample

    def __call__(self, sample, context =None):
        '''
        :param sample(dict): {'image':xx, 'label': yy}
        :param context: a dict, info about this sample processing
        :return: a dict ,a processed sample
        '''
        if isinstance(sample, Sequence):
            for i in range(len(sample)):
                sample[i] = self.apply(sample[i], context)
        else:
            sample = self.apply(sample, context)

        return sample

    def is_poly(self, segm):
        return isinstance(segm, np.ndarray) and segm.shape >= 2

    def __str__(self):
        return str(self._id)

class Decode(BaseOperator):
    def __init__(self):
        #Transform image data to numpy format following the rgb format
        super(Decode, self).__init__()

    def apply(self, sample, context=None):
        #load image if im_file field is not empty but image is empty
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()
            sample.pop('im_file')

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)
        if 'keep_ori_im' in sample and sample['keep_ori_im']:
            sample['ori_image'] = im
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample['image'] = im
        if 'h' not in sample:
            sample['h'] = im.shape[0]
        elif sample['h'] != im.shape[0]:
            logger.warning(
                'The actual image height: {} is not equal to the height :{}'
                ' in annotation, and update sample[\'h\'] by actual image'
                ' height.'.format(im.shape[0], sample['h']))
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        elif sample['w'] != im.shape[1]:
            logger.warning(
                'The actual image width: {} is not equal to the width :{}'
                ' in annotation, and update sample[\'h\'] by actual image'
                ' width.'.format(im.shape[1], sample['2']))
            sample['w'] = im.shape[1]

        sample['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        return sample

class RandomFlip(BaseOperator):
    def __init__(self, prob=0.5):
        '''
        :param prob: type float, the probabilty of flipping image
        '''
        super(RandomFlip, self).__init__()
        self.prob = prob
        if not(isinstance(self.prob, float)):
            raise  TypeError("{}: input type is invalid.".format(self))

    def apply_segm(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2])
            return flipped_poly.tolist()

        def _filp_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        flipped_segms = []
        for segm in segms:
            if self.is_poly(segm):
                flipped_segms.append([_flip_poly(poly, width)] for poly in segm)
            else:
                flipped_segms.append(_filp_rle(segm, height, width))

        return flipped_segms

    def apply_keypoint(self, gt_keypoint, width):
        for i in range(gt_keypoint.shape[1]):
            if i%2 == 0:
                old_x = gt_keypoint[:, i].copy()
                gt_keypoint[:, 1] = width - old_x
        return gt_keypoint

    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def apply_rbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        oldx3 = bbox[:, 4].copy()
        oldx4 = bbox[:, 6].copy()
        bbox[:, 0] = width - oldx1
        bbox[:, 2] = width - oldx2
        bbox[:, 4] = width - oldx3
        bbox[:, 6] = width - oldx4
        #bbox = [bbox_utils.get_best_begin_point_single(e) for e in bbox]
        bbox = [e for e in bbox]
        return bbox

    def apply(self, sample, context=None):
        '''
         Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        :param sample:
        :param context:
        :return:
        '''
        if np.random.uniform(0, 1) < self.prob:
            im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], width)
            if 'gt_poly' in sample and len(sample['gt_ploy']) > 0:
                sample['gt_ploy'] = self.apply_segm(sample['gt_ploy'], height, width)
            if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
                sample['gt_keypoint'] = self.apply_keypoint(sample['gt_keypoint'], width)
            if 'semantic' in sample and sample['semantic']:
                sample['semantic'] = sample['semantic'][:, ::-1]
            if 'gt_segm' in sample and sample['gt_segm'].any():
                sample['gt_segm'] = sample['gt_segm'][:, :, ::-1]
            if 'gt_rbox2poly' in sample and sample['gt_rbox2poly'].any():
                sample['gt_rbox2poly'] = self.apply_rbox(sample['gt_rbox2poly'], width)

            sample['flipped'] = True
            sample['image'] = im
        return sample

class RandomSelect(BaseOperator):
    '''
    Randomly choose a transformation between transforms1 and transforms2
    and the probability of choosing transforms1 is p
    '''
    def __init__(self, transforms1, transforms2, p=0.5):
        super(RandomSelect, self).__init__()
        #from paddle.vision.transforms import Compose
        self.transforms1 = Compose(transforms1)
        self.transforms2 = Compose(transforms2)
        # self.transforms1 = transforms1
        # self.transforms2 = transforms2
        self.p = p

    def apply(self, sample, context=None):
        if random.random() <  self.p:
            return self.transforms1(sample)
        return self.transforms2(sample)

class RandomShortSideResize(BaseOperator):
    def __init__(self, short_side_sizes, max_size=None,
                 interp=cv2.INTER_LINEAR, random_interp=False):
        '''
        Resize the image randomly according to the short side. if max_size is not None,
        the long side is scaled according to max_size. The whole process will be keep ratio.
        :param short_side_sizes (list|tuple): Image target short side size.
        :param max_size (int): The size of the longest side of image after resize.
        :param interp (int): The interpolation method.
        :param random_interp (bool): Whether random select interpolation method.
        '''
        super(RandomShortSideResize, self).__init__()
        assert isinstance(short_side_sizes, Sequence), 'short_side_sizes must be List or Tuple'
        self.short_side_sizes = short_side_sizes
        self.max_size = max_size
        self.interp = interp
        self.random_interp = random_interp
        self.interps = [cv2.INTER_NEAREST,
                        cv2.INTER_LINEAR,
                        cv2.INTER_AREA,
                        cv2.INTER_CUBIC,
                        cv2.INTER_LANCZOS4]

    def get_size_with_sapect_ratio(self, image_shape, size, max_size=None):
        h, w = image_shape
        if max_size is not None:
            min_original_size = float(min(w, h))
            max_original_size = float(max(w, h))
            if max_original_size / min_original_size * size > max_size:
                size = int(max_size * min_original_size / max_original_size)


        if (w <= h and w == size) or (h <= w and h == size):
            return (w, h)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (ow, oh)

    def resize(self, sample, target_size, max_size=None, interp=cv2.INTER_LINEAR):
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError('{}: image type is not numpy'.format(self))
        if len(im.shape) != 3:
            raise TypeError('{}: image is not 3-dimensional.'.format(self))
        target_size = self.get_size_with_sapect_ratio(im.shape[:2], target_size, max_size)
        im_scale_y, im_scale_x = target_size[1] / im.shape[0], target_size[0] / im.shape[1]
        sample['image'] = cv2.resize(im, target_size, interpolation=interp)
        sample['im_shape'] = np.asarray(target_size[::-1], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['sacle_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0]*im_scale_y, scale_factor[1]*im_scale_x], dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray([im_scale_y, im_scale_x], dtype=np.float32)

        if 'gt_bbox' in sample and len(sample['gt_sample']) > 0:
            sample['gt_bbox'] = self.apply_bbox(
                sample['gt_bbox'], [im_scale_x, im_scale_y], target_size)

        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'],
                                                im.shape[:2], [im_scale_x, im_scaleimport copy
import random
import uuid
import cv2
import paddle
import shapely
import functools

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import numpy as np
from collections.abc import Sequence
from logger import logger
import pycocotools.mask as mask_util

from compose import Compose


class BaseOperator():
    def __int__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def apply(self, sample, context=None):
        '''
        :param sample(dict): {'image':xx, 'label': yy}
        :param context: a dict, info about this sample processing
        :return: a dict ,a processed sample
        '''
        #if context is None:
        return sample

    def __call__(self, sample, context =None):
        '''
        :param sample(dict): {'image':xx, 'label': yy}
        :param context: a dict, info about this sample processing
        :return: a dict ,a processed sample
        '''
        if isinstance(sample, Sequence):
            for i in range(len(sample)):
                sample[i] = self.apply(sample[i], context)
        else:
            sample = self.apply(sample, context)

        return sample

    def is_poly(self, segm):
        return isinstance(segm, np.ndarray) and segm.shape >= 2

    def __str__(self):
        return str(self._id)

class Decode(BaseOperator):
    def __init__(self):
        #Transform image data to numpy format following the rgb format
        super(Decode, self).__init__()

    def apply(self, sample, context=None):
        #load image if im_file field is not empty but image is empty
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()
            sample.pop('im_file')

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)
        if 'keep_ori_im' in sample and sample['keep_ori_im']:
            sample['ori_image'] = im
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample['image'] = im
        if 'h' not in sample:
            sample['h'] = im.shape[0]
        elif sample['h'] != im.shape[0]:
            logger.warning(
                'The actual image height: {} is not equal to the height :{}'
                ' in annotation, and update sample[\'h\'] by actual image'
                ' height.'.format(im.shape[0], sample['h']))
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        elif sample['w'] != im.shape[1]:
            logger.warning(
                'The actual image width: {} is not equal to the width :{}'
                ' in annotation, and update sample[\'h\'] by actual image'
                ' width.'.format(im.shape[1], sample['2']))
            sample['w'] = im.shape[1]

        sample['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        return sample

class RandomFlip(BaseOperator):
    def __init__(self, prob=0.5):
        '''
        :param prob: type float, the probabilty of flipping image
        '''
        super(RandomFlip, self).__init__()
        self.prob = prob
        if not(isinstance(self.prob, float)):
            raise  TypeError("{}: input type is invalid.".format(self))

    def apply_segm(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2])
            return flipped_poly.tolist()

        def _filp_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        flipped_segms = []
        for segm in segms:
            if self.is_poly(segm):
                flipped_segms.append([_flip_poly(poly, width)] for poly in segm)
            else:
                flipped_segms.append(_filp_rle(segm, height, width))

        return flipped_segms

    def apply_keypoint(self, gt_keypoint, width):
        for i in range(gt_keypoint.shape[1]):
            if i%2 == 0:
                old_x = gt_keypoint[:, i].copy()
                gt_keypoint[:, 1] = width - old_x
        return gt_keypoint

    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def apply_rbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        oldx3 = bbox[:, 4].copy()
        oldx4 = bbox[:, 6].copy()
        bbox[:, 0] = width - oldx1
        bbox[:, 2] = width - oldx2
        bbox[:, 4] = width - oldx3
        bbox[:, 6] = width - oldx4
        #bbox = [bbox_utils.get_best_begin_point_single(e) for e in bbox]
        bbox = [e for e in bbox]
        return bbox

    def apply(self, sample, context=None):
        '''
         Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        :param sample:
        :param context:
        :return:
        '''
        if np.random.uniform(0, 1) < self.prob:
            im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], width)
            if 'gt_poly' in sample and len(sample['gt_ploy']) > 0:
                sample['gt_ploy'] = self.apply_segm(sample['gt_ploy'], height, width)
            if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
                sample['gt_keypoint'] = self.apply_keypoint(sample['gt_keypoint'], width)
            if 'semantic' in sample and sample['semantic']:
                sample['semantic'] = sample['semantic'][:, ::-1]
            if 'gt_segm' in sample and sample['gt_segm'].any():
                sample['gt_segm'] = sample['gt_segm'][:, :, ::-1]
            if 'gt_rbox2poly' in sample and sample['gt_rbox2poly'].any():
                sample['gt_rbox2poly'] = self.apply_rbox(sample['gt_rbox2poly'], width)

            sample['flipped'] = True
            sample['image'] = im
        return sample

class RandomSelect(BaseOperator):
    '''
    Randomly choose a transformation between transforms1 and transforms2
    and the probability of choosing transforms1 is p
    '''
    def __init__(self, transforms1, transforms2, p=0.5):
        super(RandomSelect, self).__init__()
        #from paddle.vision.transforms import Compose
        self.transforms1 = Compose(transforms1)
        self.transforms2 = Compose(transforms2)
        # self.transforms1 = transforms1
        # self.transforms2 = transforms2
        self.p = p

    def apply(self, sample, context=None):
        if random.random() <  self.p:
            return self.transforms1(sample)
        return self.transforms2(sample)

class RandomShortSideResize(BaseOperator):
    def __init__(self, short_side_sizes, max_size=None,
                 interp=cv2.INTER_LINEAR, random_interp=False):
        '''
        Resize the image randomly according to the short side. if max_size is not None,
        the long side is scaled according to max_size. The whole process will be keep ratio.
        :param short_side_sizes (list|tuple): Image target short side size.
        :param max_size (int): The size of the longest side of image after resize.
        :param interp (int): The interpolation method.
        :param random_interp (bool): Whether random select interpolation method.
        '''
        super(RandomShortSideResize, self).__init__()
        assert isinstance(short_side_sizes, Sequence), 'short_side_sizes must be List or Tuple'
        self.short_side_sizes = short_side_sizes
        self.max_size = max_size
        self.interp = interp
        self.random_interp = random_interp
        self.interps = [cv2.INTER_NEAREST,
                        cv2.INTER_LINEAR,
                        cv2.INTER_AREA,
                        cv2.INTER_CUBIC,
                        cv2.INTER_LANCZOS4]

    def get_size_with_sapect_ratio(self, image_shape, size, max_size=None):
        h, w = image_shape
        if max_size is not None:
            min_original_size = float(min(w, h))
            max_original_size = float(max(w, h))
            if max_original_size / min_original_size * size > max_size:
                size = int(max_size * min_original_size / max_original_size)


        if (w <= h and w == size) or (h <= w and h == size):
            return (w, h)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (ow, oh)

    def resize(self, sample, target_size, max_size=None, interp=cv2.INTER_LINEAR):
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError('{}: image type is not numpy'.format(self))
        if len(im.shape) != 3:
            raise TypeError('{}: image is not 3-dimensional.'.format(self))
        target_size = self.get_size_with_sapect_ratio(im.shape[:2], target_size, max_size)
        im_scale_y, im_scale_x = target_size[1] / im.shape[0], target_size[0] / im.shape[1]
        sample['image'] = cv2.resize(im, target_size, interpolation=interp)
        sample['im_shape'] = np.asarray(target_size[::-1], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['sacle_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0]*im_scale_y, scale_factor[1]*im_scale_x], dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray([im_scale_y, im_scale_x], dtype=np.float32)

        if 'gt_bbox' in sample and len(sample['gt_sample']) > 0:
            sample['gt_bbox'] = self.apply_bbox(
                sample['gt_bbox'], [im_scale_x, im_scale_y], target_size)

        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'],
                                                im.shape[:2], [im_scale_x, im_scale_y])
        if 'semantic' in sample and sample['semantic']:
            semantic = sample['semantic']
            semantic = cv2.resize(semantic.astype('float32'),
                                  target_size, self.interp)
            semantic = np.asarray(semantic, 0)
            sample['semantic'] = semantic
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(gt_segm, target_size, cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)
        return sample

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox.astype('float32')

    def apply_segm(self, segms, im_size, scale):
        def _resize_poly(poly, im_scale_x, im_scale_y):
            resized_poly = np.array(poly).astype('float32')
            resized_poly[0::2] *= im_scale_x
            resized_poly[1::2] *= im_scale_y
            return resized_poly.tolist()

        def _resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, im_h, im_w)
            mask = mask_util.decode(rle)
            mask = cv2.resize(mask, fx=im_scale_x, fy=im_scale_y, interpolation=self.interp)
            rle = mask_util.decode(np.array(mask, order='F', dtype=np.uint8))
            return rle
        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if self.is_poly(segm):
                #if is_poly(segm):
                resized_segms.append([
                    _resize_poly(poly, im_scale_x, im_scale_y) for poly in segm])
            else:
                resized_segms.append(_resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))
        return resized_segms

    def apply(self, sample, context=None):
        target_size = random.choice(self.short_side_sizes)
        interp = random.choice(self.interps) if self.random_interp else self.interp
        return self.resize(sample, target_size, self.max_size, interp)

class RandomSizeCrop(BaseOperator):
    def __init__(self, min_size, max_size):
        super(RandomSizeCrop, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.paddle_crop = paddle.vision.transforms.functional.crop

    @staticmethod
    def get_crop_params(img_shape, out_size):
        '''
        Get paramters for crop for a random crop.
        :param img_shape (list | tuple): Image's height and width.
        :param out_size (list | tuple): Expected output size of the crop.
        :return tuple: params (i, j, h, w) to be passed to crop for random crop.
        '''
        h, w = img_shape
        th, tw = out_size
        if h+1 < th or w+1 < tw:
            return 0, 0, h, w

    def crop(self, sample, region):
        image_shape = sample['image'].shape
        sample['image'] = self.paddle_crop(sample['image'], *region)
        keep_index = None

        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], region)
            bbox = sample['gt_bbox'].reshape([-1, 2, 2])
            area = (bbox[:, 1, :] - bbox[:, 0, :]).prod(axis=1)
            keep_index = np.where(area > 0)[0]
            if len(keep_index) > 0:
                sample['gt_bbox'] = sample['gt_bbox'][keep_index]
                sample['gt_class'] = sample['gt_class'][keep_index]
                if 'gt_score' in sample:
                    sample['gt_score'] = sample['gt_score'][keep_index]
                if 'is_crowd' in sample:
                    sample['is_crowd'] = sample['is_crowd'][keep_index]
            else:
                sample['gt_bbox'] = np.zeros([0, 4], dtype=np.float32)
                sample['gt_class'] = np.zeros([0, 1], dtype=np.float32)
                if 'gt_score' in sample:
                    sample['gt_score'] = np.zeros([0 ,1], dtype=np.float32)
                if 'gt_crowd' in sample:
                    sample['gt_crowd'] = np.zeros([0, 1], dtype=np.float32)

        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_ploy'], region, image_shape)
            if keep_index is not None:
                sample['gt_segm'] = sample['gt_segm'][keep_index]
        return sample

    def apply_bbox(self, bbox, region):
        i, j, h, w = region
        region_size = np.asarray([w, h])
        crop_bbox = bbox - np.asarray([j, i, j ,i])
        crop_bbox = np.minimum(crop_bbox.reshape[-1, 2, 2], region_size)
        crop_bbox = crop_bbox.clip(min=0)
        return crop_bbox.reshape([-1, 4]).astype('float32')

    def apply_segm(self, segms, region, image_shape):
        def _crop_poly(segm, crop):
            xmin, ymin, xmax, ymax = crop
            crop_coord = [xmin, ymin, xmin, ymax, xmax, ymax, xmax , ymin]
            crop_p = np.array(crop_coord).reshape(4, 2)
            crop_p = Polygon(crop_p)
            crop_segm = list()
            for poly in segm:
                poly = np.array(poly).reshape(len(poly)//2, 2)
                polygon = Polygon(poly)
                if not polygon.is_valid:
                    exterior = polygon.exterior
                    multi_lines = exterior.intersection(exterior)
                    polygons = shapely.ops.polygonize(multi_lines)
                    polygon = MultiPolygon(polygons)
                multi_polygon = list()
                if isinstance(polygon, MultiPolygon):
                    multi_polygon = copy.deepcopy(polygon)
                else:
                    multi_polygon.append(copy.deepcopy(polygon))
                for per_polygon in multi_polygon:
                    inter = per_polygon.intersection(crop_p)
                    if not inter:
                        continue
                    if isinstance(inter, (MultiPolygon, GeometryCollection)):
                        for part in inter:
                            if not isinstance(part, Polygon):
                                continue
                            part = np.squeeze(np.array(part.exterior.coords[:-1]).reshape(1, -1))
                            part[0::2] -= xmin
                            part[1::2] -= ymin
                            crop_segm.append(part.tolist())
                    elif isinstance(inter, Polygon):
                        crop_poly = np.squeeze(np.array(inter.exterior.coords[:-1]).reshape(1, -1))
                        crop_poly[0::2] -= xmin
                        crop_poly[1::2] -= ymin
                        crop_segm.append(crop_poly.tolist())
                    else:
                        continue
            return crop_segm

        def _crop_rle(rle, crop, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        i, j, h, w = region
        crop = [j, i, j+w, i+h]
        height, width = image_shape
        crop_segms = []
        for segm in segms:
            if self.is_poly(segm):
                crop_segms.append(_crop_poly(segm, crop))
            else:
                crop_segms.append(_crop_rle(segm, crop, height, width))
        return crop_segms

    def apply(self, sample, context=None):
        h = random.randint(self.min_size, min(sample['image'].shape[0], self.max_size))
        w = random.randint(self.min_size, min(sample['image'].shape[1], self.max_size))
        region = self.get_crop_params(sample['image'].shape[:2], [h, w])
        return self.crop(sample, region)

class NormalizeImage(BaseOperator):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[1, 1, 1], is_scale=True):
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale

        if not (isinstance(self.mean, list) and isinstance(self.std, list)) \
            and isinstance(self.is_scale, bool):
            raise TypeError('{}: input type is invalid.'.format(self))
        if functools.reduce(lambda x, y: x*y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def apply(self, sample, context=None):
        im = sample['image']
        im = im.astype(np.float32, copy=False)
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        if self.is_scale:
            im = im / 255.0

        im -= mean
        im /= std
        sample['image'] = im
        return sample

class NormalizeBox(BaseOperator):
    """Transform the bounding box's coornidates to [0,1]."""

    def __init__(self):
        super(NormalizeBox, self).__init__()

    def apply(self, sample, context):
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        height, width, _ = im.shape
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] = gt_bbox[i][0] / width
            gt_bbox[i][1] = gt_bbox[i][1] / height
            gt_bbox[i][2] = gt_bbox[i][2] / width
            gt_bbox[i][3] = gt_bbox[i][3] / height
        sample['gt_bbox'] = gt_bbox

        if 'gt_keypoint' in sample.keys():
            gt_keypoint = sample['gt_keypoint']

            for i in range(gt_keypoint.shape[1]):
                if i % 2:
                    gt_keypoint[:, i] = gt_keypoint[:, i] / height
                else:
                    gt_keypoint[:, i] = gt_keypoint[:, i] / width
            sample['gt_keypoint'] = gt_keypoint

        return sample

class BboxXYXY2XYWH(BaseOperator):
    def __init__(self):
        super(BboxXYXY2XYWH, self).__init__()

    def apply(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.0
        sample['gt_bbox'] = bbox
        return sample

class Permute(BaseOperator):
    def __init__(self):
        super(Permute, self).__init__()

    def apply(self, sample, context=None):
        im = sample['image']
        im = im.transpose([2, 0, 1])
        sample['image'] = im
        return sample

class Resize(BaseOperator):
    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox

    def apply_segm(self, segms, im_size, scale):
        def _resize_poly(poly, im_scale_x, im_scale_y):
            resized_poly = np.array(poly).astype('float32')
            resized_poly[0::2] *= im_scale_x
            resized_poly[1::2] *= im_scale_y
            return resized_poly.tolist()

        def _resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, im_h, im_w)

            mask = mask_util.decode(rle)
            mask = cv2.resize(
                mask,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if self.is_poly(segm):
                # Polygon format
                resized_segms.append([
                    _resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                resized_segms.append(
                    _resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise TypeError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        im_shape = im.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        im = self.apply_image(sample['image'], [im_scale_x, im_scale_y])
        sample['image'] = im
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                [im_scale_x, im_scale_y],
                                                [resize_w, resize_h])

        # apply rbox
        if 'gt_rbox2poly' in sample:
            if np.array(sample['gt_rbox2poly']).shape[1] != 8:
                logger.warning(
                    "gt_rbox2poly's length shoule be 8, but actually is {}".
                    format(len(sample['gt_rbox2poly'])))
            sample['gt_rbox2poly'] = self.apply_bbox(sample['gt_rbox2poly'],
                                                     [im_scale_x, im_scale_y],
                                                     [resize_w, resize_h])

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_shape[:2],
                                                [im_scale_x, im_scale_y])

        # apply semantic
        if 'semantic' in sample and sample['semantic']:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic

        # apply gt_segm
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm,
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)

        return sample

if __name__ == "__main__":
    print("data augmented")

