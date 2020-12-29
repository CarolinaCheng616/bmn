import json

import numpy as np
from mmcv.utils import print_log

from ...utils import get_root_logger
from .accuracy import (average_recall_at_avg_proposals,
                       interpolated_precision_recall, pairwise_temporal_iou)


class ActivityNetDetection:
    """Class to evaluate detection results on ActivityNet.

    Args:
        ground_truth_filename (str | None): The filename of groundtruth.
            Default: None.
        prediction_filename (str | None): The filename of action detection
            results. Default: None.
        tiou_thresholds (np.ndarray): The thresholds of temporal iou to
            evaluate. Default: ``np.linspace(0.5, 0.95, 10)``.
        verbose (bool): Whether to print verbose logs. Default: False.
    """

    def __init__(self,
                 ground_truth_filename=None,
                 prediction_filename=None,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 verbose=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.ground_truth_filename = ground_truth_filename
        self.prediction_filename = prediction_filename
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.ap = None
        self.logger = get_root_logger()
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            log_msg = (
                '[INIT] Loaded ground_truth from '
                f'{self.ground_truth_filename}, prediction from '
                f'{self.prediction_filename}.\n'
                f'Number of ground truth instances: {len(self.ground_truth)}\n'
                f'Number of predictions: {len(self.prediction)}\n'
                f'Fixed threshold for tiou score: {self.tiou_thresholds}')
            print_log(log_msg, logger=self.logger)

    @staticmethod
    def _import_ground_truth(ground_truth_filename):
        """Read ground truth file and return the ground truth instances and the
        activity classes.

        Args:
            ground_truth_filename (str): Full path to the ground truth json
                file.

        Returns:
            tuple[list, dict]: (ground_truth, activity_index).
                ground_truth contains the ground truth instances, which is in a
                    dict format.
                activity_index contains classes index.
        """
        with open(ground_truth_filename, 'r') as f:
            data = json.load(f)
        # Checking format
        activity_index, class_idx = {}, 0
        ground_truth = []
        for video_id, video_info in data.items():
            for anno in video_info['annotations']:
                if anno['label'] not in activity_index:
                    activity_index[anno['label']] = class_idx
                    class_idx += 1
                # old video_anno
                ground_truth_item = {}
                ground_truth_item['video-id'] = video_id[2:]
                ground_truth_item['t-start'] = float(anno['segment'][0])
                ground_truth_item['t-end'] = float(anno['segment'][1])
                ground_truth_item['label'] = activity_index[anno['label']]
                ground_truth.append(ground_truth_item)

        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Read prediction file and return the prediction instances.

        Args:
            prediction_filename (str): Full path to the prediction json file.

        Returns:
            List: List containing the prediction instances (dictionaries).
        """
        with open(prediction_filename, 'r') as f:
            data = json.load(f)
        # Read predictions.
        prediction = []
        for video_id, video_info in data['results'].items():
            for result in video_info:
                prediction_item = dict()
                prediction_item['video-id'] = video_id
                prediction_item['label'] = self.activity_index[result['label']]
                prediction_item['t-start'] = float(result['segment'][0])
                prediction_item['t-end'] = float(result['segment'][1])
                prediction_item['score'] = result['score']
                prediction.append(prediction_item)

        return prediction

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class."""
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = []
        prediction_by_label = []
        for i in range(len(self.activity_index)):
            ground_truth_by_label.append([])
            prediction_by_label.append([])
        for gt in self.ground_truth:
            ground_truth_by_label[gt['label']].append(gt)
        for pred in self.prediction:
            prediction_by_label[pred['label']].append(pred)

        for i in range(len(self.activity_index)):
            ap_result = compute_average_precision_detection(
                ground_truth_by_label[i],
                prediction_by_label[i],
                tiou_thresholds=self.tiou_thresholds)
            ap[:, i] = ap_result

        return ap

    def evaluate(self):
        """Evaluates a prediction file.

        For the detection task we measure the interpolated mean average
        precision to measure the performance of a method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        return self.mAP, self.average_mAP


class TruNetDetection:
    """Class to evaluate detection results on TruNet dataset.

    Args:
        ground_truth_filename (str | None): The filename of groundtruth.
            Default: None.
        prediction_filename (str | None): The filename of action detection
            results. Default: None.
        tiou_thresholds (np.ndarray): The thresholds of temporal iou to
            evaluate. Default: ``np.linspace(0.5, 0.95, 10)``.
        verbose (bool): Whether to print verbose logs. Default: False.
    """

    def __init__(self,
                 ground_truth_filename=None,
                 prediction_filename=None,
                 proposal_num=13,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 verbose=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.ground_truth_filename = ground_truth_filename
        self.prediction_filename = prediction_filename
        self.proposal_num = proposal_num
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.ap = None
        self.logger = get_root_logger()
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            log_msg = (
                '[INIT] Loaded ground_truth from '
                f'{self.ground_truth_filename}, prediction from '
                f'{self.prediction_filename}.\n'
                f'Number of ground truth instances: {len(self.ground_truth)}\n'
                f'Number of predictions: {len(self.prediction)}\n'
                f'Fixed threshold for tiou score: {self.tiou_thresholds}')
            print_log(log_msg, logger=self.logger)

    @staticmethod
    def _import_ground_truth(ground_truth_filename):
        """Read ground truth file and return the ground truth instances and the
        activity classes.

        Args:
            ground_truth_filename (str): Full path to the ground truth json
                file.

        Returns:
            tuple[list, dict]: (ground_truth, activity_index).
                ground_truth contains the ground truth instances, which is in a
                    dict format.
                activity_index contains classes index.
        """
        with open(ground_truth_filename, 'r') as f:
            data = json.load(f)
        # Checking format
        # activity_index, class_idx = {}, 0

        ground_truth = []
        # avg_anno = 0
        # video_number = len(data)
        for video_id, video_info in data.items():
            # avg_anno += len(video_info['annotations'])
            for anno in video_info['annotations']:
                ground_truth_item = dict()
                ground_truth_item['video-id'] = video_id
                ground_truth_item['t-start'] = float(anno['segment'][0])
                ground_truth_item['t-end'] = float(anno['segment'][1])
                # ground_truth_item['label'] = activity_index[anno['label']]
                ground_truth_item['label'] = 0
                ground_truth.append(ground_truth_item)
        # print(f'average annos per video:{avg_anno/video_number}')
        return ground_truth, {0: 0}

    def _import_prediction(self, prediction_filename):
        """Read prediction file and return the prediction instances.

        Args:
            prediction_filename (str): Full path to the prediction json file.

        Returns:
            List: List containing the prediction instances (dictionaries).
        """

        threshold = 0.9

        with open(prediction_filename, 'r') as f:
            data = json.load(f)
        # Read predictions.
        prediction = []
        # avg_proposal = 0
        score_sum = 0
        selected_proposal = 0
        min_score_list = []
        max_score_list = []
        for video_id, video_info in data.items():
            video_info = sorted(
                video_info, key=lambda x: x['score'], reverse=True)
            video_min_score = float('inf')
            video_max_score = 0
            for result in video_info[:self.proposal_num]:
                if result['score'] < threshold:
                    break
                prediction_item = dict()
                prediction_item['video-id'] = video_id
                prediction_item['label'] = 0
                prediction_item['t-start'] = float(result['segment'][0])
                prediction_item['t-end'] = float(result['segment'][1])
                prediction_item['score'] = result['score']
                prediction.append(prediction_item)
                score_sum += result['score']
                selected_proposal += 1
                video_min_score = min(video_min_score, result['score'])
                video_max_score = max(video_max_score, result['score'])
            min_score_list.append(video_min_score)
            max_score_list.append(video_max_score)
        return prediction

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class."""
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = []
        prediction_by_label = []
        for i in range(len(self.activity_index)):
            ground_truth_by_label.append([])
            prediction_by_label.append([])
        for gt in self.ground_truth:
            ground_truth_by_label[gt['label']].append(gt)
        for pred in self.prediction:
            prediction_by_label[pred['label']].append(pred)

        for i in range(len(self.activity_index)):
            # import pdb
            # pdb.set_trace()
            ap_result = compute_average_precision_detection(
                ground_truth_by_label[i],
                prediction_by_label[i],
                self.proposal_num,
                tiou_thresholds=self.tiou_thresholds)
            ap[:, i] = ap_result

        return ap

    def evaluate(self):
        """Evaluates a prediction file.

        For the detection task we measure the interpolated mean average
        precision to measure the performance of a method.
        """
        self.ap = self.wrapper_compute_average_precision()
        # ap.shape = [len(threshold), len(class)]

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        self.ARAN()

        return self.mAP, self.average_mAP

    def ARAN(self):
        # import ground truth by array
        with open(self.ground_truth_filename, 'r') as f:
            data = json.load(f)
        ground_truth = dict()
        for video_id, video_info in data.items():
            vinfo = []
            for anno in video_info['annotations']:
                vinfo.append([anno['segment'][0], anno['segment'][1]])
            ground_truth[video_id] = np.array(vinfo)

        with open(self.prediction_filename, 'r') as f:
            data = json.load(f)
        prediction = dict()
        total_num_proposals = 0
        for video_id, video_info in data.items():
            vinfo = []
            for anno in video_info:
                total_num_proposals += 1
                vinfo.append(
                    [anno['segment'][0], anno['segment'][1], anno['score']])
            prediction[video_id] = np.array(vinfo)
        recall, _, _, auc = average_recall_at_avg_proposals(
            ground_truth,
            prediction,
            total_num_proposals,
            max_avg_proposals=100,
            temporal_iou_thresholds=np.linspace(0.5, 0.95, 10))
        print(f'AR@{self.proposal_num}: '
              f'{np.mean(recall[:, self.proposal_num - 1])}')
        print(f'auc: {auc}')


def compute_average_precision_detection(ground_truth,
                                        prediction,
                                        proposal_num=13,
                                        tiou_thresholds=np.linspace(
                                            0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as true
    positive. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (list[dict]): List containing the ground truth instances
            (dictionaries). Required keys are 'video-id', 't-start' and
            't-end'.
        prediction (list[dict]): List containing the prediction instances
            (dictionaries). Required keys are: 'video-id', 't-start', 't-end'
            and 'score'.
        proposal_num (int): Number of average number to be reported in AR@AN.
        tiou_thresholds (np.ndarray): A 1darray indicates the temporal
            intersection over union threshold, which is optional.
            Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        Float: ap, Average precision score.
    """
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    # Sort predictions by decreasing score order.
    prediction.sort(key=lambda x: -x['score'])
    # Initialize true positive and false positive vectors.
    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))

    # Adaptation to query faster
    ground_truth_by_videoid = {}
    for i, item in enumerate(ground_truth):
        item['index'] = i
        ground_truth_by_videoid.setdefault(item['video-id'], []).append(item)

    # Assigning true positive to truly grount truth instances.
    for idx, pred in enumerate(prediction):
        if pred['video-id'] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred['video-id']]
        else:
            fp[:, idx] = 1
            continue

        tiou_arr = pairwise_temporal_iou(
            np.array([pred['t-start'], pred['t-end']]),
            np.array([np.array([gt['t-start'], gt['t-end']]) for gt in gts]))
        tiou_arr = tiou_arr.reshape(-1)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]['index']] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1
    # shape: [num_thresholds, num_preds]
    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(tiou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :],
                                                  recall_cumsum[t_idx, :])

    return ap
