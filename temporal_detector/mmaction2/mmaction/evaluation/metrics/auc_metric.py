from typing import Optional, Union, Tuple, Dict, List, Sequence, Any
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from abc import ABCMeta, abstractmethod
import copy
from mmaction.registry import METRICS
from mmengine.evaluator import BaseMetric
import datetime

@METRICS.register_module()
class AucMetric(BaseMetric):
    """AUC evaluation metric."""
    default_prefix: Optional[str] = 'auc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = ('auc', ),
                 collect_device: str = 'cpu',
                 metric_options: Optional[Dict] = dict(),
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        for metric in metrics:
            assert metric in ['auc'], f"Metric {metric} is not supported."

        self.metrics = metrics
        self.metric_options = metric_options

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_score']
            label = data_sample['gt_label']
            frame_dir = data_sample['frame_dir']

            if isinstance(pred, dict):
                for item_name, score in pred.items():
                    pred[item_name] = score.cpu().numpy()
            else:
                pred = pred.cpu().numpy()

            result['pred'] = pred
            if label.size(0) == 1:
                result['label'] = label.item()
            else:
                result['label'] = label.cpu().numpy()
            self.results.append(result)
            result['frame_dir'] = frame_dir

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]

        try:
            preds = [x['pred']['item'][1] for x in results]
        except:
            preds = [x['pred'][1] for x in results]

        frame_dir = [x['frame_dir'] for x in results]
        return self.calculate(preds, labels, frame_dir)

    def calculate(self,
                  preds: List[np.ndarray],
                  labels: List[Union[int, np.ndarray]],
                  frame_dir: Optional[List[str]] = None) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        for metric in self.metrics:
            if metric == 'auc':
                if isinstance(labels[0], np.ndarray):
                    auc = roc_auc_score(np.vstack(labels),
                                        np.vstack(preds),
                                        average='macro')
                else:
                    auc = roc_auc_score(labels, preds)
                eval_results['auc'] = auc
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if frame_dir is not None:
            with open(f'{current_datetime}_', 'w') as f:
                for i in range(len(frame_dir)):
                    f.write(f'{frame_dir[i]} {preds[i]} {labels[i]}\n')

        return eval_results
