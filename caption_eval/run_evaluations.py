# encoding: utf-8
# Copyright 2017 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluation utility for image Chinese captioning task."""
# __author__ = 'ZhengHe'
# python2.7
# python run_evaluations.py --submit=your_result_json_file --ref=reference_json_file

import sys
import time

reload(sys)
sys.setdefaultencoding('utf8')
from coco_caption.pycxtools.coco import COCO
from coco_caption.pycxevalcap.eval import COCOEvalCap


def compute_m1(json_predictions_file, reference_file):
    """Compute m1_score"""
    m1_score = {}
    m1_score['error'] = 0
    try:
        coco = COCO(reference_file)
        coco_res = coco.loadRes(json_predictions_file)

        # create coco_eval object.
        coco_eval = COCOEvalCap(coco, coco_res)

        # evaluate results
        coco_eval.evaluate()
    except Exception:
        m1_score['error'] = 1
    else:
        # print output evaluation scores
        for metric, score in coco_eval.eval.items():
            print '%s: %.3f'%(metric, score)
            m1_score[metric] = score
    return m1_score


def main():
    time_s = time.time()
    # json_predictions_file = '../test/val_30000_20171004_model_20.json'
    # dir_prediction = '../test/val_30000_20171005_model_50.json'
    dir_prediction = '../test/val_30000_20171005_model_40.json'

    dir_reference = '../test/val_30000_reference.json'
    method_score = compute_m1(dir_prediction, dir_reference)
    score_mean = (method_score['CIDEr'] + method_score['Bleu_4'] + method_score['ROUGE_L'] + method_score['METEOR']) / 4.0
    time_spent = (time.time() - time_s) / 60
    print method_score
    print ("AI-challenger score: %.5f | %.5f | %.5f | %.5f | %.5f" %
           (method_score['Bleu_4'], method_score['CIDEr'], method_score['METEOR'], method_score['ROUGE_L'], score_mean))
    print 'time cost:', time_spent, 'min'

if __name__ == "__main__":
    main()
