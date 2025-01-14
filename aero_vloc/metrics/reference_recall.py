#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova, Mikhail Kiselyov
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Dict, Tuple

from nnsb.localization_pipeline import LocalizationPipeline
from nnsb.dataset import Queries


def get_recall(
    eval_q: Queries,
    localization_pipeline: LocalizationPipeline,
    k_closest: int,
) -> Tuple[float, Dict[str, float]]:
    """
    The metric finds the number of correctly matched frames. The frame is considered correctly matched
    if the best match is in the list of positive frames, which is a list of K closest frames to the query.

    :param eval_q: Queries, sequence of images
    :param localization_pipeline: Instance of LocalizationPipeline class
    :param k_closest: Specifies how many predictions for each query the global localization should make.
    If this value is greater than 1, the best match will be chosen with local matcher
    below which the frame will be considered correctly matched

    :return: Recall value
    """
    recall_value = 0

    predictions = localization_pipeline.process_all(eval_q, k_closest)

    for pred, positives in zip(predictions, eval_q.get_positives()):
        if pred in positives:
            recall_value += 1

    recall = recall_value / eval_q.queries_num
    return recall
