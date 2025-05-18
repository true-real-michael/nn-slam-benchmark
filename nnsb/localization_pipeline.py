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
from tqdm import tqdm

from typing import List, Optional, Tuple
from nnsb.retrieval_system import RetrievalSystem
from nnsb.dataset import Queries


class LocalizationPipeline:
    """
    Allows to create a localizator based on the retrieval system.
    """

    def __init__(
        self,
        retrieval_system: RetrievalSystem,
    ):
        self.retrieval_system = retrieval_system

    def process_all(self, query_seq: Queries, k_closest: int):
        results = self.retrieval_system.process_batch(query_seq, k_closest, k_closest)
        return [result[0][0] for result in results]

    def __call__(
        self,
        query_seq: Queries,
        k_closest: int,
    ) -> List[Optional[Tuple[float, float]]]:
        """
        Calculates UAV locations using the retrieval system.

        :param query_seq: The sequence of images for which locations should be calculated
        :param k_closest: Specifies how many predictions for each query the global localization should make.
        :return: List of geocoordinates. Also, the values can be None if the location could not be determined
        """
        localization_results = []
        for query_image in tqdm(query_seq, desc="Localization"):
            (
                res_prediction,
                matched_kpts_query,
                matched_kpts_reference,
            ) = self.retrieval_system(
                query_image, k_closest, feature_matcher_k_closest=1
            )

            res_prediction = res_prediction[0]
            localization_results.append(res_prediction)
        self.retrieval_system.end_of_query_seq()
        return localization_results
