#  Copyright (c) 2025, Ivan Moskalenko, Anastasiia Kornilova, Mikhail Kiselev
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
from abc import ABC, abstractmethod

from nnsb.method import Method


class FeatureMatcher(Method, ABC):
    """Base class for all feature matchers.

    This abstract class defines the interface for feature matchers, which
    match features between a query image and a database image.
    """

    def __call__(self, query_feat, db_feat):
        """Runs the feature matching pipeline.

        Args:
            query_feat: Features from the query image.
            db_feat: Features from the database image.

        Returns:
            Matching results, typically containing the number of matches
            and corresponding points.
        """
        query_feat = self.preprocess(query_feat)
        db_feat = self.preprocess(db_feat)
        matches = self.backend((query_feat, db_feat))
        return self.postprocess(query_feat, db_feat, matches)

    def preprocess(self, feat):
        """Preprocesses features for matching.

        Args:
            feat: Input features to preprocess.

        Returns:
            Preprocessed features.
        """
        return feat

    @abstractmethod
    def postprocess(self, query_feat, db_feat, matches):
        """Postprocesses the matcher output.

        Args:
            query_feat: Features from the query image.
            db_feat: Features from the database image.
            matches: Raw matching results from the backend.

        Returns:
            Processed matching results.
        """
        pass
