#  Copyright (c) 2025, Mikhail Kiselev, Anastasiia Kornilova
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


class Backend(ABC):
    """Abstract base class for all backends.

    This class defines the interface that all backend implementations must follow.
    A backend is responsible for executing model inference using a specific
    framework or runtime.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initializes the backend.

        Args:
            *args: Positional arguments for initialization.
            **kwargs: Keyword arguments for initialization.
        """
        pass

    @abstractmethod
    def __call__(self, x):
        """Executes the model on the input data.

        Args:
            x: Input data for model inference.

        Returns:
            Model output after inference.
        """
        pass
