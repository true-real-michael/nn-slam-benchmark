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
from .eigenplaces import EigenPlaces
from .cosplace import CosPlace
from .mixvpr import MixVPR, MixVPRShrunk
from .netvlad import NetVLAD
from .salad import SALAD, SALADShrunk
from .sela import Sela, SelaShrunk

__all__ = [
    "EigenPlaces",
    "CosPlace",
    "MixVPR",
    "MixVPRShrunk",
    "NetVLAD",
    "Sela",
    "SelaShrunk",
    "SALAD",
    "SALADShrunk",
]
