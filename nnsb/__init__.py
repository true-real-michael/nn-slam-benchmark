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
#  See the License for the specific language governging permissions and
#  limitations under the License.
import nnsb.backend as backend
import nnsb.dataset as dataset
import nnsb.feature_matchers as feature_matchers
import nnsb.feature_detectors as feature_detectors
import nnsb.vpr_systems as vpr_systems
import nnsb.benchmarking as benchmarking
import nnsb.method as method

_submodules = [
    backend,
    dataset,
    feature_matchers,
    feature_detectors,
    vpr_systems,
    benchmarking,
    method,
]

__all__ = []

for mod in _submodules:
    mod_all = getattr(mod, "__all__", [])
    __all__.extend(mod_all)
    for name in mod_all:
        globals()[name] = getattr(mod, name)
