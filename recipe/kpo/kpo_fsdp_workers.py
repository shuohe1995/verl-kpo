# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
KPO FSDP workers: register KPO policy loss into verl's registry, then re-export
AsyncActorRolloutRefWorker so that Ray workers load this module first and see "kpo" loss.
All changes stay inside recipe/kpo (no modification to verl).
"""

# Register KPO policy loss into verl's POLICY_LOSS_REGISTRY *before* importing
# verl.workers.fsdp_workers, so that when dp_actor is loaded it will find "kpo".
import verl.trainer.ppo.core_algos as _verl_core_algos
from recipe.kpo.core_algos import POLICY_LOSS_REGISTRY as _kpo_registry

if "kpo" in _kpo_registry and "kpo" not in _verl_core_algos.POLICY_LOSS_REGISTRY:
    _verl_core_algos.POLICY_LOSS_REGISTRY["kpo"] = _kpo_registry["kpo"]

from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker  # noqa: E402


class KpoAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Worker class defined in recipe so Ray loads this module first and KPO is registered."""


__all__ = ["AsyncActorRolloutRefWorker", "KpoAsyncActorRolloutRefWorker"]
