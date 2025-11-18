# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import logging
import time

import ray
from ray.util.collective import collective

logger = logging.getLogger(__name__)


@ray.remote
class ParameterSynchronizer:
    """
    Unified parameter synchronizer, responsible for synchronizing model parameters between actor and rollout
    Based on the mature synchronization mode implementation of one_step_off_policy
    Merges the functions of the original multiple synchronizer classes
    """

    def __init__(self, config, trainer, rollouter, mq):
        self.config = config
        self.trainer = trainer
        self.rollouter = rollouter
        self.mq_client = mq
        self.actor_wg = ray.get(trainer.get_actor_wg.remote())
        self.rollout_wg = ray.get(rollouter.get_rollout_wg.remote())

        # Basic attributes
        self.weights_info = None
        self.sync_group_initialized = False
        self.sync_group_name = "actor_rollout"
        self.wait_last_update = None
        self.wait_last_resume = None

        # Statistics
        self.current_version = 0

        self._init_weights_info()
        # self._init_sync_group()

    def get_current_param_version(self) -> int:
        """Get current parameter version number"""
        return self.current_version

    def get_weights_info(self):
        """Get weights info"""
        return self.weights_info

    def _init_weights_info(self):
        self.weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(self.weights_info)

    def _init_sync_group(self):
        print("[ParameterSynchronizer] Initializing parameter synchronization group...")
        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        collective.create_collective_group(
            actor_rollout_workers,
            len(actor_rollout_workers),
            list(range(0, len(actor_rollout_workers))),
            backend="hccl",
            group_name=self.sync_group_name,
        )

    def sync_weights(self, version, validate=False, global_steps=0):
        """Sync weights between trainer and rollouter, and update parameter version"""
        start_time = time.time()

        self.current_version = version
        print(f"[ParameterSynchronizer] Starting weight synchronization (version {self.current_version})...")

        ray.get(self.rollouter.pause.remote())

        # Update MQ version
        self.mq_client.update_param_version_sync(version)

        # sync weights
        self.actor_wg.sync_rollout_weights()
        ray.get(self.rollout_wg.sync_rollout_weights())
        end_time = time.time()
        print(f"[ParameterSynchronizer] sync_weights success. cost {end_time - start_time:.2f} seconds")

        # Async Update rollout version & validation
        self.wait_last_update = self.rollouter.update_param_version.remote(version, validate, global_steps)
        self.wait_last_resume = self.rollouter.resume.remote(self.wait_last_update)

    def sync_weights_through_rank0(self, version, validate=False, global_steps=0):
        start_time = time.time()

        self.current_version = version
        print(f"[ParameterSynchronizer] Starting weight updating (version {self.current_version})...")
        ray.get(self.rollouter.pause.remote())

        # Update MQ version
        self.mq_client.update_param_version_sync(version)

        # actor_rank0 = self.actor_wg
        # print(self.actor_wg.workers)

        # weights_ref_dict_future = self.actor_wg.workers[0].prepare_rollout_weights.remote()
        # weights_ref_dict = ray.get(weights_ref_dict_future)

        # rollout_futures = [
        #     worker.load_rollout_weights.remote(weights_ref_dict)
        #     for worker in self.rollout_wg.workers
        # ]
        # ray.get(rollout_futures)

        weights_ref_dict = self.actor_wg.prepare_rollout_weights()
        self.rollout_wg.load_rollout_weights(weights_ref_dict[0])

        end_time = time.time()
        print(f"[ParameterSynchronizer] sync_weights success. cost {end_time - start_time:.2f} seconds")

        # Async Update rollout version & validation
        self.wait_last_update = self.rollouter.update_param_version.remote(version, validate, global_steps)
        self.wait_last_resume = self.rollouter.resume.remote(self.wait_last_update)

    def update_weights_with_ckpt_engine(self, version, validate=False, global_steps=0):
        """Update weights of rollouter, using ckpt engine"""
        start_time = time.time()

        self.current_version = version
        print(f"[ParameterSynchronizer] Starting weight updating (version {self.current_version})...")

        ray.get(self.rollouter.pause.remote())

        # Update MQ version
        self.mq_client.update_param_version_sync(version)

        # ray.get(self.rollout_wg.load_weight())
        # ray.get(self.rollouter.)
    
        end_time = time.time()
        print(f"[ParameterSynchronizer] sync_weights success. cost {end_time - start_time:.2f} seconds")

        # Async Update rollout version & validation
        self.wait_last_update = self.rollouter.update_param_version.remote(version, validate, global_steps)
        self.wait_last_resume = self.rollouter.resume.remote(self.wait_last_update)

    def wait_last_valid(self):
        print("[ParameterSynchronizer] Waiting last sync and validate...")
        start_time = time.time()
        if self.wait_last_update:
            ray.get(self.wait_last_update)
        if self.wait_last_resume:
            ray.get(self.wait_last_resume)
        print(f"[ParameterSynchronizer] Wait last validate cost: {time.time() - start_time:.2f} seconds")
