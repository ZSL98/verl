import asyncio
from typing import Any, Dict, Optional

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
# from verl.workers.rollout.async_server import AsyncServerBase
from typing import Any
AsyncServerBase = Any

from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class
from verl.experimental.agent_loop.agent_loop import AgentLoopManager

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../SkyRL/skyrl-agent"))
from skyrl_agent import AutoAgentRunner
from .skyagent_async_vllm_server import SkyAgentvLLMReplica


class SkyAgentLoopManager(AgentLoopManager):
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group
        self.rollout_replica_class = SkyAgentvLLMReplica

        self._initialize_llm_servers()
        # self._init_agent_loop_workers()

        # init tokenizer
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        # init generator
        self.server_manager = AsyncLLMServerManager(config, self.server_handles)
        self.skyagent_generator = AutoAgentRunner.from_task(
            task_yaml=config.skyrl_agent.task_yaml, infer_engine=self.server_manager, tokenizer=self.tokenizer
        )

        # Initially we're in sleep mode.
        self.sleep()

    # initialize here with the custom `async_server_class` implementation
    def _initialize_llm_servers(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.worker_group:
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(rollout_config.prometheus, self.server_addresses)

    def _postprocess(self, inputs: Dict[str, list[Any]]) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # inputs
        self.tokenizer.padding_side = "left"
        max_prompt_length = max(
            max([len(input_ids) for input_ids in inputs["prompt_token_ids"]]),
            self.config.actor_rollout_ref.rollout.prompt_length,
        )
        outputs = self.tokenizer.pad(
            [{"input_ids": input_ids} for input_ids in inputs["prompt_token_ids"]],
            padding="max_length",
            max_length=max_prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # responses
        self.tokenizer.padding_side = "right"
        max_response_length = max(
            max([len(response) for response in inputs["response_ids"]]),
            self.config.actor_rollout_ref.rollout.response_length,
        )
        outputs = self.tokenizer.pad(
            [{"input_ids": response_ids} for response_ids in inputs["response_ids"]],
            padding="max_length",
            max_length=max_response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # response_mask
        response_length = response_ids.shape[1]
        loss_masks = [loss_mask + [0] * (response_length - len(loss_mask)) for loss_mask in inputs["loss_masks"]]
        response_mask = torch.tensor(loss_masks, dtype=torch.long)
        assert (
            response_ids.shape == response_mask.shape
        ), f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
        response_mask = response_mask * response_attention_mask

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        return DataProto(
            batch=batch,
            non_tensor_batch={"rewards": np.array(inputs["rewards"])},
            meta_info={"rollout_metrics": inputs["rollout_metrics"], "timing": {}},
        )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        skyagent_output = asyncio.run(
            self.skyagent_generator.run(prompts, val_mode=prompts.meta_info.get("val_mode", False))
        )
        output = self._postprocess(skyagent_output)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        return output

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])