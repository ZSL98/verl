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

import os
import yaml
from pathlib import Path

import logging
import re
from typing import Any, Dict, List, Optional

import datasets

from verl.tools.base_tool import OpenAIFunctionToolSchema, ToolResponse
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op
from recipe.fully_async_policy.code_gym.communicator.client import CodeGymClient
from uuid import uuid4
from typing import Any, Optional
logger = logging.getLogger(__name__)

#TODO(P0)-hjl: adapt the code below
class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
        self._instance_dict = {}
        self._code_gym = CodeGymClient()


    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        # 从 kwargs 中获取 create_kwargs
        create_kwargs = kwargs.get("create_kwargs", {})

        # 提取 request_id（如果有的话）
        request_id = create_kwargs.get("request_id", None)
        ground_truth = ground_truth or create_kwargs.get("ground_truth", None)

        # 保存到实例字典中
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
            "request_id": request_id,  # 保存 request_id
        }

        logger.info(f"Created tool instance {instance_id} for request_id={request_id}")
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        # code = parameters["code"]
        # matches = self.code_pattern.findall(code)
        # if matches:
        #     code = matches[0].strip()
        # # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        # lines = code.split("\n")
        # for i, line in reversed(list(enumerate(lines))):
        #     if line == "":
        #         continue
        #     if not lines[i].startswith("print"):
        #         lines[i] = f"print({line})"
        #     break
        # code = "\n".join(lines)

        # timeout = parameters.get("timeout", self.default_timeout)
        # language = parameters.get("language", self.default_language)
        # if not isinstance(code, str):
        #     code = str(code)

        # result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        # # sandbox has no score or metrics, use Nones
        # return result, None, None
        command = parameters.get("command")
        if command is None:
            # Backward-compatible alias (older schema used `commands`)
            command = parameters.get("commands")

        if isinstance(command, list):
            if not command:
                command = ""
            else:
                if len(command) > 1:
                    logger.warning("Received multiple commands; only the first one will be submitted.")
                command = command[0]

        command = str(command or "").strip()

        state = self._instance_dict[instance_id]
        request_id = state.get("request_id")
        if request_id is None:
            logger.warning("request_id not found in instance state. Using instance_id as fallback.")
            request_id = instance_id
        if command:
            self._code_gym.submit_bind_task(request_id, command=command)
        else:
            logger.warning("Empty command received; skip submitting bind task.")
        result_text = ""
        tool_response = ToolResponse(text=result_text)
        reward = 0.0
        return tool_response, reward, None


answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""


class CustomRLHFDataset(RLHFDataset):

    def _resolve_prompt_yaml_path(self) -> Path:
        # Precedence: hydra config -> env var -> default
        prompt_yaml_path = None
        try:
            prompt_yaml_path = self.config.get("prompt_yaml_path", None)
        except Exception:
            prompt_yaml_path = None
        prompt_yaml_path = prompt_yaml_path or os.getenv("VERL_PROMPT_YAML_PATH")

        if not prompt_yaml_path:
            return Path(__file__).resolve().parent / "swebench.yaml"

        raw_path = Path(os.path.expanduser(str(prompt_yaml_path)))
        candidates = [raw_path] if raw_path.is_absolute() else [Path.cwd() / raw_path, Path(__file__).resolve().parent / raw_path]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Prompt template file not exists: {prompt_yaml_path} (tried: {', '.join(str(c) for c in candidates)})"
        )

    def _load_yaml_templates(self) -> tuple[str, str]:
        yaml_path = self._resolve_prompt_yaml_path()
        if not yaml_path.exists():
            raise FileNotFoundError(f"Prompt template file not exists: {yaml_path}")

        self.yaml_config_path = str(yaml_path)
        with yaml_path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        agent_cfg = yaml_data.get("agent", {}) or {}
        system_template = str(agent_cfg.get("system_template", "") or "").strip()
        instance_template = str(agent_cfg.get("instance_template", "") or "").strip()

        if not system_template:
            raise ValueError(f"Missing `agent.system_template` in prompt yaml: {yaml_path}")
        if not instance_template:
            raise ValueError(f"Missing `agent.instance_template` in prompt yaml: {yaml_path}")

        return system_template, instance_template

    def _read_files_and_tokenize(self):
        self.system_template, self.instance_template = self._load_yaml_templates()
        single_data = {
            "data_source": "single_sample",
            "prompt": [
                {"role": "system", "content": self.system_template},
                {"role": "user", "content": self.instance_template}
            ],
            "ability": "ADAPTER",
            "reward_model": {"ground_truth": 0},
            "agent_name": "async_partial_tool_agent"
        }
        self.dataframe = datasets.Dataset.from_dict({
            k: [v] for k, v in single_data.items()
        })

        print(f"dataset len: {len(self.dataframe)}")

#TODO(P1)-zsl: adapt the code below
def compute_score(data_source, solution_str, ground_truth, extra_info):
    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    num_turns = extra_info["num_turns"]
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
