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

import asyncio
import logging
import os
import traceback
from typing import Any, Optional, Dict
from uuid import uuid4
from jinja2 import Template

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from recipe.async_skyrl.agent_loop.partial_tool_agent_loop import AsyncPartialToolAgentLoop

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments import Environment, get_environment

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# --- Helper classes and functions from mini_swe_agent ---

class DefaultAgentWithReminder(DefaultAgent):
    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the output."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        remaining = self.config.step_limit - self.model.n_calls

        if remaining == 1:
            observation = f"{observation}\nREMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            observation = f"{observation}\nREMINDER: You have {remaining} turns left to arrive at the solution."

        self.add_message("user", observation)
        return output

def get_docker_image_name(instance: dict, data_source: str) -> str:
    """Get the image name for a SWEBench/SWE-Gym instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        iid = instance["instance_id"]
        if "swe-gym" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_s_")  # to comply with docker image naming convention
            image_name = f"docker.io/xingyaoww/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        elif "swe-bench" in data_source.lower():
            # Docker doesn't allow double underscore, so we replace them with a magic token
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        else:
            # Fallback or default behavior
             image_name = f"docker.io/swebench/sweb.eval.x86_64.{iid}:latest".lower()
    return image_name

def get_sb_environment(config: dict, instance: dict, data_source: str) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_docker_image_name(instance, data_source)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = f"docker://{image_name}"
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env

# --- End helper classes ---

@register("mini_swe_agent")
class MiniSweAgentLoop(AsyncPartialToolAgentLoop):
    def __init__(self, trainer_config, **kwargs):
        super().__init__(trainer_config, **kwargs)
        self.active_agents: Dict[str, DefaultAgentWithReminder] = {}
        # Assuming sweagent_config is passed in trainer_config or we need to load it
        # For now, we'll look for it in trainer_config.config.sweagent_config
        self.sweagent_config = trainer_config.config.get("sweagent_config", {})
        self.data_source = trainer_config.config.get("data_source", "swe-bench")

    async def _init_agent_data(self, kwargs: dict, param_version: int) -> AgentData:
        # We need to initialize the MiniSweAgent here
        # kwargs["extra_info"] should contain the instance data
        extra_info = kwargs.get("extra_info", {})
        instance = extra_info.get("instance", {})
        if not instance:
             # Fallback: try to parse from raw_prompt if it's a dict, or fail
             if isinstance(kwargs.get("raw_prompt"), dict):
                 instance = kwargs["raw_prompt"]
             else:
                 logger.warning("No instance data found in extra_info or raw_prompt. MiniSweAgent might fail.")

        request_id = uuid4().hex
        
        # Initialize Environment and Agent
        # We need a dummy model for DefaultAgent because we will drive it externally
        # But DefaultAgent expects a model object. We can pass a dummy or mock.
        # Actually DefaultAgent uses model to count calls and maybe other things.
        # We can pass a simple object with n_calls attribute.
        
        class DummyModel:
            def __init__(self):
                self.n_calls = 0
                self.cost = 0.0
        
        model = DummyModel()
        
        try:
            env = get_sb_environment(self.sweagent_config, instance, self.data_source)
            agent = DefaultAgentWithReminder(model, env, **self.sweagent_config.get("agent", {}))
            
            # Initialize agent with problem statement
            # DefaultAgent.run() does this:
            # self.history.append({"role": "system", "content": self.system_prompt})
            # self.history.append({"role": "user", "content": problem_statement})
            # We need to replicate initialization
            
            problem_statement = instance.get("problem_statement", "")
            # We can manually set up the history
            # DefaultAgent doesn't have explicit init method for history other than what's in run()
            # But we can use add_message
            
            # Reset/Init logic from DefaultAgent.run (simplified)
            agent.history = []
            agent.add_message("system", agent.system_prompt)
            agent.add_message("user", problem_statement)
            
            self.active_agents[request_id] = agent
            
        except Exception as e:
            logger.error(f"Failed to initialize MiniSweAgent: {e}")
            raise e

        # Initialize AgentData
        # We use agent.history to populate messages
        messages = agent.history
        
        agent_data = AgentData(
            messages=messages,
            image_data=None,
            metrics={},
            request_id=request_id,
            tools_kwargs={},
        )
        agent_data.extra_fields["agent_request_id"] = request_id
        
        return agent_data

    def _restore_from_output(self, output: AgentLoopOutput):
        agent_data, state = super()._restore_from_output(output)
        request_id = agent_data.extra_fields.get("agent_request_id")
        if request_id and request_id in self.active_agents:
            # Agent is already active
            pass
        else:
            # If we can't restore the agent (e.g. worker restart), we are in trouble.
            # For now, we assume sticky sessions and no restarts.
            logger.warning(f"Could not find active agent for request_id {request_id}. Resuming might fail if environment is needed.")
        return agent_data, state

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        request_id = agent_data.extra_fields.get("agent_request_id")
        agent = self.active_agents.get(request_id)
        
        if not agent:
             raise RuntimeError("Agent not found in _handle_pending_state")

        # Get prompt from agent
        # DefaultAgent.get_prompt() returns the formatted prompt string or messages
        # We assume we want to send this to the LLM
        
        # Note: DefaultAgent.get_prompt() might return a string if it formats it, 
        # or we might need to use agent.history directly.
        # DefaultAgent usually formats history into a string based on templates.
        
        # If we use a chat model in verl, we might want list of messages.
        # If DefaultAgent manages the prompt format (e.g. ReAct style in one string), we should use that.
        
        # Let's check DefaultAgent.get_prompt implementation in minisweagent if possible.
        # Assuming it returns what needs to be fed to the model.
        
        # If we are using a VLLM server that expects chat messages (list of dicts), 
        # and DefaultAgent returns a string, we might need to wrap it.
        
        # However, ToolAgentLoop usually expects `agent_data.messages` to be list of dicts 
        # and uses `processor.apply_chat_template`.
        
        # If `mini_swe_agent` does its own formatting, we might want to bypass `apply_chat_template` 
        # or pass the pre-formatted string.
        
        # For simplicity, let's assume we use `agent.history` as the messages 
        # and let `ToolAgentLoop`'s `_handle_pending_state` logic (via super or copied) handle tokenization.
        
        # But wait, `DefaultAgent` might add "User: ... Assistant: ..." formatting inside the content.
        # If `minisweagent` is designed for completion models, `get_prompt()` returns a string.
        # If it's for chat models, it might rely on the model wrapper to format.
        
        # In `mini_swe_generator.py`, `DefaultAgentWithReminder` is used with `get_model(litellm_model_name, ...)`.
        # `DefaultAgent` calls `self.model(self.get_prompt())`.
        
        # We will use `agent.history` as the source of truth for messages.
        agent_data.messages = agent.history
        
        # We call super()._handle_pending_state to do tokenization
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        
        # Call super to generate response
        # This will populate agent_data.response_ids
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination=True)
        
        if state == AgentState.TERMINATED:
             # If super thinks it's terminated (e.g. length limit), we should respect it?
             # Or we check if we can parse an action.
             pass

        request_id = agent_data.extra_fields.get("agent_request_id")
        agent = self.active_agents.get(request_id)
        
        # Decode response
        response_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        
        # Update agent's model n_calls (dummy)
        agent.model.n_calls += 1
        
        # Add assistant message to history
        # DefaultAgent expects the response to be added
        agent.add_message("assistant", response_text)
        
        # Parse action
        # DefaultAgent.parse_action(response)
        # response can be string or dict depending on model. 
        # Here we have text.
        try:
            action = agent.parse_action(response_text)
            # If action is found, we go to PROCESSING_TOOLS
            # DefaultAgent.parse_action usually returns a dict or raises/returns None?
            # Actually DefaultAgent.parse_action implementation depends on the agent config.
            # If it fails to parse, it might return something that execute_action handles (e.g. "Invalid action")
            
            # We assume if we got a response, we try to execute it.
            # Unless the agent decides it's done.
            # DefaultAgent checks for "submit" action to finish.
            
            # We'll transition to PROCESSING_TOOLS to execute the action (even if it's submit)
            return AgentState.PROCESSING_TOOLS
            
        except Exception as e:
            logger.warning(f"Failed to parse action: {e}")
            # If parsing fails, maybe we should let the agent know?
            # Or maybe it's just a thought step?
            # If mini_swe_agent expects every step to be an action, then this is an error.
            # We'll assume we go to PROCESSING_TOOLS to let execute_action handle it (it might report error to agent)
            return AgentState.PROCESSING_TOOLS

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        request_id = agent_data.extra_fields.get("agent_request_id")
        agent = self.active_agents.get(request_id)
        
        # We need the last response to execute
        # agent.history[-1] is the assistant message we just added
        response = agent.history[-1]["content"]
        
        try:
            # execute_action expects the parsed action
            # We parsed it in generating state but didn't store it.
            # We can parse again.
            action = agent.parse_action(response)
            
            # Check if it is submission (termination)
            # DefaultAgent usually has a way to check if it's done.
            # But execute_action usually handles "submit".
            # If "submit", execute_action might return the result and we should terminate.
            
            # In `mini_swe_generator.py`:
            # output = self.execute_action(self.parse_action(response))
            # observation = self.render_template(..., output=output)
            
            # We need to run this in executor because it involves subprocess
            def run_step():
                output = agent.execute_action(action)
                observation = agent.render_template(agent.config.action_observation_template, output=output)
                
                # Reminder logic
                remaining = agent.config.step_limit - agent.model.n_calls
                if remaining == 1:
                    observation = f"{observation}\nREMINDER: You only have 1 turn left. Please provide the final answer"
                elif remaining > 1:
                    observation = f"{observation}\nREMINDER: You have {remaining} turns left to arrive at the solution."
                
                agent.add_message("user", observation)
                return output, observation

            output, observation = await self.loop.run_in_executor(None, run_step)
            
            # Check for termination
            # If the action was "submit", `agent.should_terminate` might be true?
            # DefaultAgent doesn't seem to have `should_terminate` flag exposed easily.
            # But `execute_action` for submit usually returns something specific.
            
            # In `minisweagent`, `agent.run` loop breaks if `exit_status` is set.
            # `exit_status` is returned by `agent.run`.
            # `agent.run` calls `step()`. `step()` returns `exit_status`.
            # `step()` logic:
            #   action = parse_action(response)
            #   output = execute_action(action)
            #   if action['name'] == 'submit': return 'submitted'
            
            # We need to check if action was submit.
            # We can check `action` content.
            if isinstance(action, dict) and action.get("name") == "submit":
                return AgentState.TERMINATED
            
            # Also check step limit
            if agent.model.n_calls >= agent.config.step_limit:
                return AgentState.TERMINATED

            # If not terminated, we go back to PENDING to generate next prompt
            return AgentState.PENDING

        except Exception as e:
            logger.error(f"Error in processing tools: {e}")
            # If error, we might want to terminate or add error message
            agent.add_message("user", f"Error executing action: {e}")
            return AgentState.PENDING

    def _build_completed_output(self, agent_data: AgentData, param_version: int) -> AgentLoopOutput:
        # We need to return the result
        # For SWE-bench, we might want to return the patch or the submission.
        # The `agent.history` contains the trajectory.
        
        # We can return the full history.
        return AgentLoopOutput(
            token_ids=[], # We don't return a single sequence of token ids
            log_probs=[],
            extra_fields={
                "messages": agent_data.messages,
                "is_cancel": False
            }
        )

    def _build_cancelled_output(self, agent_data: AgentData, state: AgentState) -> AgentLoopOutput:
        return AgentLoopOutput(
            token_ids=[],
            log_probs=[],
            extra_fields={
                "messages": agent_data.messages,
                "is_cancel": True,
                "agent_request_id": agent_data.extra_fields.get("agent_request_id")
            }
        )

