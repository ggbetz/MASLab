"methods/mas_base/mas_base.py"

import os
from contextvars import ContextVar
from typing import cast

# Suppress MCP server verbose logging
os.environ.setdefault("MCP_LOG_LEVEL", "ERROR")
os.environ.setdefault("FASTMCP_LOG_LEVEL", "ERROR")

# OpenAI Agents SDK imports
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    TResponseInputItem,
    set_trace_processors,
    set_tracing_disabled,
)
from agents.items import HandoffCallItem, ToolCallItem
from agents.mcp import MCPServerStdio
from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from methods.utils import handle_retry_error, load_config

# disable OpenAI tracing
set_trace_processors([])
set_tracing_disabled(True)


# Context variable for the current sample UID
_current_sample_uid: ContextVar[str | None] = ContextVar(
    "current_sample_uid", default=None
)


class MAS:
    def __init__(self, general_config, method_config_name=None):
        if method_config_name is not None:
            # Get the child class's module path
            child_module_path = os.path.dirname(
                os.path.abspath(self.__class__.__module__.replace(".", "/"))
            )
            self.method_config = load_config(
                os.path.join(child_module_path, "configs", f"{method_config_name}.yaml")
            )

        self.model_api_config = general_config["model_api_config"]
        self.model_name = general_config["model_name"]
        self.model_temperature = general_config["model_temperature"]
        self.model_max_tokens = general_config["model_max_tokens"]
        self.model_timeout = general_config["model_timeout"]

        # How strictly to enforce logical_agents configuration for agent IDs.
        # Defaults to "warn" to preserve backward-compatible behavior.
        self.agent_config_mode = general_config.get("agent_config_mode", "warn")

        # Tracking compute costs and tool usage per sample.
        # Maps sample_uid (str) -> { model_name -> stats_dict }.
        self.sample_token_stats = {}

        self.memory_bank = {}
        self.tools = {}

        # Configure a default OpenAI client for the Agents SDK based on the
        # primary model endpoint. For now we simply use the first endpoint
        # for self.model_name.

        primary_cfg = self.model_api_config[self.model_name]["model_list"][0]

        primary_base_url = primary_cfg["model_url"]
        primary_api_key = primary_cfg["api_key"]
        self.api_model_identifier = primary_cfg.get("model_name", self.model_name)

        # Debug: Log the configuration
        logger.debug(f"Using base_url: {primary_base_url}")
        logger.debug(
            f"Using model identifier: {self.model_name} ({self.api_model_identifier})"
        )
        logger.debug("MCP server logging suppressed (MCP_LOG_LEVEL=ERROR)")

        self.client = AsyncOpenAI(base_url=primary_base_url, api_key=primary_api_key)

    async def inference(self, sample):
        """Default async inference: simple single-call helper.

        Methods should override this with their own async implementations.
        """
        query = sample["query"]
        response = await self.call_llm_for_agent_async(agent_id="default", prompt=query)
        return {"response": response}

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry_error_callback=handle_retry_error,
    )
    async def call_llm(
        self,
        prompt=None,
        system_prompt=None,
        messages=None,
        model_name=None,
        temperature=None,
        sample_uid=None,
    ):
        """Async convenience wrapper that routes to the default logical agent."""

        return await self.call_llm_for_agent_async(
            agent_id="default",
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            model_name=model_name,
            temperature=temperature,
            sample_uid=sample_uid,
        )

    async def call_llm_for_agent_async(
        self,
        agent_id,
        *,
        prompt=None,
        system_prompt=None,
        messages=None,
        model_name=None,
        temperature=None,
        sample_uid=None,
        no_cache=False,
    ):
        """Async core LLM call that supports logical agents.

        Uses the OpenAI Agents SDK `Runner.run` coroutine and integrates
        with MCP servers via the async lifecycle helpers.
        """

        # Resolve model name used for selection in model_api_config
        effective_model_name = model_name if model_name is not None else self.model_name
        if effective_model_name in self.model_api_config:
            effective_api_model_identifier = self.model_api_config[
                effective_model_name
            ]["model_list"][0].get("model_name", effective_model_name)
        else:
            effective_api_model_identifier = effective_model_name

        # Normalize messages into a transcript string for the Agent input.
        if messages is None:
            assert prompt is not None, (
                "'prompt' must be provided if 'messages' is not provided."
            )
            if system_prompt is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

        # Temperature is passed via RunConfig.model_settings
        effective_temperature = (
            temperature if temperature is not None else self.model_temperature
        )

        # If no sample_uid was provided explicitly, try to read it from
        # the current sample context. This allows inference entrypoints
        # to establish per-sample isolation without plumbing the uid
        # through every call site.
        if sample_uid is None:
            sample_uid = _current_sample_uid.get(None)

        # Build a fresh Agent and any MCP servers for this call.
        agent, mcp_servers = self._build_agent_for_call(
            agent_id, effective_api_model_identifier, sample_uid=sample_uid
        )

        # Connect MCP servers for this call, if any.
        for server in mcp_servers:
            if hasattr(server, "connect"):
                try:
                    await server.connect()
                    logger.debug(
                        f"Connected MCP server: {getattr(server, 'name', 'unknown')} (agent_id={agent_id}, sample_uid={sample_uid})"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to connect MCP server {getattr(server, 'name', 'unknown')} (agent_id={agent_id}, sample_uid={sample_uid}): {e}"
                    )

        model_settings = ModelSettings(temperature=effective_temperature)
        run_config = RunConfig(model_settings=model_settings)

        try:
            response_input = cast(list[TResponseInputItem], messages)
            result = await Runner.run(agent, response_input, run_config=run_config)
        except Exception:
            # Try with a simple transcript
            transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            result = await Runner.run(agent, transcript, run_config=run_config)
        finally:
            # Always attempt to clean up MCP servers after the run.
            for server in mcp_servers:
                if hasattr(server, "cleanup"):
                    try:
                        await server.cleanup()
                        logger.debug(
                            f"Cleaned up MCP server: {getattr(server, 'name', 'unknown')} (agent_id={agent_id}, sample_uid={sample_uid})"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error cleaning up MCP server {getattr(server, 'name', 'unknown')} (agent_id={agent_id}, sample_uid={sample_uid}): {e}"
                        )

        response = result.final_output

        if not isinstance(response, str):
            raise ValueError(f"Invalid response from LLM: {response}")

        # Aggregate token usage from the Agents SDK result. Each
        # raw_response is a ModelResponse with a Usage object.
        prompt_tokens = 0
        completion_tokens = 0
        try:
            raw_responses = getattr(result, "raw_responses", []) or []
            for model_response in raw_responses:
                usage = getattr(model_response, "usage", None)
                if usage is None:
                    continue
                input_tokens = getattr(usage, "input_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
                if isinstance(input_tokens, int):
                    prompt_tokens += input_tokens
                if isinstance(output_tokens, int):
                    completion_tokens += output_tokens
        except Exception:
            # If usage is unavailable or has an unexpected shape, we
            # still want the call to succeed, so ignore errors here.
            prompt_tokens = 0
            completion_tokens = 0

        # Aggregate tool usage from the Agents SDK result. The new_items
        # list contains RunItem objects, including ToolCallItem and
        # HandoffCallItem instances for tool invocations.
        num_tool_calls = 0
        tool_calls_by_name: dict[str, int] = {}
        try:
            new_items = getattr(result, "new_items", []) or []
            for item in new_items:
                if isinstance(item, ToolCallItem):
                    raw = getattr(item, "raw_item", None)
                    tool_name = getattr(raw, "name", None) or getattr(raw, "type", None)
                    if not tool_name:
                        tool_name = type(raw).__name__ if raw is not None else "unknown"
                    num_tool_calls += 1
                    tool_calls_by_name[tool_name] = (
                        tool_calls_by_name.get(tool_name, 0) + 1
                    )
                elif isinstance(item, HandoffCallItem):
                    raw = getattr(item, "raw_item", None)
                    tool_name = getattr(raw, "name", None) or "handoff"
                    num_tool_calls += 1
                    tool_calls_by_name[tool_name] = (
                        tool_calls_by_name.get(tool_name, 0) + 1
                    )
        except Exception:
            num_tool_calls = 0
            tool_calls_by_name = {}

        # Only log stats when we have a sample_uid.
        if sample_uid is not None:
            stats_for_sample = self.sample_token_stats.setdefault(sample_uid, {})

            if effective_model_name not in stats_for_sample:
                stats_for_sample[effective_model_name] = {
                    "num_llm_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "num_tool_calls": 0,
                    "tool_calls_by_name": {},
                }

            model_stats = stats_for_sample[effective_model_name]
            model_stats["num_llm_calls"] += 1
            model_stats["prompt_tokens"] += prompt_tokens
            model_stats["completion_tokens"] += completion_tokens
            model_stats["num_tool_calls"] += num_tool_calls

            for name, count in tool_calls_by_name.items():
                current = model_stats["tool_calls_by_name"].get(name, 0)
                model_stats["tool_calls_by_name"][name] = current + count

        return response

    def _build_agent_for_call(self, agent_id, model_name, sample_uid=None):
        """Build a fresh Agent and any required MCP servers for a single call.

        This helper enforces logical_agents configuration (if present), applies
        per-agent overrides, and constructs MCP servers based on the method
        configuration. It does not cache anything; callers are responsible for
        connecting and cleaning up the returned MCP servers.

        Args:
            agent_id: Logical agent identifier (e.g., "default", "debater_0")
            model_name: Model name to use for this agent
            sample_uid: Optional sample identifier (used only for logging)

        Returns:
            Tuple of (Agent instance, list of MCP server instances)
        """

        agent_config = {}
        if hasattr(self, "method_config") and "logical_agents" in self.method_config:
            logical_agents_cfg = self.method_config["logical_agents"] or {}
            if agent_id not in logical_agents_cfg:
                msg = (
                    f"Agent ID '{agent_id}' not found in method_config.logical_agents "
                    f"for {self.__class__.__name__}; using default Agent settings instead."
                )
                mode = getattr(self, "agent_config_mode", "warn")
                if mode == "strict":
                    raise ValueError(msg)
                elif mode == "warn":
                    logger.warning(msg)
            agent_config = logical_agents_cfg.get(agent_id, {})

        # Use configured model name or default to provided model_name
        effective_agent_model = agent_config.get("model_name", model_name)

        # Use configured instructions or default
        instructions = agent_config.get("instructions", "You are a helpful assistant.")

        # Build MCP servers for this agent, if configured
        mcp_servers = []
        mcp_server_names = agent_config.get("mcp_servers", [])

        if mcp_server_names:
            if (
                not hasattr(self, "method_config")
                or "mcp_servers" not in self.method_config
            ):
                raise ValueError(
                    f"Agent '{agent_id}' references MCP servers but method_config.mcp_servers is missing."
                )

            for server_name in mcp_server_names:
                server_config = self.method_config["mcp_servers"].get(server_name)
                if not server_config:
                    raise ValueError(
                        f"MCP server '{server_name}' not found in configuration for agent '{agent_id}'."
                    )
                try:
                    server = self._create_mcp_server(server_name, server_config)
                    mcp_servers.append(server)
                except Exception as e:
                    logger.warning(
                        f"Failed to create MCP server {server_name} for agent {agent_id}: {e}"
                    )

        agent = Agent(
            name=str(agent_id),
            instructions=instructions,
            model=OpenAIChatCompletionsModel(
                model=effective_agent_model, openai_client=self.client
            ),
            mcp_servers=mcp_servers,
        )

        return agent, mcp_servers

    class _SampleContext:
        """Async context manager for per-sample isolation.

        This context manager ensures proper isolation between different samples
        by managing the routing context via a ContextVar. MCP servers and
        Agents are now fully ephemeral per call and are managed inside
        call_llm_for_agent_async.

        Entry (__aenter__):
        - Sets the ContextVar so call_llm* methods can route to the correct sample

        Exit (__aexit__):
        - Clears the ContextVar and cleans up per-sample stats
        """

        def __init__(self, mas: "MAS", sample_uid: str | None):
            self._mas = mas
            self._sample_uid = sample_uid
            self._token = None

        async def __aenter__(self):
            # Set the current sample UID for this task
            self._token = _current_sample_uid.set(self._sample_uid)
            return self._sample_uid

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # Clear the current sample UID context
            if self._token is not None:
                _current_sample_uid.reset(self._token)

            # Clean up per-sample token stats now that processing is done
            if self._sample_uid is not None:
                self._mas.sample_token_stats.pop(self._sample_uid, None)

    def sample_context(self, sample_uid: str | None):
        """Return an async context manager for the given sample UID."""

        return MAS._SampleContext(self, sample_uid)

    def _create_mcp_server(self, server_name, server_config):
        """Create an MCP server instance based on configuration.

        Args:
            server_name: Name of the MCP server
            server_config: Configuration dict with 'type', 'command', 'args', etc.

        Returns:
            MCP server instance
        """
        if server_config.get("type") == "stdio":
            # Create stdio MCP server
            command = server_config.get("command")
            args = server_config.get("args", [])

            if not command:
                raise ValueError(
                    f"stdio MCP server '{server_name}' missing 'command' in config"
                )

            server = MCPServerStdio(
                name=server_name,
                params={
                    "command": command,
                    "args": args,
                },
                cache_tools_list=True,
            )

            return server
        else:
            raise ValueError(
                f"Unsupported MCP server type: {server_config.get('type')}"
            )

    async def get_sample_stats(self, sample_uid):
        return self.sample_token_stats.pop(sample_uid, {})

    def optimizing(self, val_data):
        """For methods that requires validation data such as GPTSwarm and ADAS"""
        pass

    def retrieve_memory(self):
        pass

    def update_memory(self):
        pass

    def get_tool(self):
        pass
