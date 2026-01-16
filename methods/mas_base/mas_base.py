"methods/mas_base/mas_base.py"

import os
from contextvars import ContextVar

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
    set_trace_processors,
)
from agents.items import HandoffCallItem, ToolCallItem
from agents.mcp import MCPServerStdio
from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from methods.utils import handle_retry_error, load_config

set_trace_processors([])  # disable OpenAI tracing


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

        # Logical agent registry (agent_id -> backend-specific state)
        self._logical_agents = {}

        # MCP server registry per agent_id and sample_uid:
        # {agent_id: {sample_uid: {server_name: MCP server instance}}}
        self._mcp_servers = {}

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

        # Build a simple transcript; later this can be replaced with
        # structured items if needed.
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

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

        # Get or create a persistent logical agent. This will also
        # synchronously create any MCP server objects referenced in the
        # method configuration, but it will NOT start them yet.
        agent = self._get_or_create_logical_agent(
            agent_id, effective_api_model_identifier, sample_uid=sample_uid
        )

        model_settings = ModelSettings(temperature=effective_temperature)
        run_config = RunConfig(model_settings=model_settings)

        result = await Runner.run(agent, transcript, run_config=run_config)
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

    def _get_or_create_logical_agent(self, agent_id, model_name, sample_uid=None):
        """Get or create a persistent logical agent for the given agent_id, model, and sample.

        Args:
            agent_id: Logical agent identifier (e.g., "default", "debater_0")
            model_name: Model name to use for this agent

        Returns:
            Agent instance from the OpenAI Agents SDK
        """
        # For MCP-enabled methods, distinguish agents by sample_uid as well
        uses_mcp = (
            hasattr(self, "method_config")
            and "logical_agents" in self.method_config
            and "mcp_servers" in self.method_config
        )

        # If a logical_agents block is present, optionally enforce that the
        # requested agent_id is configured there. This avoids silent fallback
        # to default settings when an agent ID is mistyped or missing.
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

        if uses_mcp:
            cache_key = f"{agent_id}:{model_name}:{sample_uid}"
        else:
            cache_key = f"{agent_id}:{model_name}"

        if cache_key not in self._logical_agents:
            # Check if there's a specific configuration for this agent
            agent_config = {}
            if (
                hasattr(self, "method_config")
                and "logical_agents" in self.method_config
            ):
                agent_config = self.method_config["logical_agents"].get(agent_id, {})

            # Use configured model name or default to provided model_name
            effective_agent_model = agent_config.get("model_name", model_name)

            # Use configured instructions or default
            instructions = agent_config.get(
                "instructions", "You are a helpful assistant."
            )

            # Get MCP servers for this agent (synchronously create, will be started async)
            mcp_server_names = agent_config.get("mcp_servers", [])
            mcp_servers = []

            if mcp_server_names:
                try:
                    mcp_servers = self._get_mcp_servers(
                        agent_id, sample_uid, mcp_server_names
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create MCP servers for agent {agent_id}: {e}"
                    )
                    mcp_servers = []

            # Create the agent with MCP servers
            agent = Agent(
                name=str(agent_id),
                instructions=instructions,
                model=OpenAIChatCompletionsModel(
                    model=effective_agent_model, openai_client=self.client
                ),
                mcp_servers=mcp_servers,
            )

            # Cache the agent
            self._logical_agents[cache_key] = agent

        return self._logical_agents[cache_key]

    async def _connect_mcp_servers_for_sample(self, sample_uid: str | None) -> None:
        """Start and register all MCP servers for the given sample UID.

        This method iterates through all MCP servers associated with the
        sample_uid and starts them. Called from SampleContext.__aenter__.
        """
        if sample_uid is None:
            return
        # Look up all servers for this sample_uid and connect them if
        # needed. We do not maintain a separate "started" set; we assume
        # that connecting twice is either a no-op or guarded by the
        # underlying implementation.
        for agent_id, per_agent in self._mcp_servers.items():
            per_sample = per_agent.get(sample_uid)
            if not per_sample:
                continue
            for server_name, server in per_sample.items():
                if hasattr(server, "connect"):
                    try:
                        await server.connect()
                        logger.debug(
                            f"Started MCP server: {server_name} (agent_id={agent_id}, sample_uid={sample_uid})"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to start MCP server {server_name} (agent_id={agent_id}, sample_uid={sample_uid}): {e}"
                        )

    async def _cleanup_mcp_servers_for_sample(self, sample_uid: str | None) -> None:
        """Shutdown and cleanup all MCP servers for the given sample UID.

        This method stops all MCP servers associated with the sample_uid
        and removes the corresponding logical agents from cache. Called from
        SampleContext.__aexit__.
        """
        if sample_uid is None:
            return

        # Stop and remove MCP servers for this sample_uid
        for agent_id, per_agent in list(self._mcp_servers.items()):
            if sample_uid not in per_agent:
                continue
            per_sample = per_agent[sample_uid]
            for server_name, server in list(per_sample.items()):
                if hasattr(server, "cleanup"):
                    try:
                        await server.cleanup()
                        logger.debug(
                            f"Stopped MCP server: {server_name} (agent_id={agent_id}, sample_uid={sample_uid})"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error stopping MCP server {server_name} (agent_id={agent_id}, sample_uid={sample_uid}): {e}"
                        )
                per_sample.pop(server_name, None)
            if not per_sample:
                per_agent.pop(sample_uid, None)
            if not per_agent:
                self._mcp_servers.pop(agent_id, None)

        # Remove any logical agents keyed with this sample_uid
        keys_to_remove: list[str] = []
        suffix = f":{sample_uid}"
        for key in self._logical_agents.keys():
            if key.endswith(suffix):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self._logical_agents.pop(key, None)

    class _SampleContext:
        """Async context manager for per-sample isolation and MCP server lifecycle.

        This context manager ensures proper isolation between different samples
        by managing MCP servers and routing context. For a given sample_uid:

        Entry (__aenter__):
        - Sets the ContextVar so call_llm* methods can route to the correct sample
        - Proactively creates all MCP servers that might be needed for this sample
        - Starts and connects all MCP servers for this sample

        Exit (__aexit__):
        - Shuts down all MCP servers associated with this sample
        - Cleans up resources and resets the context
        """

        def __init__(self, mas: "MAS", sample_uid: str | None):
            self._mas = mas
            self._sample_uid = sample_uid
            self._token = None

        async def __aenter__(self):
            # Set the current sample UID for this task
            self._token = _current_sample_uid.set(self._sample_uid)

            # Proactively create all MCP servers that might be needed for this sample
            self._mas._create_mcp_servers_for_sample(self._sample_uid)

            # Start and connect all MCP servers for this sample
            await self._mas._connect_mcp_servers_for_sample(self._sample_uid)

            return self._sample_uid

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # Shutdown all MCP servers for this sample
            try:
                await self._mas._cleanup_mcp_servers_for_sample(self._sample_uid)
            finally:
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

    def _get_mcp_servers(self, agent_id, sample_uid, server_names):
        """Get MCP servers for a given agent_id and sample_uid.


        Args:
            agent_id: Logical agent identifier
            sample_uid: Sample identifier used to isolate servers
            server_names: List of MCP server names to retrieve

        Returns:
            List of MCP server instances
        """
        servers = []

        # Get or create nested maps for this agent and sample
        per_agent = self._mcp_servers.setdefault(agent_id, {})
        per_sample = per_agent.setdefault(sample_uid, {})

        for server_name in server_names:
            server = per_sample.get(server_name)
            if server is None:
                raise ValueError(
                    f"MCP server '{server_name}' not found for agent '{agent_id}' "
                    f"and sample '{sample_uid}'. Servers must be created by "
                    f"SampleContext before agent creation."
                )
            servers.append(server)

        return servers

    def _get_all_mcp_servers_for_sample(
        self, sample_uid: str | None
    ) -> dict[str, dict[str, list[str]]]:
        """Get all MCP servers needed for all agents in the method config.

        Returns:
            Dict mapping agent_id to list of server_names needed for that agent
        """
        agent_servers = {}

        if (
            not hasattr(self, "method_config")
            or "logical_agents" not in self.method_config
        ):
            return agent_servers

        for agent_id, agent_config in self.method_config["logical_agents"].items():
            server_names = agent_config.get("mcp_servers", [])
            if server_names:
                agent_servers[agent_id] = server_names

        return agent_servers

    def _create_mcp_servers_for_sample(self, sample_uid: str | None) -> None:
        """Create all MCP servers needed for the given sample.

        This method proactively creates all MCP servers that might be needed
        by any agent for this sample, based on the method configuration.
        """
        if sample_uid is None:
            return

        agent_servers = self._get_all_mcp_servers_for_sample(sample_uid)

        for agent_id, server_names in agent_servers.items():
            # Get or create nested maps for this agent and sample
            per_agent = self._mcp_servers.setdefault(agent_id, {})
            per_sample = per_agent.setdefault(sample_uid, {})

            for server_name in server_names:
                if server_name not in per_sample:
                    # Get server configuration
                    server_config = self.method_config["mcp_servers"].get(server_name)
                    if not server_config:
                        raise ValueError(
                            f"MCP server '{server_name}' not found in configuration"
                        )

                    # Create and cache the server for this agent+sample
                    server = self._create_mcp_server(server_name, server_config)
                    per_sample[server_name] = server

    def optimizing(self, val_data):
        """For methods that requires validation data such as GPTSwarm and ADAS"""
        pass

    def retrieve_memory(self):
        pass

    def update_memory(self):
        pass

    def get_tool(self):
        pass
