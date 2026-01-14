import asyncio
import os

# OpenAI Agents SDK imports
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    set_trace_processors,
)
from agents.mcp import MCPServerStdio
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from methods.utils import handle_retry_error, load_config

set_trace_processors([])  # disable OpenAI tracing


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

        # Tracking compute costs
        self.token_stats = {
            self.model_name: {
                "num_llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        }

        self.memory_bank = {}
        self.tools = {}

        # Logical agent registry (agent_id -> backend-specific state)
        self._logical_agents = {}

        # MCP server registry (server_name -> MCP server instance)
        self._mcp_servers = {}

        # Configure a default OpenAI client for the Agents SDK based on the
        # primary model endpoint. For now we simply use the first endpoint
        # for self.model_name.
        primary_cfg = self.model_api_config[self.model_name]["model_list"][0]
        primary_base_url = primary_cfg["model_url"]
        primary_api_key = primary_cfg["api_key"]

        # Debug: Print the configuration (can be removed in production)
        print(f"[DEBUG] Using base_url: {primary_base_url}")
        print(f"[DEBUG] Using model: {self.model_name}")

        self.client = AsyncOpenAI(base_url=primary_base_url, api_key=primary_api_key)

        # Flag to track if MCP servers have been started
        self._mcp_servers_started = False

        # Lock used to serialize MCP server startup in async code paths
        self._mcp_start_lock = None

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
    ):
        """Async convenience wrapper that routes to the default logical agent."""

        return await self.call_llm_for_agent_async(
            agent_id="default",
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            model_name=model_name,
            temperature=temperature,
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
    ):
        """Async core LLM call that supports logical agents.

        Uses the OpenAI Agents SDK `Runner.run` coroutine and integrates
        with MCP servers via the async lifecycle helpers.
        """

        # Resolve model name used for selection in model_api_config
        effective_model_name = model_name if model_name is not None else self.model_name

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

        # Get or create a persistent logical agent. This will also
        # synchronously create any MCP server objects referenced in the
        # method configuration, but it will NOT start them yet.
        agent = self._get_or_create_logical_agent(agent_id, effective_model_name)

        # Ensure MCP servers are started before using agents.
        await self._ensure_mcp_servers_started()

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

        if effective_model_name not in self.token_stats:
            self.token_stats[effective_model_name] = {
                "num_llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        self.token_stats[effective_model_name]["num_llm_calls"] += 1
        self.token_stats[effective_model_name]["prompt_tokens"] += prompt_tokens
        self.token_stats[effective_model_name]["completion_tokens"] += completion_tokens

        return response

    def _get_or_create_logical_agent(self, agent_id, model_name):
        """Get or create a persistent logical agent for the given agent_id and model.

        Args:
            agent_id: Logical agent identifier (e.g., "default", "debater_0")
            model_name: Model name to use for this agent

        Returns:
            Agent instance from the OpenAI Agents SDK
        """
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
                    mcp_servers = self._get_or_create_mcp_servers(mcp_server_names)
                except Exception as e:
                    print(
                        f"Warning: Failed to create MCP servers for agent {agent_id}: {e}"
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

    def _get_or_create_mcp_servers(self, server_names):
        """Get or create MCP servers for a given list of server names.

        Args:
            server_names: List of MCP server names to create/retrieve

        Returns:
            List of MCP server instances
        """
        servers = []

        if (
            not hasattr(self, "method_config")
            or "mcp_servers" not in self.method_config
        ):
            return servers

        for server_name in server_names:
            if server_name not in self._mcp_servers:
                # Get server configuration
                server_config = self.method_config["mcp_servers"].get(server_name)
                if not server_config:
                    raise ValueError(
                        f"MCP server '{server_name}' not found in configuration"
                    )

                # Create and cache the server
                server = self._create_mcp_server(server_name, server_config)
                self._mcp_servers[server_name] = server

            servers.append(self._mcp_servers[server_name])

        return servers

    async def _ensure_mcp_servers_started(self):
        """Ensure MCP servers are started exactly once, in a serialized way."""
        # Fast path: already started or no servers configured
        if self._mcp_servers_started or not self._mcp_servers:
            return

        # Lazily create the lock the first time we need it, so we only
        # construct it when an event loop is running.
        if self._mcp_start_lock is None:
            self._mcp_start_lock = asyncio.Lock()

        async with self._mcp_start_lock:
            # Double-check inside the lock in case another task started them
            if self._mcp_servers_started or not self._mcp_servers:
                return

            await self._start_mcp_servers()
            self._mcp_servers_started = True

    async def _start_mcp_servers(self):
        """Start all MCP servers that have been created.

        We call the Agents SDK `connect()` coroutine directly rather than
        using `__aenter__` so that the internal async context managers
        (e.g., stdio_client) are always entered from the same task that
        will later call `cleanup()`.
        """
        for server_name, server in self._mcp_servers.items():
            if hasattr(server, "connect"):
                try:
                    await server.connect()
                    print(f"Started MCP server: {server_name}")
                except Exception as e:
                    print(f"Failed to start MCP server {server_name}: {e}")

    async def _stop_mcp_servers(self):
        """Stop all MCP servers that have been started.

        We use the Agents SDK `cleanup()` coroutine instead of
        `__aexit__` for the same reason as `_start_mcp_servers`.
        """
        for server_name, server in self._mcp_servers.items():
            if hasattr(server, "cleanup"):
                try:
                    await server.cleanup()
                    print(f"Stopped MCP server: {server_name}")
                except Exception as e:
                    print(f"Error stopping MCP server {server_name}: {e}")

    async def __aenter__(self):
        """Async context manager entry - start MCP servers.

        This path is for advanced async usage where the caller controls
        the event loop. MCP servers will be connected using the current
        loop and cleaned up on exit.
        """
        await self._ensure_mcp_servers_started()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - stop MCP servers."""
        if self._mcp_servers_started and self._mcp_servers:
            await self._stop_mcp_servers()
            self._mcp_servers_started = False

    def shutdown_mcp_servers(self):
        """Synchronously stop all MCP servers if they were started.

        This is intended for synchronous entrypoints like `inference.py`
        that call `call_llm` directly. It ensures that MCP servers are
        cleaned up on the same event loop that started them, avoiding
        anyio stdio_client shutdown warnings.
        """
        if not self._mcp_servers_started:
            return
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(self._stop_mcp_servers())
                self._mcp_servers_started = False
        except Exception as e:
            print(f"Warning: Failed to shutdown MCP servers: {e}")

    def get_token_stats(self):
        return self.token_stats

    def optimizing(self, val_data):
        """For methods that requires validation data such as GPTSwarm and ADAS"""
        pass

    def retrieve_memory(self):
        pass

    def update_memory(self):
        pass

    def get_tool(self):
        pass
