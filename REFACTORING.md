REFACTORING.md (temporary design: logical agents + MCP, no internal sessions)**

**Goal:** define a new `MAS.call_llm_for_agent` API and show how to adapt `LLM_Debate_Main` to use persistent logical Agents with optional MCP tools, without using Agents SDK sessions and without heavily rewriting existing method logic.

---

**1. Scope and goals**

- Introduce a **logical agent** abstraction on top of `MAS`:
  - A logical agent is identified by an `agent_id` string (e.g., `"default"`, `"debater_0"`, `"critic"`).
  - Each logical agent corresponds to:
    - A persistent OpenAI Agents **`Agent`** instance.
    - Zero or more persistent **MCP servers** attached to that Agent.
- Add a new API:

  ```python
  def call_llm_for_agent(...): ...
  ```

  and make existing `call_llm` a thin wrapper that delegates to `agent_id="default"`.

- Keep **methods’ existing context management**:
  - Methods continue to construct `messages` / `agent_context` lists.
  - Each `call_llm_for_agent` call receives the full context for that turn via `messages`.
- **Do not use Agents `Session`s (for now)**:
  - No internal conversation memory inside the SDK.
  - “Session” is still fully managed by the methods via `messages` lists.

This allows:

- Persistent MCP tools and per-role Agents *within* a method (e.g., LLM debate participants).
- Minimal change in method code: mostly replacing `self.call_llm(...)` with `self.call_llm_for_agent(agent_id, ...)`.

---

**2. `call_llm_for_agent` API**

**Signature (Python)**

```python
def call_llm_for_agent(
    self,
    agent_id: str,
    *,
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
) -> str:
    ...
```

- `agent_id`:
  - Stable identifier for the logical agent (e.g., `"default"`, `"debater_0"`).
  - Determines which persistent `Agent` + MCP servers to use.
- `prompt`, `system_prompt`, `messages`:
  - Exactly the same semantics as current `MAS.call_llm`:
    - If `messages` is provided, it represents the full context to send.
    - If `messages` is `None`, `prompt` must be provided; `system_prompt` is optional.
- `model_name`:
  - Optional override; if `None`, falls back to `self.model_name`.
- `temperature`:
  - Optional override; if `None`, falls back to `self.model_temperature`.

**Return value**

- Returns the model’s final answer as a `str`.
- This mirrors the current `call_llm` behavior so methods can treat it identically.

**Error behavior**

- If both `messages` and `prompt` are `None`: raise `AssertionError` (same as current code).
- If `Runner.run` fails or returns a non-string final output:
  - Raise `ValueError` with a useful message (mirroring the current “Invalid response from LLM” check).

---

**3. Internal behavior of `call_llm_for_agent`**

High-level algorithm:

1. **Normalize arguments**

   - If `messages` is `None`, construct MAS-style messages:

     ```python
     assert prompt is not None
     if system_prompt is not None:
         messages = [
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": prompt},
         ]
     else:
         messages = [{"role": "user", "content": prompt}]
     ```

   - `model_name_effective = model_name or self.model_name`
   - `temperature_effective = temperature or self.model_temperature`

2. **Retrieve or create the logical agent**

   - Call a helper:

     ```python
     logical_state = self._get_or_create_logical_agent(
         agent_id=agent_id,
         model_name=model_name_effective,
     )
     ```

   - `logical_state` holds:
     - `logical_state.agent` — an `Agent` instance from the Agents SDK.
     - `logical_state.mcp_servers` — list of MCP servers attached to that agent (if any).
   - On first call for a given `(agent_id, model_name_effective)`:
     - Determine MCP config and instructions from `self.method_config` (see section 5).
     - Instantiate MCP servers as needed (lifecycle specifics handled in `MAS`; details out-of-scope for this doc).
     - Construct the Agents SDK `Agent`:

       ```python
       agent = Agent(
           name=agent_id,
           instructions=instructions_from_config_or_default,
           model=model_name_effective,
           mcp_servers=servers_for_this_agent,
       )
       ```

     - Store in `self._logical_agents[agent_id]`.

3. **Map MAS messages to a Runner input**

   For the initial implementation (simple and robust):

   - Convert MAS messages to a single prompt string:

     ```python
     # Example; exact format can be tuned later
     transcript = "\n".join(
         f"{m['role']}: {m['content']}" for m in messages
     )
     input_text = transcript
     ```

   - This preserves the full context that methods already craft.
   - Future improvement: use `agents.items` to represent role-structured messages instead of a flat string.

4. **Call the Agent via `Runner`**

   - Use the Agents SDK:

     ```python
     from agents import Runner, RunConfig

     run_config = RunConfig(
         model_settings={"temperature": temperature_effective},
         # potential future: model_provider based on self.model_api_config
     )

     result = Runner.run_sync(
         logical_state.agent,
         input_text,
         run_config=run_config,
     )
     ```

   - Extract the final output:

     ```python
     response = result.final_output
     ```

   - For now, we assume `response` is `str` for typical assistant agents.

5. **Token accounting**

   - The legacy `MAS.call_llm` tracks:

     ```python
     self.token_stats[model_name] = {
         "num_llm_calls": ...,
         "prompt_tokens": ...,
         "completion_tokens": ...,
     }
     ```

   - Initial implementation:
     - Increment `num_llm_calls` on each successful `call_llm_for_agent`.
     - Optionally, log token usage if available from `result`.
     - Later, once we know how to get usage from the Agents API, we can fill `prompt_tokens` and `completion_tokens`.

6. **Return**

   - Validate and return:

     ```python
     if not isinstance(response, str):
         raise ValueError(f"Invalid response from LLM: {response}")
     return response
     ```

---

**4. Compatibility wrapper: `call_llm`**

To minimize churn across methods, keep a thin wrapper:

```python
def call_llm(
    self,
    prompt=None,
    system_prompt=None,
    messages=None,
    model_name=None,
    temperature=None,
):
    return self.call_llm_for_agent(
        agent_id="default",
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
        model_name=model_name,
        temperature=temperature,
    )
```

- All existing methods can continue to call `self.call_llm(...)` and behave as before, but with a single logical agent `"default"`.
- More complex methods can explicitly use per-role `agent_id`s where needed.

---

**5. Configuring logical agents + MCP (conceptual)**

We will centralize per-role configuration in each method’s YAML config. For LLM-Debate (`LLM_Debate_Main`), for example:

```yaml
agents_num: 3
rounds_num: 4

# Shared MCP server definitions
mcp_servers:
  filesystem:
    type: "stdio"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "./samples"]
  custom_tools:
    type: "stdio"
    command: "my-mcp-server"
    args: []

# Per-logical-agent configuration
logical_agents:
  debater_0:
    role: "proposition"
    model_name: "gpt-4o-mini-2024-07-18"
    mcp_servers:
      - "filesystem"
  debater_1:
    role: "opposition"
    model_name: "gpt-4o-mini-2024-07-18"
    mcp_servers: []
  debater_2:
    role: "moderator"
    model_name: "gpt-4o-mini-2024-07-18"
    mcp_servers:
      - "custom_tools"
```

`_get_or_create_logical_agent` will:

- Look up `agent_cfg = self.method_config.get("logical_agents", {}).get(agent_id, {})`.
- Use `agent_cfg["model_name"]` if present, else default to `self.model_name`.
- Resolve MCP server references in `agent_cfg["mcp_servers"]` using the top-level `mcp_servers` map.
  - For each name in `agent_cfg["mcp_servers"]`, fetch `server_cfg = self.method_config["mcp_servers"][name]` and instantiate the server accordingly.
- Attach the instantiated MCP server objects to the `Agent`.

The exact MCP lifecycle (event loop, `async with`, etc.) is handled privately inside `MAS` and is orthogonal to the `call_llm_for_agent` public API.

---

**6. Refactoring `LLM_Debate_Main`**

**Original (simplified)**

```python
class LLM_Debate_Main(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        self.agents_num = self.method_config["agents_num"]
        self.rounds_num = self.method_config["rounds_num"]
    
    def inference(self, sample):

        query = sample["query"]

        agent_contexts = [[{"role": "user", "content": f"""{query} Make sure to state your answer at the end of the response."""}] for agent in range(self.agents_num)]

        for round in range(self.rounds_num):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = self.construct_message(agent_contexts_other, query, 2*round - 1)
                    agent_context.append(message)

                response = self.call_llm(messages=agent_context)
                agent_context.append({"role": "assistant", "content": response})
        
        answers = [agent_context[-1]['content'] for agent_context in agent_contexts]
        
        final_answer = self.aggregate(query, answers)
        return {"response": final_answer}
```

**Refactored to use logical agents (no sessions)**

```python
class LLM_Debate_Main(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)

        self.agents_num = self.method_config["agents_num"]
        self.rounds_num = self.method_config["rounds_num"]
    
    def inference(self, sample):
        query = sample["query"]

        # One context list per debater, as before
        agent_contexts = [
            [
                {
                    "role": "user",
                    "content": f"""{query} Make sure to state your answer at the end of the response."""
                }
            ]
            for _ in range(self.agents_num)
        ]

        for round_idx in range(self.rounds_num):
            for i, agent_context in enumerate(agent_contexts):
                if round_idx != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = self.construct_message(
                        agent_contexts_other,
                        query,
                        2 * round_idx - 1,
                    )
                    agent_context.append(message)

                # NEW: use a per-debater logical agent id
                agent_id = f"debater_{i}"
                response = self.call_llm_for_agent(
                    agent_id=agent_id,
                    messages=agent_context,
                )

                agent_context.append({"role": "assistant", "content": response})
        
        answers = [ctx[-1]["content"] for ctx in agent_contexts]
        final_answer = self.aggregate(query, answers)
        return {"response": final_answer}
```

Key points:

- We did **not** change how `agent_contexts` are constructed or updated:
  - Each debater still sees the full message history we build.
- The only API change is from:

  ```python
  response = self.call_llm(messages=agent_context)
  ```

  to:

  ```python
  response = self.call_llm_for_agent(agent_id=f"debater_{i}", messages=agent_context)
  ```

- Internally, `call_llm_for_agent` uses a persistent `Agent` per `agent_id`.
  - MCP servers configured for `debater_0`, `debater_1`, etc. are persistent for the entire lifetime of this `LLM_Debate_Main` instance.
- No Agents SDK `Session` is used; persistence of **context** is still via `agent_context`.

---

**7. Summary**

- `call_llm_for_agent` provides a unified, backward-compatible way to:
  - Introduce persistent, per-role Agents + MCP servers behind existing MAS methods.
  - Keep methods’ custom multi-agent patterns and `messages` construction intact.
- `LLM_Debate_Main` is a minimal demonstration:
  - One logical agent per debater (`debater_0`, `debater_1`, ...).
  - Each call uses that debater’s persistent Agent + tools, while preserving MASLab’s current debate logic.
- Agents `Session` objects are intentionally **not used** in this initial design to avoid rewrites to how methods manage context.

Once this pattern is implemented in `MAS`, other methods (ChatDev, AutoGen, etc.) can be migrated in the same way by choosing appropriate `agent_id`s at each call site and, optionally, richer `logical_agents` configuration in their YAML files.
