# ~/project/autogen/.venv/lib/python3.12/site-packages/autogen_agentchat/agents/_assistant_agent.py

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import ListMemory, MemoryContent
from autogen_ext.models.openai import OpenAIChatCompletionClient

import sys
from pathlib import Path

src_path = Path.home() / "project/src"
sys.path.append(str(src_path))

from autogen_llm import create_model_client_from_config

async def main() -> None:
    # Create a model client.
    model_client = create_model_client_from_config(
        config_file=".server_deployed_LLMs",
        config_section="zhipu_glm",
        model_name="glm-4.5"
    )

    # Create a list-based memory with some initial content.
    memory = ListMemory()
    await memory.add(MemoryContent(content="User likes pizza.", mime_type="text/plain"))
    await memory.add(MemoryContent(content="User dislikes cheese.", mime_type="text/plain"))

    # Create an AssistantAgent instance with the model client and memory.
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        memory=[memory],
        system_message="You are a helpful assistant.",
    )

    result = await agent.run(task="What is a good dinner idea?")
    print(result.messages[-1].content)  # type: ignore

    # below shows the model context gets updated with memory content
    new_model_context = await agent._model_context.get_messages()
    print(f'      ==|| {new_model_context}')


asyncio.run(main())