import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.models.openai import OpenAIChatCompletionClient


import sys
from pathlib import Path

src_path = Path.home() / "project/src"
sys.path.append(str(src_path))

from autogen_llm import create_model_client_from_config

async def main() -> None:
    # Create a model client.
    model_client = create_model_client_from_config()

    # Create a model context that only keeps the last 2 messages (1 user + 1 assistant).
    model_context = BufferedChatCompletionContext(buffer_size=2)

    # Create an AssistantAgent instance with the model client and context.
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        model_context=model_context,
        system_message="You are a helpful assistant.",
    )

    result = await agent.run(task="Name two cities in North America.")
    print(result.messages[-1].content)  # type: ignore

    result = await agent.run(task="My favorite color is blue.")
    print(result.messages[-1].content)  # type: ignore

    result = await agent.run(task="Did I ask you any question?")
    print(result.messages[-1].content)  # type: ignore


asyncio.run(main())

