# NOT tested 
import asyncio
import sys
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.mem0 import Mem0Memory   # same class works for cloud & local
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Add your custom src path
src_path = Path.home() / "project/src"
sys.path.append(str(src_path))
from autogen_llm import create_model_client_from_config


async def get_weather(city: str, units: str = "imperial") -> str:
    if units == "imperial":
        return f"The weather in {city} is 73 °F and Sunny."
    elif units == "metric":
        return f"The weather in {city} is 23 °C and Sunny."
    else:
        return f"Sorry, I don't know the weather in {city}."


async def main() -> None:
    # Toggle between cloud and local
    USE_CLOUD = False   # 🔄 flip to True if you want cloud Mem0 with API key

    # Create a model client
    model_client = create_model_client_from_config(
        config_file=".server_deployed_LLMs",
        config_section="zhipu_glm",
        model_name="glm-4.5"
    )

    # Local config for Mem0 (no API key needed)
    local_config = {
        "path": "./local_memories",   # directory where memories are persisted
        "db": "sqlite",               # backend (sqlite is the default)
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"  # local embedding model
    }

    mem0_memory = Mem0Memory(
        is_cloud=False,
        config=local_config,
        limit=5
    )

    # Add user preferences to memory
    await mem0_memory.add(
        MemoryContent(
            content="The weather should be in metric units",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "preferences", "type": "units"},
        )
    )

    await mem0_memory.add(
        MemoryContent(
            content="Meal recipe must be vegan",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "preferences", "type": "dietary"},
        )
    )

    # Create assistant with memory
    assistant_agent = AssistantAgent(
        name="assistant_agent",
        model_client=model_client,
        tools=[get_weather],
        memory=[mem0_memory],
    )

    # Ask about the weather
    stream = assistant_agent.run_stream(task="What are my dietary preferences?")
    await Console(stream)

    # Serialize the memory configuration
    print("=" * 80)
    config_json = mem0_memory.dump_component().model_dump_json()
    print(f"Memory config JSON: {config_json[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
