"""
before using this example, please make sure you have installed chromadb package:
uv pip install chromadb
This also needs connections to huggingface to use sentence-transformers embedding model, please also install sentence-transformers package:
uv pip install sentence-transformers

# NOTE: using huggingface mirror: run with `HF_ENDPOINT=https://hf-mirror.com python examples/DocExamples/BuiltIn_AssistantAgent_useMemory_chroma.py`

"""
import tempfile

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient

import sys
from pathlib import Path

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

# synthesized for tool test only
async def get_recipe(dietary: str) -> str:
    if dietary == "vegan":
        return "Here is a vegan recipe: Vegan Buddha Bowl with Quinoa and Roasted Vegetables."
    elif dietary == "vegetarian":
        return "Here is a vegetarian recipe: Caprese Salad with Fresh Mozzarella and Basil."
    else:
        return "Here is a recipe: Grilled Chicken with Lemon and Herbs."

async def main() -> None:
    # Use a temporary directory for ChromaDB persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        chroma_user_memory = ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name="preferences",
                persistence_path=tmpdir,  # Use the temp directory here
                k=2,  # Return top k results
                score_threshold=0.4,  # Minimum similarity score
                embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                    model_name="all-MiniLM-L6-v2"  # Use default model for testing
                ),
            )
        )
        # Add user preferences to memory
        await chroma_user_memory.add(
            MemoryContent(
                content="The weather should be in metric units",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "units"},
            )
        )

        await chroma_user_memory.add(
            MemoryContent(
                content="Meal recipe must be vegan",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "dietary"},
            )
        )

        model_client = create_model_client_from_config(
            config_file=".server_deployed_LLMs",
            config_section="zhipu_glm",
            model_name="glm-4.5"
        )

        # Create assistant agent with ChromaDB memory
        assistant_agent = AssistantAgent(
            name="assistant_agent",
            model_client=model_client,
            tools=[get_weather],
            memory=[chroma_user_memory],
        )

        stream = assistant_agent.run_stream(task="What is the weather in New York?")
        await Console(stream)

        # resp = await assistant_agent._model_context.get_messages()
        # print(f'      ==|| {resp}')

        await model_client.close()
        await chroma_user_memory.close()

import asyncio
asyncio.run(main())