"""
before using this example, please make sure you have installed chromadb package:
uv pip install chromadb
This also needs connections to huggingface to use sentence-transformers embedding model, please also install sentence-transformers package:
uv pip install sentence-transformers

# NOTE: using huggingface mirror: run with `HF_ENDPOINT=https://hf-mirror.com python examples/scientific_collaboration_v3/ResearcherAgent_useMemory.py`
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

import sys
from pathlib import Path

src_path = Path.home() / "project/src"
sys.path.append(str(src_path))

from autogen_llm import create_model_client_from_config

# synthesized for tool test only
async def search_researcher_works(researcher_name: str) -> str:
    if researcher_name.lower() == "alice":
        return "Alice recently published works: AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"
    elif researcher_name.lower() == "bob":
        return "Bob recently published works: Exploring the Intersection of Natural Language Processing and Computer Vision"
    elif researcher_name.lower() == "charlie":
        return "Charlie recently published works: Advances in Robotics and Autonomous Systems"
    else:
        return f"{researcher_name} recently published works in various subfields and applications."

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

        # Add some initial memory content
        await chroma_user_memory.add(
            MemoryContent(
                content="User prefers research papers on machine learning and AI.",
                mime_type=MemoryMimeType.TEXT,
            )
        )
        await chroma_user_memory.add(
            MemoryContent(
                content="User is interested in recent advancements in natural language processing.",
                mime_type=MemoryMimeType.TEXT,
            )
        )

        # also add some memory in json format, first level entries are a researcher's past topics, and second level entries are specific papers and coauthors with frequency of collaboration
        await chroma_user_memory.add(
            MemoryContent(
            # when using JSON format, please make sure the content is a dict
            content={
                "past_topics": ["machine learning", "AI", "natural language processing"],
                "papers": [
                {
                    "title": "Advancements in AI",
                    "coauthors": ["Alice", "Bob"],
                    "collaboration_frequency": 5
                },
                {
                    "title": "NLP Techniques",
                    "coauthors": ["Charlie"],
                    "collaboration_frequency": 2
                }
                ]
            },
            mime_type=MemoryMimeType.JSON,
            )
        )

        # Create a model client.
        model_client = create_model_client_from_config(
            config_file=".server_deployed_LLMs",
            config_section="zhipu_glm",
            model_name="glm-4.5"
        )

        # Create an AssistantAgent instance with the model client and memory.
        agent = AssistantAgent(
            name="researcher_agent",
            model_client=model_client,
            memory=[chroma_user_memory],
            system_message="You are a researcher.",
            tools=[search_researcher_works],
        )

        # result = await agent.run(task="Share me some recently published works from your most frequently collaborated researchers") # not work
        # result = await agent.run(task="Who are your most frequently collaborated researchers? Then share me some recently published works from them.") # not work
        # result = await agent.run(task="Who are your most frequently collaborated researchers?") # works
        # result = await agent.run(task="Who are your most frequently collaborated researchers? Only return the collaborator names") # works
        result = await agent.run(task="Who are your most frequently collaborated researchers? Only return the collaborator names, then use available tools to find their recently published works.") # partially works, expected results should only include Bob and Alice's work, but Charlie's work was returned

        print(result.messages[-1].content)  # type: ignore

        # below shows the model context gets updated with memory content
        new_model_context = await agent._model_context.get_messages()
        print(f'      ==|| {new_model_context}')

        await model_client.close()
        await chroma_user_memory.close()

import asyncio
asyncio.run(main())
