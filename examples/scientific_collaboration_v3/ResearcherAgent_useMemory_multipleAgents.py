"""
before using this example, please make sure you have installed chromadb package:
uv pip install chromadb
This also needs connections to huggingface to use sentence-transformers embedding model, please also install sentence-transformers package:
uv pip install sentence-transformers

# NOTE: using huggingface mirror: run with `HF_ENDPOINT=https://hf-mirror.com python examples/scientific_collaboration_v3/ResearcherAgent_useMemory_twoAgents.py`
"""

import asyncio
import tempfile

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console


from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)

import sys
from pathlib import Path

# other sources
src_path = Path.home() / 'project/src'
sys.path.append(str(src_path))

from autogen_llm import create_model_client_from_config

def read_profile_json(aid: int, team_id: int=321) -> dict:
    import json
    file_path = Path.home() / f'project/science_world_simulator/research_team_simulator/researcher_profile/team{team_id}/{aid}.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

async def config_agent(aid: int, team_id: int = 321) -> AssistantAgent:
    # Use a permanent directory for ChromaDB persistence
    persist_dir = Path.home() / "project/science_world_simulator/research_team_simulator/chroma_memory" / f"team{team_id}" / f"aid{aid}"
    persist_dir.mkdir(parents=True, exist_ok=True)

    chroma_user_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="preferences",
            persistence_path=str(persist_dir),
            k=2,  # Return top k results
            score_threshold=0.4,  # Minimum similarity score
            embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                model_name="all-MiniLM-L6-v2"  # Use default embedding model
            ),
        )
    )

    # Load researcher profile JSON --> If previously added memory to local storage, can skip below add step and use above setting to get those memories back to the agent. Can customize an agnet with self._memory linked with this path, and add summaries of conversations to his memory.
    researcher_profile_data = read_profile_json(aid, team_id)
    await chroma_user_memory.add(
        MemoryContent(
            content=researcher_profile_data,  # dict works since mime_type=JSON
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
        name=f"agent_{aid}",
        model_client=model_client,
        memory=[chroma_user_memory],
        system_message=f"You are a researcher represented by a unique identifier, aid (author identifier) {aid}. You discuss with team members and/or other researchers. Reply in chinese. Say 'TERMINATE' if you reach a consensus to end the discussion.",
    )

    return agent



async def main() -> None:

    team_id = 658
    agent1 = await config_agent(aid=4022351, team_id=team_id)
    agent2 = await config_agent(aid=4863871, team_id=team_id)
    agent3 = await config_agent(aid=4960577, team_id=team_id)

    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat([agent1, agent2, agent3], termination_condition=termination)

    # await Console(team.run_stream(task="Introduce yourselves to each other and then discuss your research interests and possible collaborations. Discussing 3-5 reserch topics and if you find other related coauthors from your past collaborations (publication history), suggest them. Say the uppercase of 'Terminate', if you ALL reach a consensus on the discussion.")) # This always makes the first agent ends his responses as TERMINATE  --> Put this requirement in system message instead

    await Console(team.run_stream(task="Introduce yourselves to each other and then discuss your research interests and possible collaborations. Discussing 3-5 reserch topics and if you find other related coauthors from your past collaborations (publication history), suggest them."))

    
    # await Console(agent1.run_stream(task="What are your research interests and who are your most frequent collaborators?"))
    # await Console(agent1.run_stream(task="What are your research interests?")) # works


asyncio.run(main())