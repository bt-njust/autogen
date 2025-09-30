"""
before using this example, please make sure you have installed chromadb package:
uv pip install chromadb
This also needs connections to huggingface to use sentence-transformers embedding model, please also install sentence-transformers package:
uv pip install sentence-transformers

# NOTE: using huggingface mirror: run with `HF_ENDPOINT=https://hf-mirror.com python examples/scientific_collaboration_v3/main_v2.py`

- add Phase 1 summary to participants' memories
- use Moderator agent to control the discussion and summarize the discussion at the end
- add FunctionCallTermination to end the discussion when moderator calls "approve" function
- using jinja2 template for prompts
"""

import asyncio
import tempfile
from jinja2 import Template

from typing import List, Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.base import TaskResult

from autogen_agentchat.messages import TextMessage, MemoryQueryEvent, ThoughtEvent, FunctionExecutionResult

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
from json_extractor import extract_valid_json

# termination ======
from typing import Sequence

from autogen_agentchat.base import TerminatedException, TerminationCondition
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, StopMessage, ToolCallExecutionEvent
from autogen_core import Component
from pydantic import BaseModel
from typing_extensions import Self


class FunctionCallTerminationConfig(BaseModel):
    """Configuration for the termination condition to allow for serialization
    and deserialization of the component.
    """

    function_name: str


class FunctionCallTermination(TerminationCondition, Component[FunctionCallTerminationConfig]):
    """Terminate the conversation if a FunctionExecutionResult with a specific name is received."""

    component_config_schema = FunctionCallTerminationConfig
    component_provider_override = "autogen_agentchat.conditions.FunctionCallTermination"
    """The schema for the component configuration."""

    def __init__(self, function_name: str) -> None:
        self._terminated = False
        self._function_name = function_name

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException("Termination condition has already been reached")
        for message in messages:
            if isinstance(message, ToolCallExecutionEvent):
                for execution in message.content:
                    if execution.name == self._function_name:
                        self._terminated = True
                        return StopMessage(
                            content=f"Function '{self._function_name}' was executed.",
                            source="FunctionCallTermination",
                        )
        return None

    async def reset(self) -> None:
        self._terminated = False

    def _to_config(self) -> FunctionCallTerminationConfig:
        return FunctionCallTerminationConfig(
            function_name=self._function_name,
        )

    @classmethod
    def _from_config(cls, config: FunctionCallTerminationConfig) -> Self:
        return cls(
            function_name=config.function_name,
        )

def end_discussion() -> None:
    """Approve the message when all feedbacks have been addressed."""
    pass

# termination ======

def read_profile_json(aid: int, team_id: int=321) -> dict:
    import json
    file_path = Path.home() / f'project/science_world_simulator/research_team_simulator/researcher_profile/team{team_id}/{aid}.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

async def config_agent(aid: int, system_message: str, team_id: int = 321, model_name: Literal['glm-4.5', 'glm-4.5-air', 'qwen-plus'] = 'glm-4.5', is_moderator: bool = False) -> AssistantAgent:
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

    await chroma_user_memory.clear()  # Clear existing memory if needed

    # Load researcher profile JSON --> If previously added memory to local storage, can skip below add step and use above setting to get those memories back to the agent. Can customize an agnet with self._memory linked with this path, and add summaries of conversations to his memory.
    researcher_profile_data = read_profile_json(aid, team_id)
    await chroma_user_memory.add(
        MemoryContent(
            content=researcher_profile_data,  # dict works since mime_type=JSON
            mime_type=MemoryMimeType.JSON,
        )
    )

    # Create a model client.
    # use qwen
    if model_name.startswith('qwen'):
        model_client = create_model_client_from_config()
    # use glm
    else:
        model_client = create_model_client_from_config(
            config_file=".server_deployed_LLMs",
            config_section="zhipu_glm",
            # model_name="glm-4.5-air"
            model_name=model_name
        )

    # Create an AssistantAgent instance with the model client and memory.
    if is_moderator:
        # equip with end_discussion tool
        agent = AssistantAgent(
            name=f"moderator_{aid}",
            model_client=model_client,
            memory=[chroma_user_memory],
            tools=[end_discussion],
            system_message=system_message
        )
    else:
        agent = AssistantAgent(
            name=f"agent_{aid}",
            model_client=model_client,
            memory=[chroma_user_memory],
            system_message=system_message
        )

    return agent

async def close_agents(*agents):
    for agent in agents:
        for mem in agent._memory:
            await mem.close()


async def main() -> None:

    team_id = 658
    aid1, aid2, aid3 = 4022351, 4863871, 4960577
    # model_name = 'glm-4.5'
    model_name = 'qwen-plus'
    prompt_path = Path.home() / "project/science_world_simulator/research_team_simulator/Prompts"
    phase1_system_prompt_file_Moderator = "Phase1_initialTopics_Collaborators_Moderator.jinjia2"
    with open(prompt_path / phase1_system_prompt_file_Moderator, 'r') as f:
        phase1_system_prompt_template_Moderator = Template(f.read())

    # Moderator
    agent1 = await config_agent(aid=aid1, team_id=team_id, model_name=model_name, system_message=phase1_system_prompt_template_Moderator.render(aid=aid1), is_moderator=True)
    # test meeting memory, works
    # await Console(agent1.run_stream(task="How about your last discussion?")) # should include the above memory content
    # await close_agents(agent1)

    phase1_system_prompt_file_Researcher = "Phase1_initialTopics_Collaborators.jinjia2"
    with open(prompt_path / phase1_system_prompt_file_Researcher, 'r') as f:
        phase1_system_prompt_template_Researcher = Template(f.read())
    agent2 = await config_agent(aid=aid2, team_id=team_id, model_name=model_name, system_message=phase1_system_prompt_template_Researcher.render(aid=aid2))

    # agent3 = await config_agent(aid=aid3, team_id=team_id, model_name=model_name, system_message=phase1_system_prompt_template_Researcher.render(aid=aid3))


    # print(agent1._system_messages)
    function_call_termination = FunctionCallTermination(function_name="end_discussion")

    # text_mention_termination = TextMentionTermination("TERMINATE")
    # max_messages_termination = MaxMessageTermination(max_messages=25)
    # termination = text_mention_termination | max_messages_termination
    # team = RoundRobinGroupChat([agent1, agent2, agent3], termination_condition=termination, max_turns=24)

    team = RoundRobinGroupChat([agent1, agent2], termination_condition=function_call_termination, max_turns=24)

    log_file = Path.home() / f'project/science_world_simulator/research_team_simulator/sample_team_details/test_results/team{team_id}_{model_name}_Test9.txt'
    agent_discussion_history = []

    # async for msg in team.run_stream(task="Introduce yourselves to each other and then discuss your research interests and possible collaborations. Discussing 3-5 reserch topics and suggest collaborators if you find other related coauthors from your past collaborations (publication history)."):
    async for msg in team.run_stream(task="Now starting the discussion"):
        if isinstance(msg, TaskResult):
            print("Stop Reason:", msg.stop_reason)
            agent_discussion_history.append(f"Stop Reason: {msg.stop_reason}\n")
        elif isinstance(msg, (ThoughtEvent, MemoryQueryEvent, FunctionExecutionResult)):
            pass
        else:
            sender = msg.source
            sender_fmt = f"{'='*10} {sender} {'='*10}"
            content = msg.content
            # agent_discussion_history.append((sender, content))
            print(sender_fmt)
            print(content)
            agent_discussion_history.append(f"\n{sender_fmt}\n{content}\n")

    with open(log_file, 'w', encoding='utf-8') as f:
        f.writelines(agent_discussion_history)

    # # print(agent_discussion_history[-2])
    # discussion_summary = extract_valid_json(agent_discussion_history[-2])
    # memory_of_meeting = {"Summary of the last meeting": discussion_summary}
    # await agent1._memory[0].add(
    #     MemoryContent(
    #         content=memory_of_meeting,
    #         mime_type=MemoryMimeType.JSON
    #     )
    # )

    # await agent2._memory[0].add(
    #     MemoryContent(
    #         content=memory_of_meeting,
    #         mime_type=MemoryMimeType.JSON
    #     )
    # )



    # save state to local json
    # team_state_dict = await team.save_state()
    # import json
    # with open(Path.home() / f'project/science_world_simulator/research_team_simulator/sample_team_details/team{team_id}_state.json', 'w') as f:
    #     json.dump(team_state_dict, f, indent=2)

    # await Console(agent1.run_stream(task="What are your research interests and who are your most frequent collaborators?"))
    # await Console(agent1.run_stream(task="What are your research interests?")) # works


    # TEST add memory for agent1
    # SUCCEEDS
    # await agent1._memory[0].clear()  # clear existing memory
    # await agent1._memory[0].add(
    #     MemoryContent(
    #         content="I prefer beef and noodles.", # DO NOT use 'User', since the agent takes this as the preference of who asked the question
    #         mime_type=MemoryMimeType.TEXT,
    #         metadata={"category": "preferences", "type": "units"},
    #     )
    # )
    # await Console(agent1.run_stream(task="What's your favorite food?")) # should include the above memory content
    # await close_agents(agent1)

    # await close_agents(agent1, agent2, agent3)
    await close_agents(agent1, agent2)
    await team.reset()


if __name__ == "__main__":
    asyncio.run(main())
