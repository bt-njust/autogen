import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

import sys
from pathlib import Path

# other sources
src_path = Path.home() / 'project/src'
sys.path.append(str(src_path))

from autogen_llm import create_model_client_from_config

async def main() -> None:
    model_client = create_model_client_from_config(
        config_file=".server_deployed_LLMs",
        config_section="ali_official",
        model_name="qwen-plus"
    )

    agent1 = AssistantAgent("Assistant1", model_client=model_client)
    agent2 = AssistantAgent("Assistant2", model_client=model_client)
    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat([agent1, agent2], termination_condition=termination)
    await Console(team.run_stream(task="Tell me some jokes."))


asyncio.run(main())
