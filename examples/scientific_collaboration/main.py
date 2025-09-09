"""Scientific Collaboration Example

This example demonstrates how AutoGen can simulate realistic academic research scenarios
where multiple researchers discuss and select joint research topics based on their
profiles and collaboration history.

The simulation includes:
- Researchers with distinct expertise and collaboration histories
- Topic proposal and discussion mechanisms  
- Consensus-building processes
- Realistic academic conversation patterns
"""

import argparse
import asyncio
import configparser
import logging
import random
import sys
from typing import Annotated, Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from autogen_core import (
    AgentId,
    AgentRuntime,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
)
from autogen_core.model_context import BufferedChatCompletionContext, ChatCompletionContext
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
    ModelFamily,
)
from autogen_core.tool_agent import ToolAgent, tool_agent_caller_loop
from autogen_core.tools import FunctionTool, Tool, ToolSchema
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel


@dataclass
class ResearchTopic:
    """Represents a research topic proposal."""
    title: str
    description: str
    proposer: str
    required_expertise: List[str] = field(default_factory=list)
    votes: int = 0
    supporters: List[str] = field(default_factory=list)


@dataclass
class CollaborationHistory:
    """Represents past collaboration between researchers."""
    collaborators: List[str]
    topic: str
    outcome: str
    year: int


@dataclass
class ResearcherProfile:
    """Represents a researcher's academic profile."""
    name: str
    expertise: List[str]
    institution: str
    recent_publications: List[str]
    collaboration_history: List[CollaborationHistory] = field(default_factory=list)
    research_interests: List[str] = field(default_factory=list)


class CollaborationMessage(BaseModel):
    """Message for collaboration discussions."""
    content: str
    sender: str
    message_type: str  # "introduction", "proposal", "discussion", "vote", "consensus"
    topic_id: Optional[str] = None


class CollaborationState:
    """Manages the state of the scientific collaboration."""
    
    def __init__(self, max_topics: int = 8):
        self.researchers: Dict[str, ResearcherProfile] = {}
        self.topics: Dict[str, ResearchTopic] = {}
        self.discussion_round = 0
        self.consensus_reached = False
        self.selected_topics: List[str] = []
        self.max_topics = max_topics
        
    def add_researcher(self, profile: ResearcherProfile):
        """Add a researcher to the collaboration."""
        self.researchers[profile.name] = profile
        
    def add_topic(self, topic: ResearchTopic) -> Optional[str]:
        """Add a research topic and return its ID, or None if limit reached."""
        if len(self.topics) >= self.max_topics:
            return None
        topic_id = f"topic_{len(self.topics) + 1}"
        self.topics[topic_id] = topic
        return topic_id
        
    def vote_for_topic(self, topic_id: str, voter: str):
        """Record a vote for a topic."""
        if topic_id in self.topics and voter not in self.topics[topic_id].supporters:
            self.topics[topic_id].votes += 1
            self.topics[topic_id].supporters.append(voter)
            
    def get_remaining_topic_slots(self) -> int:
        """Get the number of remaining topic slots available."""
        return max(0, self.max_topics - len(self.topics))
            
    def get_top_topics(self, n: int = 3) -> List[ResearchTopic]:
        """Get the top N topics by votes."""
        sorted_topics = sorted(self.topics.values(), key=lambda t: t.votes, reverse=True)
        return sorted_topics[:n]


def create_model_client_from_config(config_file: str = ".server_deployed_LLMs", config_section: str = "ali_official") -> OpenAIChatCompletionClient:
    """Create model client using configparser approach from the provided configuration."""
    config = configparser.ConfigParser()
    
    # Try reading config file from multiple possible locations
    config_paths = [
        Path(config_file),
        Path.home() / 'project/pkg2.0/code' / config_file,
        Path(__file__).parent / config_file
    ]
    
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path is None:
        raise FileNotFoundError(f"Configuration file {config_file} not found in any of: {config_paths}")
    
    config.read(config_path)
    
    if config_section not in config:
        raise ValueError(f"Configuration section '{config_section}' not found in {config_path}")
    
    base_url = config[config_section]['base_url']
    api_key = config[config_section]['api_key']
    
    model_info = {
        'vision': False,
        'function_calling': True,
        'family': ModelFamily.UNKNOWN,
        'structured_output': True,
        'json_output': True  # Starting in v0.4.7, the required fields are enforced.
    }

    model_client = OpenAIChatCompletionClient(
        model="qwen-max",
        base_url=base_url,
        api_key=api_key,
        model_info=model_info
    )
    
    return model_client


# Global collaboration state
collaboration_state = CollaborationState()


@default_subscription
class ResearcherAgent(RoutedAgent):
    """Agent representing a researcher in the collaboration."""
    
    def __init__(
        self,
        profile: ResearcherProfile,
        model_client: ChatCompletionClient,
        model_context: ChatCompletionContext,
        tool_schema: List[ToolSchema],
        tool_agent_type: str,
    ) -> None:
        super().__init__(description=f"Researcher: {profile.name}")
        self.profile = profile
        self._model_client = model_client
        self._model_context = model_context
        self._tool_schema = tool_schema
        self._tool_agent_id = AgentId(tool_agent_type, self.id.key)
        
        # Create system message with researcher's profile
        system_prompt = self._create_system_prompt()
        self._system_messages: List[LLMMessage] = [SystemMessage(content=system_prompt)]
        
    def _create_system_prompt(self) -> str:
        """Create a system prompt based on the researcher's profile."""
        collab_history = ""
        if self.profile.collaboration_history:
            collab_history = "\n\nPast Collaborations:\n"
            for collab in self.profile.collaboration_history:
                collab_history += f"- {collab.topic} with {', '.join(collab.collaborators)} ({collab.year}) - {collab.outcome}\n"
        
        return f"""You are {self.profile.name}, a researcher from {self.profile.institution}.

Your expertise areas: {', '.join(self.profile.expertise)}
Your research interests: {', '.join(self.profile.research_interests)}

Recent publications:
{chr(10).join(f"- {pub}" for pub in self.profile.recent_publications)}
{collab_history}

You are participating in a scientific collaboration meeting to discuss and select joint research topics. 

Guidelines for your behavior:
1. Be professional and academically rigorous
2. Draw from your expertise when evaluating proposals
3. Consider past collaboration experiences
4. Propose topics that align with your interests but could benefit from interdisciplinary collaboration
5. Be constructive in discussions and provide specific feedback
6. Vote thoughtfully based on feasibility, impact, and your ability to contribute

Use the available tools to:
- Propose research topics
- Vote on proposals
- Get information about current topics and votes
- Check collaboration state"""

    @message_handler
    async def handle_message(self, message: CollaborationMessage, ctx: MessageContext) -> None:
        """Handle incoming collaboration messages."""
        # Add the message to model context
        await self._model_context.add_message(
            UserMessage(content=f"[{message.message_type.upper()}] {message.sender}: {message.content}", 
                       source=message.sender)
        )
        
        # Generate response using tools
        messages = await tool_agent_caller_loop(
            self,
            tool_agent_id=self._tool_agent_id,
            model_client=self._model_client,
            input_messages=self._system_messages + (await self._model_context.get_messages()),
            tool_schema=self._tool_schema,
            cancellation_token=ctx.cancellation_token,
        )
        
        # Add assistant messages to context
        for msg in messages:
            await self._model_context.add_message(msg)
            
        # Publish response
        if messages and isinstance(messages[-1].content, str):
            response = CollaborationMessage(
                content=messages[-1].content,
                sender=self.profile.name,
                message_type="discussion"
            )
            await self.publish_message(response, DefaultTopicId())


def propose_topic(
    researcher_name: str,
    title: Annotated[str, "Title of the research topic"],
    description: Annotated[str, "Detailed description of the research topic"],
    required_expertise: Annotated[str, "Comma-separated list of expertise areas needed"],
) -> Annotated[str, "Result of the topic proposal"]:
    """Propose a new research topic for collaboration."""
    # Check if we've reached the maximum number of topics
    if len(collaboration_state.topics) >= collaboration_state.max_topics:
        return f"Cannot propose new topic '{title}'. Maximum topic limit ({collaboration_state.max_topics}) has been reached. No more topics can be proposed."
    
    expertise_list = [exp.strip() for exp in required_expertise.split(",")]
    topic = ResearchTopic(
        title=title,
        description=description,
        proposer=researcher_name,
        required_expertise=expertise_list
    )
    topic_id = collaboration_state.add_topic(topic)
    
    if topic_id is None:
        # This shouldn't happen given the check above, but defensive programming
        return f"Cannot propose new topic '{title}'. Maximum topic limit ({collaboration_state.max_topics}) has been reached."
    
    remaining_slots = collaboration_state.get_remaining_topic_slots()
    
    print(f"\n🔬 NEW TOPIC PROPOSED by {researcher_name}")
    print(f"📋 Title: {title}")
    print(f"📝 Description: {description}")
    print(f"🎯 Required expertise: {', '.join(expertise_list)}")
    print(f"🆔 Topic ID: {topic_id}")
    print(f"📊 Remaining topic slots: {remaining_slots}")
    
    return f"Successfully proposed topic '{title}' (ID: {topic_id}). {remaining_slots} topic slots remaining. Other researchers can now vote on it."


def vote_for_topic(
    voter_name: str,
    topic_id: Annotated[str, "ID of the topic to vote for"],
    reasoning: Annotated[str, "Reasoning for the vote"],
) -> Annotated[str, "Result of the vote"]:
    """Vote for a research topic."""
    if topic_id not in collaboration_state.topics:
        return f"Topic {topic_id} does not exist."
        
    if voter_name in collaboration_state.topics[topic_id].supporters:
        return f"You have already voted for topic {topic_id}."
        
    collaboration_state.vote_for_topic(topic_id, voter_name)
    topic = collaboration_state.topics[topic_id]
    
    print(f"\n🗳️ VOTE CAST by {voter_name}")
    print(f"📋 Topic: {topic.title}")
    print(f"💭 Reasoning: {reasoning}")
    print(f"📊 Current votes: {topic.votes}")
    
    return f"Successfully voted for '{topic.title}'. Current votes: {topic.votes}"


def get_current_topics() -> Annotated[str, "List of all proposed topics with their current vote counts"]:
    """Get information about all currently proposed topics."""
    if not collaboration_state.topics:
        return "No topics have been proposed yet."
        
    topics_info = "Current Research Topics:\n\n"
    for topic_id, topic in collaboration_state.topics.items():
        topics_info += f"🆔 {topic_id} | 📋 {topic.title}\n"
        topics_info += f"   👤 Proposed by: {topic.proposer}\n"
        topics_info += f"   📝 Description: {topic.description}\n"
        topics_info += f"   🎯 Required expertise: {', '.join(topic.required_expertise)}\n"
        topics_info += f"   🗳️ Votes: {topic.votes} | 👥 Supporters: {', '.join(topic.supporters)}\n\n"
        
    return topics_info


def get_researcher_profiles() -> Annotated[str, "Information about all researchers in the collaboration"]:
    """Get information about all researchers and their expertise."""
    if not collaboration_state.researchers:
        return "No researchers are currently registered."
        
    profiles_info = "Researcher Profiles:\n\n"
    for name, profile in collaboration_state.researchers.items():
        profiles_info += f"👤 {profile.name} ({profile.institution})\n"
        profiles_info += f"   🔬 Expertise: {', '.join(profile.expertise)}\n"
        profiles_info += f"   🎯 Interests: {', '.join(profile.research_interests)}\n"
        if profile.collaboration_history:
            profiles_info += f"   🤝 Past collaborations: {len(profile.collaboration_history)}\n"
        profiles_info += "\n"
        
    return profiles_info


def get_collaboration_status() -> Annotated[str, "Current status of the collaboration"]:
    """Get the current status of the collaboration process."""
    status = f"Collaboration Status:\n"
    status += f"👥 Researchers: {len(collaboration_state.researchers)}\n"
    status += f"📋 Topics proposed: {len(collaboration_state.topics)}/{collaboration_state.max_topics}\n"
    status += f"📊 Remaining topic slots: {collaboration_state.get_remaining_topic_slots()}\n"
    status += f"🔄 Discussion round: {collaboration_state.discussion_round}\n"
    
    if collaboration_state.topics:
        top_topics = collaboration_state.get_top_topics(3)
        status += f"\n🏆 Top Topics by Votes:\n"
        for i, topic in enumerate(top_topics, 1):
            status += f"   {i}. {topic.title} ({topic.votes} votes)\n"
            
    return status


async def setup_collaboration(runtime: AgentRuntime, model_client: ChatCompletionClient) -> None:
    """Set up the scientific collaboration with researchers and tools."""
    
    # Define researcher profiles
    researchers = [
        ResearcherProfile(
            name="Dr. Sarah Chen",
            expertise=["Machine Learning", "Deep Learning", "Computer Vision"],
            institution="MIT",
            research_interests=["Multimodal AI", "Federated Learning", "AI Safety"],
            recent_publications=[
                "Federated Learning for Computer Vision: A Survey (2023)",
                "Robust Multimodal AI Systems (2023)",
                "Privacy-Preserving Deep Learning (2022)"
            ],
            collaboration_history=[
                CollaborationHistory(["Dr. James Wilson"], "Privacy-Preserving ML", "3 joint publications", 2022),
                CollaborationHistory(["Dr. Maria Garcia", "Dr. Alex Kim"], "Federated AI Systems", "NSF grant awarded", 2021)
            ]
        ),
        ResearcherProfile(
            name="Dr. James Wilson",
            expertise=["Data Science", "Statistical Analysis", "Big Data Analytics"],
            institution="Stanford University",
            research_interests=["Healthcare Analytics", "Social Media Analysis", "Ethical AI"],
            recent_publications=[
                "Ethical Considerations in Healthcare AI (2023)",
                "Large-Scale Social Media Sentiment Analysis (2023)",
                "Statistical Methods for Biased Data (2022)"
            ],
            collaboration_history=[
                CollaborationHistory(["Dr. Sarah Chen"], "Privacy-Preserving ML", "3 joint publications", 2022),
                CollaborationHistory(["Dr. Lisa Zhang"], "Healthcare AI Ethics", "Policy recommendations", 2021)
            ]
        ),
        ResearcherProfile(
            name="Dr. Maria Garcia",
            expertise=["Human-Computer Interaction", "UX Research", "Accessibility"],
            institution="UC Berkeley",
            research_interests=["AI-Human Collaboration", "Accessible AI", "Inclusive Design"],
            recent_publications=[
                "Designing Inclusive AI Interfaces (2023)",
                "User Trust in AI Systems (2023)",
                "Accessibility in Machine Learning Tools (2022)"
            ],
            collaboration_history=[
                CollaborationHistory(["Dr. Sarah Chen", "Dr. Alex Kim"], "Federated AI Systems", "NSF grant awarded", 2021),
                CollaborationHistory(["Dr. Michael Brown"], "Accessible AI Design", "5 joint publications", 2020)
            ]
        ),
        ResearcherProfile(
            name="Dr. Alex Kim",
            expertise=["Computational Biology", "Bioinformatics", "Systems Biology"],
            institution="Harvard Medical School",
            research_interests=["AI for Drug Discovery", "Genomics", "Personalized Medicine"],
            recent_publications=[
                "AI-Driven Drug Discovery Pipelines (2023)",
                "Genomic Data Analysis with Deep Learning (2023)",
                "Personalized Medicine through AI (2022)"
            ],
            collaboration_history=[
                CollaborationHistory(["Dr. Sarah Chen", "Dr. Maria Garcia"], "Federated AI Systems", "NSF grant awarded", 2021),
                CollaborationHistory(["Dr. Robert Taylor"], "AI in Drug Discovery", "Patent filed", 2022)
            ]
        ),
        ResearcherProfile(
            name="Dr. Lisa Zhang",
            expertise=["Cybersecurity", "Network Security", "AI Security"],
            institution="Carnegie Mellon University",
            research_interests=["Adversarial AI", "Secure ML", "Privacy-Preserving Systems"],
            recent_publications=[
                "Adversarial Attacks on Federated Learning (2023)",
                "Secure Multi-Party Computation for ML (2023)",
                "Privacy Attacks on Deep Learning Models (2022)"
            ],
            collaboration_history=[
                CollaborationHistory(["Dr. James Wilson"], "Healthcare AI Ethics", "Policy recommendations", 2021),
                CollaborationHistory(["Dr. Thomas Lee"], "Secure Federated Learning", "Top-tier conference papers", 2022)
            ]
        )
    ]
    
    # Add researchers to collaboration state
    for researcher in researchers:
        collaboration_state.add_researcher(researcher)
    
    # Register researcher agents and their specific tool agents
    for researcher in researchers:
        researcher_safe_name = researcher.name.replace(' ', '_').replace('.', '')
        
        # Create tools specific to this researcher
        def make_researcher_tools(researcher_name: str) -> List[Tool]:
            def propose_topic_for_researcher(title: str, description: str, required_expertise: str) -> str:
                return propose_topic(researcher_name, title, description, required_expertise)
            
            def vote_for_topic_for_researcher(topic_id: str, reasoning: str) -> str:
                return vote_for_topic(researcher_name, topic_id, reasoning)
            
            return [
                FunctionTool(
                    propose_topic_for_researcher,
                    name="propose_topic",
                    description="Propose a new research topic for collaboration",
                ),
                FunctionTool(
                    vote_for_topic_for_researcher,
                    name="vote_for_topic", 
                    description="Vote for a research topic with reasoning",
                ),
                FunctionTool(
                    get_current_topics,
                    name="get_current_topics",
                    description="Get information about all proposed topics and their votes",
                ),
                FunctionTool(
                    get_researcher_profiles,
                    name="get_researcher_profiles",
                    description="Get information about all researchers and their expertise",
                ),
                FunctionTool(
                    get_collaboration_status,
                    name="get_collaboration_status", 
                    description="Get the current status of the collaboration",
                ),
            ]
        
        researcher_tools = make_researcher_tools(researcher.name)
        
        # Register tool agent for this researcher
        await ToolAgent.register(
            runtime,
            f"CollaborationToolAgent_{researcher_safe_name}",
            lambda tools=researcher_tools: ToolAgent(description="Tool agent for scientific collaboration.", tools=tools),
        )
        
        # Register researcher agent
        await ResearcherAgent.register(
            runtime,
            f"Researcher_{researcher_safe_name}",
            lambda r=researcher, safe_name=researcher_safe_name, tools=researcher_tools: ResearcherAgent(
                profile=r,
                model_client=model_client,
                model_context=BufferedChatCompletionContext(buffer_size=15),
                tool_schema=[tool.schema for tool in tools],
                tool_agent_type=f"CollaborationToolAgent_{safe_name}",
            ),
        )


async def run_collaboration_round(
    runtime: AgentRuntime, 
    round_num: int, 
    phase: str,
    researchers: List[str]
) -> None:
    """Run a single round of collaboration discussion."""
    
    print(f"\n{'='*60}")
    print(f"🔄 COLLABORATION ROUND {round_num}: {phase.upper()}")
    print(f"{'='*60}")
    
    collaboration_state.discussion_round = round_num
    
    # Different prompts for different phases
    if phase == "introduction":
        prompt = "Please introduce yourself, your expertise, and mention any relevant past collaborations. Then look at who else is participating and suggest potential areas for collaboration."
    elif phase == "proposal":
        prompt = "Based on the introductions and the expertise available, propose 1-2 specific research topics that would benefit from interdisciplinary collaboration. Use the propose_topic tool."
    elif phase == "discussion":
        prompt = "Review the proposed topics and provide thoughtful feedback. Consider feasibility, your potential contributions, and the overall impact. Vote for topics you find most promising using the vote_for_topic tool."
    elif phase == "consensus":
        prompt = "Review the voting results and discuss the top-voted topics. Share your thoughts on next steps and how to move forward with the selected research directions."
    else:
        prompt = f"Continue the discussion for phase: {phase}"
    
    # Send messages to researchers in random order
    shuffled_researchers = researchers.copy()
    random.shuffle(shuffled_researchers)
    
    for researcher in shuffled_researchers:
        message = CollaborationMessage(
            content=prompt,
            sender="Moderator",
            message_type=phase
        )
        
        researcher_id = AgentId(f"Researcher_{researcher.replace(' ', '_').replace('.', '')}", "default")
        await runtime.send_message(message, researcher_id)
        
        # Add some delay between messages to avoid overwhelming
        await asyncio.sleep(1)


async def main(config_file: str = ".server_deployed_LLMs", config_section: str = "ali_official", num_rounds: int = 3) -> None:
    """Main entry point for the scientific collaboration simulation."""
    
    print("🧪 SCIENTIFIC COLLABORATION SIMULATION")
    print("=" * 50)
    print("Simulating a realistic academic research collaboration")
    print("where multiple researchers discuss and select joint research topics.")
    print()
    
    # Initialize runtime and model using new configuration approach
    runtime = SingleThreadedAgentRuntime()
    model_client = create_model_client_from_config(config_file, config_section)
    
    # Set up the collaboration
    await setup_collaboration(runtime, model_client)
    
    # Start the runtime
    runtime.start()
    
    # Get researcher names
    researcher_names = [
        "Dr. Sarah Chen",
        "Dr. James Wilson", 
        "Dr. Maria Garcia",
        "Dr. Alex Kim",
        "Dr. Lisa Zhang"
    ]
    
    try:
        # Run collaboration phases
        phases = ["introduction", "proposal", "discussion", "consensus"]
        
        for i, phase in enumerate(phases, 1):
            await run_collaboration_round(runtime, i, phase, researcher_names)
            
            # Wait for discussions to settle
            await asyncio.sleep(5)
            
            # Show status after each round
            if phase in ["proposal", "discussion"]:
                print(f"\n📊 STATUS AFTER {phase.upper()} PHASE:")
                print(get_collaboration_status())
                print()
        
        # Final summary
        print(f"\n🎯 FINAL COLLABORATION RESULTS")
        print("=" * 50)
        print(get_current_topics())
        print(get_collaboration_status())
        
        top_topics = collaboration_state.get_top_topics(3)
        if top_topics:
            print("\n🏆 SELECTED RESEARCH DIRECTIONS:")
            for i, topic in enumerate(top_topics, 1):
                print(f"{i}. {topic.title} ({topic.votes} votes)")
                print(f"   Proposed by: {topic.proposer}")
                print(f"   Supporters: {', '.join(topic.supporters)}")
                print()
                
    except KeyboardInterrupt:
        print("\n⚠️ Collaboration interrupted by user")
    finally:
        # Clean shutdown
        await runtime.stop_when_idle()
        await model_client.close()
        print("\n✅ Collaboration simulation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a scientific collaboration simulation.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--config-file", type=str, help="Path to the model configuration file.", default=".server_deployed_LLMs"
    )
    parser.add_argument(
        "--config-section", type=str, help="Configuration section to use.", default="ali_official"
    )
    parser.add_argument(
        "--num-rounds", type=int, help="Number of discussion rounds.", default=4
    )
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("autogen_core").setLevel(logging.DEBUG)
        handler = logging.FileHandler("scientific_collaboration.log")
        logging.getLogger("autogen_core").addHandler(handler)

    try:
        asyncio.run(main(args.config_file, args.config_section, args.num_rounds))
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Please ensure the configuration file '{args.config_file}' exists and contains the section '{args.config_section}'.")
        print("\n💡 Tip: To see how this example works without an API key, run:")
        print("    python demo.py")
        sys.exit(1)
    except Exception as e:
        if "api_key" in str(e).lower() or "authentication" in str(e).lower() or "connection" in str(e).lower():
            print(f"❌ Error: {e}")
            print("\n💡 This appears to be an API key or connection issue.")
            print("To test the collaboration system without an API key, run:")
            print("    python demo.py")
            sys.exit(1)
        else:
            print(f"❌ Error: {e}")
            raise
