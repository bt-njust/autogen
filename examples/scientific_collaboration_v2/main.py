"""Scientific Collaboration V2 Example

This enhanced example demonstrates how AutoGen can simulate realistic academic research scenarios
with improved team structures and role-based collaboration. Based on the problem statement requirements,
this version includes:

- Enhanced role system: leader, co-leader, newcomer, incumbent
- No topic voting strategy - discussion-based consensus instead
- 4-phase collaboration: introduction, proposal, discussion, consensus
- Topic source identification (granted project, future grant, research expansion, etc.)
- Termination conditions and round limits
- Valid researcher names using only numbers, '_', or '-'
- Improved asyncio handling to prevent endless loops
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
from enum import Enum

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


class TeamRole(Enum):
    """Enhanced team role definitions."""
    LEADER = "leader"  # Single leader in the team
    CO_LEADER = "co_leader"  # Multiple leaders scenario
    INCUMBENT = "incumbent"  # Established team member (3+ years)
    NEWCOMER = "newcomer"  # 1-2 collaborations with current team


class AcademicPosition(Enum):
    """Academic position hierarchy."""
    PROFESSOR = "professor"
    ASSOCIATE_PROFESSOR = "associate_professor"
    ASSISTANT_PROFESSOR = "assistant_professor"
    POSTDOC = "postdoc"
    PHD_CANDIDATE = "phd_candidate"


class TopicSource(Enum):
    """Source/type of research topic."""
    GRANTED_PROJECT = "granted_project"
    FUTURE_GRANT_PROJECT = "future_grant_project"
    RESEARCH_EXPANSION = "research_expansion"
    EXPLORE_NEW_DIRECTIONS = "explore_new_directions"
    ASSIGNMENT_FROM_PROFESSOR = "assignment_from_professor"
    PHD_INITIATIVE = "phd_initiative"


@dataclass
class CollaborationHistory:
    """Simple collaboration history record."""
    collaborators: List[str]
    topic: str
    outcome: str
    year: int


@dataclass
class ResearcherProfile:
    """Enhanced researcher profile with team role and academic position."""
    name: str
    academic_position: AcademicPosition
    team_role: TeamRole
    expertise: List[str]
    institution: str
    research_interests: List[str]
    recent_publications: List[str]
    collaboration_history: List[CollaborationHistory] = field(default_factory=list)
    years_in_team: int = 0
    current_workload: str = "moderate"  # light, moderate, heavy


@dataclass
class ResearchTopic:
    """Enhanced research topic with source information."""
    title: str
    description: str
    proposer: str
    source: TopicSource
    required_expertise: List[str] = field(default_factory=list)
    assigned_members: List[str] = field(default_factory=list)
    status: str = "proposed"  # proposed, discussed, accepted, declined
    priority_level: str = "medium"  # high, medium, low


class CollaborationMessage(BaseModel):
    """Message for collaboration discussions."""
    content: str
    sender: str
    message_type: str  # "introduction", "proposal", "discussion", "consensus"
    topic_id: Optional[str] = None
    round_number: int = 0


class CollaborationState:
    """Manages the enhanced collaboration state."""
    
    def __init__(self, max_topics: int = 6, max_rounds_per_phase: int = 3):
        self.researchers: Dict[str, ResearcherProfile] = {}
        self.topics: Dict[str, ResearchTopic] = {}
        self.discussion_round = 0
        self.consensus_reached = False
        self.selected_topics: List[str] = []
        self.max_topics = max_topics
        self.max_rounds_per_phase = max_rounds_per_phase
        self.current_phase = ""
        self.phase_round_count = 0
        self.terminated = False
        
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
        
    def get_remaining_topic_slots(self) -> int:
        """Get the number of remaining topic slots available."""
        return max(0, self.max_topics - len(self.topics))
            
    def get_topics_by_status(self, status: str) -> List[ResearchTopic]:
        """Get topics by their status."""
        return [topic for topic in self.topics.values() if topic.status == status]
        
    def should_terminate_phase(self) -> bool:
        """Check if current phase should terminate."""
        return self.phase_round_count >= self.max_rounds_per_phase or self.terminated
        
    def increment_phase_round(self):
        """Increment the phase round counter."""
        self.phase_round_count += 1
        
    def reset_phase_counter(self, phase: str):
        """Reset phase counter for new phase."""
        self.phase_round_count = 0
        self.current_phase = phase


# Global collaboration state
collaboration_state = CollaborationState()


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
        'json_output': True
    }

    model_client = OpenAIChatCompletionClient(
        model="qwen-max",
        base_url=base_url,
        api_key=api_key,
        model_info=model_info
    )
    
    return model_client


def propose_topic(
    researcher_name: str,
    title: Annotated[str, "Title of the research topic"],
    description: Annotated[str, "Detailed description of the research topic"],
    source: Annotated[str, "Source of the topic: granted_project, future_grant_project, research_expansion, explore_new_directions, assignment_from_professor, phd_initiative"],
    required_expertise: Annotated[str, "Comma-separated list of expertise areas needed"],
    priority_level: Annotated[str, "Priority level: high, medium, low"] = "medium",
) -> Annotated[str, "Result of the topic proposal"]:
    """Propose a new research topic for collaboration."""
    # Check if we've reached the maximum number of topics
    if len(collaboration_state.topics) >= collaboration_state.max_topics:
        return f"Cannot propose new topic '{title}'. Maximum topic limit ({collaboration_state.max_topics}) has been reached."
    
    try:
        topic_source = TopicSource(source.lower())
    except ValueError:
        return f"Invalid topic source '{source}'. Must be one of: {', '.join([s.value for s in TopicSource])}"
        
    expertise_list = [exp.strip() for exp in required_expertise.split(",")]
    
    topic = ResearchTopic(
        title=title,
        description=description,
        proposer=researcher_name,
        source=topic_source,
        required_expertise=expertise_list,
        priority_level=priority_level
    )
    
    topic_id = collaboration_state.add_topic(topic)
    
    if topic_id is None:
        return f"Cannot propose new topic '{title}'. Maximum topic limit has been reached."
    
    remaining_slots = collaboration_state.get_remaining_topic_slots()
    
    print(f"\n🔬 NEW TOPIC PROPOSED by {researcher_name}")
    print(f"📋 Title: {title}")
    print(f"📝 Description: {description}")
    print(f"🎯 Source: {topic_source.value}")
    print(f"🔧 Required expertise: {', '.join(expertise_list)}")
    print(f"⚡ Priority: {priority_level}")
    print(f"🆔 Topic ID: {topic_id}")
    print(f"📊 Remaining topic slots: {remaining_slots}")
    
    return f"Successfully proposed topic '{title}' (ID: {topic_id}) from {topic_source.value}. Priority: {priority_level}. {remaining_slots} topic slots remaining."


def discuss_topic(
    participant_name: str,
    topic_id: Annotated[str, "ID of the topic to discuss"],
    interest_level: Annotated[str, "Level of interest: high, medium, low, none"],
    contribution_level: Annotated[str, "Potential contribution level: lead, significant, moderate, minimal"],
    reasoning: Annotated[str, "Reasoning for the assessment"],
) -> Annotated[str, "Result of the discussion input"]:
    """Provide discussion input for a research topic instead of voting."""
    if topic_id not in collaboration_state.topics:
        return f"Topic {topic_id} does not exist."
        
    topic = collaboration_state.topics[topic_id]
    
    # Get participant profile for context
    participant_profile = collaboration_state.researchers.get(participant_name)
    
    print(f"\n💬 TOPIC DISCUSSION by {participant_name}")
    print(f"📋 Topic: {topic.title}")
    print(f"💡 Interest level: {interest_level}")
    print(f"🤝 Contribution level: {contribution_level}")
    print(f"💭 Reasoning: {reasoning}")
    
    if participant_profile:
        print(f"👤 Role: {participant_profile.team_role.value} ({participant_profile.academic_position.value})")
        print(f"📊 Current workload: {participant_profile.current_workload}")
    
    return f"Discussed '{topic.title}'. Interest: {interest_level}, Contribution: {contribution_level}. {reasoning}"


def assign_to_topic(
    assigner_name: str,
    topic_id: Annotated[str, "ID of the topic"],
    assignee_name: Annotated[str, "Name of the person to assign"],
    reasoning: Annotated[str, "Reasoning for the assignment"],
) -> Annotated[str, "Result of the assignment"]:
    """Assign a team member to a topic (typically used by leaders)."""
    if topic_id not in collaboration_state.topics:
        return f"Topic {topic_id} does not exist."
        
    if assignee_name not in collaboration_state.researchers:
        return f"Researcher {assignee_name} not found."
        
    assigner_profile = collaboration_state.researchers.get(assigner_name)
    if assigner_profile and assigner_profile.team_role not in [TeamRole.LEADER, TeamRole.CO_LEADER]:
        return f"Only leaders can make topic assignments."
        
    topic = collaboration_state.topics[topic_id]
    
    if assignee_name not in topic.assigned_members:
        topic.assigned_members.append(assignee_name)
    
    print(f"\n📋 TOPIC ASSIGNMENT by {assigner_name}")
    print(f"📝 Topic: {topic.title}")
    print(f"👤 Assigned to: {assignee_name}")
    print(f"💭 Reasoning: {reasoning}")
    print(f"👥 All assigned members: {', '.join(topic.assigned_members)}")
    
    return f"Successfully assigned {assignee_name} to '{topic.title}'. Reasoning: {reasoning}"


def get_current_topics() -> Annotated[str, "List of all proposed topics with their current status"]:
    """Get information about all currently proposed topics."""
    if not collaboration_state.topics:
        return "No topics have been proposed yet."
        
    topics_info = "Current Research Topics:\n\n"
    for topic_id, topic in collaboration_state.topics.items():
        topics_info += f"🆔 {topic_id} | 📋 {topic.title}\n"
        topics_info += f"   👤 Proposed by: {topic.proposer}\n"
        topics_info += f"   📝 Description: {topic.description}\n"
        topics_info += f"   🎯 Source: {topic.source.value}\n"
        topics_info += f"   🔧 Required expertise: {', '.join(topic.required_expertise)}\n"
        topics_info += f"   ⚡ Priority: {topic.priority_level}\n"
        topics_info += f"   📊 Status: {topic.status}\n"
        if topic.assigned_members:
            topics_info += f"   👥 Assigned members: {', '.join(topic.assigned_members)}\n"
        topics_info += "\n"
        
    return topics_info


def get_researcher_profiles() -> Annotated[str, "Information about all researchers in the collaboration"]:
    """Get information about all researchers and their roles."""
    if not collaboration_state.researchers:
        return "No researchers are currently registered."
        
    profiles_info = "Team Member Profiles:\n\n"
    for name, profile in collaboration_state.researchers.items():
        profiles_info += f"👤 {profile.name} | {profile.academic_position.value}\n"
        profiles_info += f"   🎭 Team role: {profile.team_role.value}\n"
        profiles_info += f"   🏛️ Institution: {profile.institution}\n"
        profiles_info += f"   🔬 Expertise: {', '.join(profile.expertise)}\n"
        profiles_info += f"   🎯 Interests: {', '.join(profile.research_interests)}\n"
        profiles_info += f"   ⏰ Years in team: {profile.years_in_team}\n"
        profiles_info += f"   📊 Current workload: {profile.current_workload}\n"
        if profile.collaboration_history:
            profiles_info += f"   🤝 Collaborations: {len(profile.collaboration_history)} partnerships\n"
        profiles_info += "\n"
        
    return profiles_info


def get_collaboration_status() -> Annotated[str, "Current status of the collaboration"]:
    """Get the current status of the collaboration process."""
    status = f"Collaboration Status:\n"
    status += f"👥 Team members: {len(collaboration_state.researchers)}\n"
    status += f"📋 Topics proposed: {len(collaboration_state.topics)}/{collaboration_state.max_topics}\n"
    status += f"📊 Remaining topic slots: {collaboration_state.get_remaining_topic_slots()}\n"
    status += f"🔄 Discussion round: {collaboration_state.discussion_round}\n"
    status += f"📍 Current phase: {collaboration_state.current_phase}\n"
    status += f"🔢 Phase round: {collaboration_state.phase_round_count}/{collaboration_state.max_rounds_per_phase}\n"
    
    # Show topics by status
    for status_type in ["proposed", "discussed", "accepted", "declined"]:
        topics = collaboration_state.get_topics_by_status(status_type)
        if topics:
            status += f"\n📝 {status_type.title()} topics: {len(topics)}\n"
            for topic in topics[:3]:  # Show first 3
                status += f"   - {topic.title} (by {topic.proposer})\n"
            
    return status


@default_subscription
class ResearcherAgent(RoutedAgent):
    """Agent representing a researcher in the enhanced collaboration."""
    
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
        self._message_count = 0  # Track messages to prevent endless loops
        self._max_messages_per_round = 5
        
        # Create system message with enhanced researcher's profile
        system_prompt = self._create_system_prompt()
        self._system_messages: List[LLMMessage] = [SystemMessage(content=system_prompt)]
        
    def _create_system_prompt(self) -> str:
        """Create a system prompt based on the researcher's enhanced profile."""
        collab_history = ""
        if self.profile.collaboration_history:
            collab_history = "\n\nPast Collaborations:\n"
            for collab in self.profile.collaboration_history:
                collab_history += f"- {collab.topic} with {', '.join(collab.collaborators)} ({collab.year}) - {collab.outcome}\n"
        
        role_behavior = ""
        if self.profile.team_role == TeamRole.LEADER:
            role_behavior = "As the team leader, you take initiative in guiding discussions, making assignments, and ensuring consensus. You have authority to assign topics to team members based on their expertise and workload."
        elif self.profile.team_role == TeamRole.CO_LEADER:
            role_behavior = "As a co-leader, you share leadership responsibilities and help guide discussions. You can participate in assignment decisions and facilitate consensus building."
        elif self.profile.team_role == TeamRole.INCUMBENT:
            role_behavior = "As an established team member, you contribute expertise and insights based on your experience. You can propose topics and provide guidance to newer members."
        elif self.profile.team_role == TeamRole.NEWCOMER:
            role_behavior = "As a newcomer to the team, you may have limited knowledge about team dynamics but bring fresh perspectives. You might receive topic assignments from seniors rather than proposing topics yourself."

        position_context = ""
        if self.profile.academic_position == AcademicPosition.PROFESSOR:
            position_context = "As a professor, you often have grant projects and can assign research directions to your team."
        elif self.profile.academic_position == AcademicPosition.POSTDOC:
            position_context = "As a postdoc, you have specialized skills and can take significant responsibility for research projects."
        elif self.profile.academic_position == AcademicPosition.PHD_CANDIDATE:
            position_context = "As a PhD candidate, you may propose topics for your dissertation or receive assignments from professors."

        return f"""You are {self.profile.name}, a {self.profile.academic_position.value} from {self.profile.institution}.

ROLE & POSITION:
- Team role: {self.profile.team_role.value}
- Academic position: {self.profile.academic_position.value}
- Years in team: {self.profile.years_in_team}
- Current workload: {self.profile.current_workload}

EXPERTISE & INTERESTS:
- Expertise areas: {', '.join(self.profile.expertise)}
- Research interests: {', '.join(self.profile.research_interests)}

Recent publications:
{chr(10).join(f"- {pub}" for pub in self.profile.recent_publications)}
{collab_history}

BEHAVIORAL GUIDELINES:
{role_behavior}

{position_context}

You are participating in a 4-phase scientific collaboration meeting:
1. Introduction: Present yourself and your background
2. Proposal: Propose research topics based on your role and position
3. Discussion: Discuss topics, express interest levels, and assess contributions
4. Consensus: Finalize topic assignments and next steps

IMPORTANT INTERACTION RULES:
- Do NOT send messages to yourself or create endless loops
- Keep responses focused and concise
- Use available tools appropriately based on your role
- Consider your workload when committing to topics
- Respect the team hierarchy and discussion flow
- Limit your contributions to avoid overwhelming the discussion

Use the available tools to:
- Propose research topics (if appropriate for your role)
- Discuss topics with interest and contribution levels
- Make assignments (if you're a leader)
- Check current status and other participants' profiles"""

    @message_handler
    async def handle_message(self, message: CollaborationMessage, ctx: MessageContext) -> None:
        """Handle incoming collaboration messages with loop prevention."""
        # Prevent endless loops - check if message is from self
        if message.sender == self.profile.name:
            return
            
        # Limit messages per round to prevent overwhelming
        if self._message_count >= self._max_messages_per_round:
            return
            
        self._message_count += 1
        
        # Add the message to model context
        await self._model_context.add_message(
            UserMessage(content=f"[{message.message_type.upper()}] {message.sender}: {message.content}", 
                       source=message.sender)
        )
        
        # Generate response using tools with timeout protection
        try:
            messages = await asyncio.wait_for(
                tool_agent_caller_loop(
                    self,
                    tool_agent_id=self._tool_agent_id,
                    model_client=self._model_client,
                    input_messages=self._system_messages + (await self._model_context.get_messages()),
                    tool_schema=self._tool_schema,
                    cancellation_token=ctx.cancellation_token,
                ),
                timeout=30.0  # 30 second timeout to prevent hanging
            )
        except asyncio.TimeoutError:
            print(f"⚠️ {self.profile.name} response timed out")
            return
        except Exception as e:
            print(f"⚠️ {self.profile.name} encountered error: {e}")
            return
        
        # Add assistant messages to context
        for msg in messages:
            await self._model_context.add_message(msg)
            
        # Publish response if valid
        if messages and isinstance(messages[-1].content, str):
            response = CollaborationMessage(
                content=messages[-1].content,
                sender=self.profile.name,
                message_type="discussion",
                round_number=collaboration_state.discussion_round
            )
            await self.publish_message(response, DefaultTopicId())
    
    def reset_message_count(self):
        """Reset message count for new round."""
        self._message_count = 0


async def setup_collaboration(runtime: AgentRuntime, model_client: ChatCompletionClient) -> None:
    """Set up the scientific collaboration with enhanced researchers and tools."""
    
    # Define enhanced researcher profiles with valid names (numbers, '_', '-' only)
    researchers = [
        ResearcherProfile(
            name="Prof_Chen_001",
            academic_position=AcademicPosition.PROFESSOR,
            team_role=TeamRole.LEADER,
            expertise=["Machine Learning", "Deep Learning", "Computer Vision"],
            institution="MIT",
            research_interests=["Multimodal AI", "Federated Learning", "AI Safety"],
            recent_publications=[
                "Federated Learning for Computer Vision: A Survey (2023)",
                "Robust Multimodal AI Systems (2023)",
                "Privacy-Preserving Deep Learning (2022)"
            ],
            collaboration_history=[
                CollaborationHistory(["Dr_Wilson_002"], "Privacy-Preserving ML", "3 joint publications", 2022),
                CollaborationHistory(["Dr_Garcia_003", "Dr_Kim_004"], "Federated AI Systems", "NSF grant awarded", 2021)
            ],
            years_in_team=5,
            current_workload="heavy"
        ),
        ResearcherProfile(
            name="Dr_Wilson_002",
            academic_position=AcademicPosition.ASSOCIATE_PROFESSOR,
            team_role=TeamRole.CO_LEADER,
            expertise=["Data Science", "Statistical Analysis", "Big Data Analytics"],
            institution="Stanford University",
            research_interests=["Healthcare Analytics", "Social Media Analysis", "Ethical AI"],
            recent_publications=[
                "Ethical Considerations in Healthcare AI (2023)",
                "Large-Scale Social Media Sentiment Analysis (2023)",
                "Statistical Methods for Biased Data (2022)"
            ],
            collaboration_history=[
                CollaborationHistory(["Prof_Chen_001"], "Privacy-Preserving ML", "3 joint publications", 2022),
            ],
            years_in_team=4,
            current_workload="moderate"
        ),
        ResearcherProfile(
            name="Dr_Garcia_003",
            academic_position=AcademicPosition.ASSISTANT_PROFESSOR,
            team_role=TeamRole.INCUMBENT,
            expertise=["Human-Computer Interaction", "UX Research", "Accessibility"],
            institution="UC Berkeley",
            research_interests=["AI-Human Collaboration", "Accessible AI", "Inclusive Design"],
            recent_publications=[
                "Designing Inclusive AI Interfaces (2023)",
                "User Trust in AI Systems (2023)",
                "Accessibility in Machine Learning Tools (2022)"
            ],
            collaboration_history=[
                CollaborationHistory(["Prof_Chen_001", "Dr_Kim_004"], "Federated AI Systems", "NSF grant awarded", 2021),
            ],
            years_in_team=3,
            current_workload="moderate"
        ),
        ResearcherProfile(
            name="Dr_Kim_004",
            academic_position=AcademicPosition.POSTDOC,
            team_role=TeamRole.INCUMBENT,
            expertise=["Computational Biology", "Bioinformatics", "Systems Biology"],
            institution="Harvard Medical School",
            research_interests=["AI for Drug Discovery", "Genomics", "Personalized Medicine"],
            recent_publications=[
                "AI-Driven Drug Discovery Platforms (2023)",
                "Genomic Data Analysis with Deep Learning (2023)",
                "Personalized Medicine through AI (2022)"
            ],
            collaboration_history=[
                CollaborationHistory(["Prof_Chen_001", "Dr_Garcia_003"], "Federated AI Systems", "NSF grant awarded", 2021),
            ],
            years_in_team=2,
            current_workload="light"
        ),
        ResearcherProfile(
            name="PhD_Zhang_005",
            academic_position=AcademicPosition.PHD_CANDIDATE,
            team_role=TeamRole.NEWCOMER,
            expertise=["Natural Language Processing", "Text Mining"],
            institution="MIT",
            research_interests=["Conversational AI", "Language Models", "Text Analysis"],
            recent_publications=[
                "Advances in Conversational AI (2023)",
                "Text Mining for Social Good (2023)"
            ],
            collaboration_history=[
                CollaborationHistory(["Prof_Chen_001"], "AI Safety Research", "Conference paper", 2023),
            ],
            years_in_team=1,
            current_workload="light"
        ),
        ResearcherProfile(
            name="Postdoc_Lee_006",
            academic_position=AcademicPosition.POSTDOC,
            team_role=TeamRole.NEWCOMER,
            expertise=["Robotics", "Computer Vision", "AI Control"],
            institution="Stanford University",
            research_interests=["Autonomous Systems", "Robot Learning", "Vision-based Control"],
            recent_publications=[
                "Vision-based Robot Control (2023)",
                "Autonomous Systems in Real Environments (2023)"
            ],
            collaboration_history=[
                CollaborationHistory(["Dr_Wilson_002"], "Robotics Ethics", "Workshop paper", 2023),
            ],
            years_in_team=1,
            current_workload="light"
        )
    ]
    
    # Add researchers to collaboration state
    for researcher in researchers:
        collaboration_state.add_researcher(researcher)
    
    # Register researcher agents and their specific tool agents
    for researcher in researchers:
        researcher_safe_name = researcher.name
        
        # Create tools specific to this researcher based on their role
        def make_researcher_tools(researcher_name: str, researcher_profile: ResearcherProfile) -> List[Tool]:
            tools = []
            
            # All researchers can discuss topics
            def discuss_topic_for_researcher(topic_id: str, interest_level: str, contribution_level: str, reasoning: str) -> str:
                return discuss_topic(researcher_name, topic_id, interest_level, contribution_level, reasoning)
            
            tools.append(FunctionTool(
                discuss_topic_for_researcher,
                name="discuss_topic",
                description="Discuss a research topic with interest and contribution levels",
            ))
            
            # Most researchers can propose topics (except some newcomers)
            if researcher_profile.team_role != TeamRole.NEWCOMER or researcher_profile.academic_position in [AcademicPosition.POSTDOC]:
                def propose_topic_for_researcher(title: str, description: str, source: str, required_expertise: str, priority_level: str = "medium") -> str:
                    return propose_topic(researcher_name, title, description, source, required_expertise, priority_level)
                
                tools.append(FunctionTool(
                    propose_topic_for_researcher,
                    name="propose_topic",
                    description="Propose a new research topic for collaboration",
                ))
            
            # Only leaders can assign topics
            if researcher_profile.team_role in [TeamRole.LEADER, TeamRole.CO_LEADER]:
                def assign_to_topic_for_researcher(topic_id: str, assignee_name: str, reasoning: str) -> str:
                    return assign_to_topic(researcher_name, topic_id, assignee_name, reasoning)
                
                tools.append(FunctionTool(
                    assign_to_topic_for_researcher,
                    name="assign_to_topic",
                    description="Assign a team member to a research topic",
                ))
            
            # All researchers can check status and profiles
            tools.extend([
                FunctionTool(
                    get_current_topics,
                    name="get_current_topics",
                    description="Get information about all proposed topics and their status",
                ),
                FunctionTool(
                    get_researcher_profiles,
                    name="get_researcher_profiles",
                    description="Get information about all team members and their roles",
                ),
                FunctionTool(
                    get_collaboration_status,
                    name="get_collaboration_status", 
                    description="Get the current status of the collaboration process",
                ),
            ])
            
            return tools
        
        researcher_tools = make_researcher_tools(researcher.name, researcher)
        
        # Register tool agent for this researcher
        await ToolAgent.register(
            runtime,
            f"CollaborationToolAgent_{researcher_safe_name}",
            lambda tools=researcher_tools: ToolAgent(description="Tool agent for enhanced scientific collaboration.", tools=tools),
        )
        
        # Register researcher agent
        await ResearcherAgent.register(
            runtime,
            f"Researcher_{researcher_safe_name}",
            lambda r=researcher, safe_name=researcher_safe_name, tools=researcher_tools: ResearcherAgent(
                profile=r,
                model_client=model_client,
                model_context=BufferedChatCompletionContext(buffer_size=10),  # Reduced buffer to prevent context issues
                tool_schema=[tool.schema for tool in tools],
                tool_agent_type=f"CollaborationToolAgent_{safe_name}",
            ),
        )


async def run_collaboration_round(
    runtime: AgentRuntime, 
    round_num: int, 
    phase: str,
    researchers: List[str]
) -> bool:
    """Run a single round of collaboration discussion. Returns True if should continue."""
    
    print(f"\n{'='*60}")
    print(f"🔄 COLLABORATION ROUND {round_num}: {phase.upper()}")
    print(f"{'='*60}")
    
    collaboration_state.discussion_round = round_num
    
    # Check if we should terminate this phase
    if collaboration_state.should_terminate_phase():
        print(f"⏱️ Phase {phase} terminated after {collaboration_state.phase_round_count} rounds")
        return False
    
    # Reset message counts for all agents
    for researcher_name in researchers:
        researcher_id = AgentId(f"Researcher_{researcher_name}", "default")
        try:
            agent = runtime._agent_instances.get(researcher_id)
            if hasattr(agent, 'reset_message_count'):
                agent.reset_message_count()
        except:
            pass  # Agent might not be initialized yet
    
    # Different prompts for different phases
    if phase == "introduction":
        prompt = ("Please introduce yourself, highlighting your role in the team, academic position, expertise, "
                 "and current workload. Mention your collaboration history and suggest how you might contribute "
                 "to the team's research directions. Keep your introduction concise.")
    elif phase == "proposal":
        prompt = ("Based on the introductions and your role in the team, propose 1-2 specific research topics. "
                 "Consider your position: professors might propose granted projects or assign directions, "
                 "postdocs and incumbents might suggest research expansions, while newcomers should focus on "
                 "learning opportunities. Include the topic source and required expertise.")
    elif phase == "discussion":
        prompt = ("Review the proposed topics and assess them based on your expertise, interests, and current "
                 "workload. For each relevant topic, indicate your interest level (high/medium/low/none) and "
                 "potential contribution level (lead/significant/moderate/minimal). Leaders should consider "
                 "assignments based on team members' capabilities and workloads.")
    elif phase == "consensus":
        prompt = ("Review the discussion results and work towards final consensus. Leaders should make topic "
                 "assignments considering team members' expressed interests and capabilities. Discuss next steps, "
                 "timelines, and responsibilities for the selected research directions.")
    else:
        prompt = f"Continue the collaboration discussion for phase: {phase}"
    
    # Send messages to researchers based on hierarchy (leaders first, then others)
    leader_researchers = []
    other_researchers = []
    
    for researcher in researchers:
        if researcher in collaboration_state.researchers:
            role = collaboration_state.researchers[researcher].team_role
            if role in [TeamRole.LEADER, TeamRole.CO_LEADER]:
                leader_researchers.append(researcher)
            else:
                other_researchers.append(researcher)
    
    # Randomize within each group to add variety
    random.shuffle(leader_researchers)
    random.shuffle(other_researchers)
    
    ordered_researchers = leader_researchers + other_researchers
    
    for i, researcher in enumerate(ordered_researchers):
        message = CollaborationMessage(
            content=prompt,
            sender="Moderator",
            message_type=phase,
            round_number=round_num
        )
        
        researcher_id = AgentId(f"Researcher_{researcher}", "default")
        try:
            await asyncio.wait_for(
                runtime.send_message(message, researcher_id),
                timeout=10.0  # 10 second timeout per message
            )
        except asyncio.TimeoutError:
            print(f"⚠️ Message to {researcher} timed out")
        except Exception as e:
            print(f"⚠️ Error sending message to {researcher}: {e}")
        
        # Add delay between messages to prevent overwhelming and allow processing
        if i < len(ordered_researchers) - 1:  # Don't wait after last message
            await asyncio.sleep(2)
    
    collaboration_state.increment_phase_round()
    return True


async def main(config_file: str = ".server_deployed_LLMs", config_section: str = "ali_official", num_rounds: int = 3) -> None:
    """Main entry point for the enhanced scientific collaboration simulation."""
    
    print("🧪 SCIENTIFIC COLLABORATION V2 SIMULATION")
    print("=" * 60)
    print("Enhanced academic research collaboration with:")
    print("• Team roles: leader, co-leader, incumbent, newcomer")
    print("• Academic positions: professor, associate prof, assistant prof, postdoc, PhD")
    print("• Discussion-based consensus (no voting)")
    print("• Topic sources and assignment mechanisms")
    print("• Improved asyncio handling and termination conditions")
    print()
    
    # Initialize runtime and model
    runtime = SingleThreadedAgentRuntime()
    model_client = create_model_client_from_config(config_file, config_section)
    
    # Set up the collaboration
    await setup_collaboration(runtime, model_client)
    
    # Start the runtime
    runtime.start()
    
    # Get researcher names
    researcher_names = [
        "Prof_Chen_001",
        "Dr_Wilson_002", 
        "Dr_Garcia_003",
        "Dr_Kim_004",
        "PhD_Zhang_005",
        "Postdoc_Lee_006"
    ]
    
    try:
        # Run collaboration phases
        phases = ["introduction", "proposal", "discussion", "consensus"]
        
        for phase in phases:
            collaboration_state.reset_phase_counter(phase)
            print(f"\n🎯 STARTING PHASE: {phase.upper()}")
            
            # Run multiple rounds per phase with termination conditions
            round_in_phase = 0
            while round_in_phase < collaboration_state.max_rounds_per_phase:
                round_in_phase += 1
                should_continue = await run_collaboration_round(
                    runtime, 
                    collaboration_state.discussion_round + 1, 
                    phase, 
                    researcher_names
                )
                
                if not should_continue:
                    break
                
                # Wait for discussions to settle with shorter delays
                await asyncio.sleep(3)
                
                # Show status after key phases
                if phase in ["proposal", "discussion"] and round_in_phase == 1:
                    print(f"\n📊 STATUS AFTER {phase.upper()} PHASE:")
                    print(get_collaboration_status())
                    print()
        
        # Final summary
        print(f"\n🎯 FINAL COLLABORATION RESULTS")
        print("=" * 60)
        print(get_current_topics())
        print(get_collaboration_status())
        
        # Show final assignments
        assigned_topics = [topic for topic in collaboration_state.topics.values() if topic.assigned_members]
        if assigned_topics:
            print("\n🏆 FINAL TOPIC ASSIGNMENTS:")
            for topic in assigned_topics:
                print(f"📋 {topic.title}")
                print(f"   👤 Proposed by: {topic.proposer}")
                print(f"   🎯 Source: {topic.source.value}")
                print(f"   👥 Assigned to: {', '.join(topic.assigned_members)}")
                print(f"   ⚡ Priority: {topic.priority_level}")
                print()
        else:
            print("\n📝 No topics were assigned during this collaboration.")
                
    except KeyboardInterrupt:
        print("\n⚠️ Collaboration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during collaboration: {e}")
    finally:
        # Clean shutdown with timeout
        try:
            await asyncio.wait_for(runtime.stop_when_idle(), timeout=10.0)
        except asyncio.TimeoutError:
            print("⚠️ Runtime shutdown timed out")
        
        try:
            await asyncio.wait_for(model_client.close(), timeout=5.0)
        except asyncio.TimeoutError:
            print("⚠️ Model client shutdown timed out")
        except:
            pass  # Client might already be closed
        
        print("\n✅ Enhanced collaboration simulation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an enhanced scientific collaboration simulation.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--config-file", type=str, help="Path to the model configuration file.", default=".server_deployed_LLMs"
    )
    parser.add_argument(
        "--config-section", type=str, help="Configuration section to use.", default="ali_official"
    )
    parser.add_argument(
        "--num-rounds", type=int, help="Maximum rounds per phase.", default=3
    )
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("autogen_core").setLevel(logging.DEBUG)
        handler = logging.FileHandler("collaboration_v2.log")
        logging.getLogger("autogen_core").addHandler(handler)

    try:
        asyncio.run(main(args.config_file, args.config_section, args.num_rounds))
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Please ensure the configuration file '{args.config_file}' exists and contains the section '{args.config_section}'.")
        print("\n💡 Tip: To see how this enhanced collaboration works without an API key, run:")
        print("    python demo.py")
        sys.exit(1)
    except Exception as e:
        if "api_key" in str(e).lower() or "authentication" in str(e).lower() or "connection" in str(e).lower():
            print(f"❌ Error: {e}")
            print("\n💡 This appears to be an API key or connection issue.")
            print("To test the enhanced collaboration system without an API key, run:")
            print("    python demo.py")
            sys.exit(1)
        else:
            print(f"❌ Error: {e}")
            raise