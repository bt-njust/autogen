"""Bibliographic Data Scientific Collaboration Example

This example demonstrates how AutoGen can simulate realistic academic research scenarios
for structural biology research teams identified through community detection algorithms,
based on bibliographic data and coauthorship networks.

The simulation includes:
- Researchers with roles based on publication patterns (first/corresponding author)
- Network-based role definitions from coauthorship analysis
- Community detection integration for team formation
- Publication history and journal information tracking
- Template-based prompts with placeholders for computed metrics
- Affiliation tracking and institutional collaboration patterns
"""

import argparse
import asyncio
import configparser
import logging
import random
import sys
from typing import Annotated, Any, Dict, List, Optional, Tuple
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


class AuthorRole(Enum):
    """Publication-based author roles."""
    FIRST_AUTHOR_FOCUSED = "first_author_focused"
    CORRESPONDING_AUTHOR_FOCUSED = "corresponding_author_focused"
    COLLABORATIVE_AUTHOR = "collaborative_author"
    MIDDLE_AUTHOR = "middle_author"
    SENIOR_AUTHOR = "senior_author"


class NetworkRole(Enum):
    """Network-based researcher roles."""
    HUB_CONNECTOR = "hub_connector"  # High betweenness centrality
    COMMUNITY_LEADER = "community_leader"  # High degree centrality within community
    BRIDGE_BUILDER = "bridge_builder"  # Connects different communities
    SPECIALIST = "specialist"  # High clustering coefficient
    NEWCOMER = "newcomer"  # Low overall network metrics


@dataclass
class Publication:
    """Represents a publication with detailed bibliographic information."""
    title: str
    journal: str
    year: int
    authors: List[str]
    first_author: str
    corresponding_author: str
    topic_keywords: List[str] = field(default_factory=list)
    impact_factor: Optional[float] = None
    citation_count: Optional[int] = None


@dataclass
class NetworkMetrics:
    """Network-based metrics for a researcher."""
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    community_id: Optional[str] = None
    # Placeholders for computed metrics
    collaboration_frequency: str = "{COLLABORATION_FREQUENCY}"
    network_position_score: str = "{NETWORK_POSITION_SCORE}"
    cross_community_connections: str = "{CROSS_COMMUNITY_CONNECTIONS}"


@dataclass
class PublicationPatterns:
    """Publication pattern analysis for a researcher."""
    total_publications: int = 0
    first_author_count: int = 0
    corresponding_author_count: int = 0
    collaborative_papers: int = 0
    # Placeholders for computed metrics
    first_author_ratio: str = "{FIRST_AUTHOR_RATIO}"
    corresponding_author_ratio: str = "{CORRESPONDING_AUTHOR_RATIO}"
    average_coauthors: str = "{AVERAGE_COAUTHORS}"
    research_productivity_score: str = "{RESEARCH_PRODUCTIVITY_SCORE}"


@dataclass
class AffiliationHistory:
    """Tracks researcher's institutional affiliations."""
    institution: str
    start_year: int
    end_year: Optional[int] = None
    position: Optional[str] = None
    country: Optional[str] = None


@dataclass
class BiblioCollaborationHistory:
    """Enhanced collaboration history with bibliographic details."""
    collaborators: List[str]
    publications: List[Publication]
    collaboration_strength: float  # Number of joint publications
    topic_overlap: List[str]  # Shared research topics
    duration_years: int
    institutional_type: str  # same_institution, cross_institution, international


@dataclass
class EnhancedResearcherProfile:
    """Enhanced researcher profile with bibliographic data insights."""
    name: str
    current_affiliation: str
    expertise_areas: List[str]
    research_interests: List[str]
    
    # Publication-based attributes
    publication_patterns: PublicationPatterns
    recent_publications: List[Publication]
    preferred_journals: List[str] = field(default_factory=list)
    
    # Network-based attributes  
    network_metrics: NetworkMetrics
    author_role: AuthorRole = AuthorRole.COLLABORATIVE_AUTHOR
    network_role: NetworkRole = NetworkRole.SPECIALIST
    
    # Collaboration history
    affiliation_history: List[AffiliationHistory] = field(default_factory=list)
    collaboration_history: List[BiblioCollaborationHistory] = field(default_factory=list)
    
    # Community information
    research_community: Optional[str] = None
    community_connections: Dict[str, float] = field(default_factory=dict)
    
    # Template placeholders for dynamic content
    specialization_score: str = "{SPECIALIZATION_SCORE}"
    influence_index: str = "{INFLUENCE_INDEX}"
    collaboration_diversity: str = "{COLLABORATION_DIVERSITY}"


class BiblioCollaborationMessage(BaseModel):
    """Message for bibliographic-based collaboration discussions."""
    content: str
    sender: str
    message_type: str  # "introduction", "proposal", "analysis", "vote", "consensus"
    topic_id: Optional[str] = None
    community_context: Optional[str] = None


@dataclass
class ResearchTopic:
    """Enhanced research topic with bibliographic context."""
    title: str
    description: str
    proposer: str
    required_expertise: List[str] = field(default_factory=list)
    votes: int = 0
    supporters: List[str] = field(default_factory=list)
    # Enhanced fields for bibliographic context
    target_journals: List[str] = field(default_factory=list)
    interdisciplinary_score: float = 0.0
    feasibility_factors: List[str] = field(default_factory=list)
    potential_collaborators: List[str] = field(default_factory=list)


class BiblioCollaborationState:
    """Manages the state of the bibliographic-based scientific collaboration."""
    
    def __init__(self, max_topics: int = 8):
        self.researchers: Dict[str, EnhancedResearcherProfile] = {}
        self.topics: Dict[str, ResearchTopic] = {}
        self.communities: Dict[str, List[str]] = {}  # community_id -> researcher names
        self.discussion_round = 0
        self.consensus_reached = False
        self.selected_topics: List[str] = []
        self.max_topics = max_topics
        
    def add_researcher(self, profile: EnhancedResearcherProfile):
        """Add a researcher to the collaboration."""
        self.researchers[profile.name] = profile
        # Add to community tracking
        if profile.research_community:
            if profile.research_community not in self.communities:
                self.communities[profile.research_community] = []
            self.communities[profile.research_community].append(profile.name)
        
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
            
    def get_community_members(self, community_id: str) -> List[str]:
        """Get researchers in a specific community."""
        return self.communities.get(community_id, [])
        
    def get_cross_community_collaborations(self) -> List[Tuple[str, str]]:
        """Identify potential cross-community collaborations."""
        collaborations = []
        communities = list(self.communities.keys())
        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                collaborations.append((communities[i], communities[j]))
        return collaborations
        
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
        'json_output': True
    }

    model_client = OpenAIChatCompletionClient(
        model="qwen-max",
        base_url=base_url,
        api_key=api_key,
        model_info=model_info
    )
    
    return model_client


# Global collaboration state
biblio_collaboration_state = BiblioCollaborationState()


@default_subscription
class BiblioResearcherAgent(RoutedAgent):
    """Agent representing a researcher with bibliographic data insights."""
    
    def __init__(
        self,
        profile: EnhancedResearcherProfile,
        model_client: ChatCompletionClient,
        model_context: ChatCompletionContext,
        tool_schema: List[ToolSchema],
        tool_agent_type: str,
    ) -> None:
        super().__init__(description=f"Bibliographic Researcher: {profile.name}")
        self.profile = profile
        self._model_client = model_client
        self._model_context = model_context
        self._tool_schema = tool_schema
        self._tool_agent_id = AgentId(tool_agent_type, self.id.key)
        
        # Create system message with enhanced researcher's profile
        system_prompt = self._create_system_prompt()
        self._system_messages: List[LLMMessage] = [SystemMessage(content=system_prompt)]
        
    def _create_system_prompt(self) -> str:
        """Create a comprehensive system prompt based on the researcher's bibliographic profile."""
        
        # Publication pattern analysis
        pub_analysis = f"""
Publication Profile Analysis:
- Total publications: {self.profile.publication_patterns.total_publications}
- First author papers: {self.profile.publication_patterns.first_author_count}
- Corresponding author papers: {self.profile.publication_patterns.corresponding_author_count}
- First author ratio: {self.profile.publication_patterns.first_author_ratio}
- Corresponding author ratio: {self.profile.publication_patterns.corresponding_author_ratio}
- Average coauthors per paper: {self.profile.publication_patterns.average_coauthors}
- Research productivity score: {self.profile.publication_patterns.research_productivity_score}
"""

        # Network position analysis
        network_analysis = f"""
Network Position Analysis:
- Degree centrality: {self.profile.network_metrics.degree_centrality:.3f}
- Betweenness centrality: {self.profile.network_metrics.betweenness_centrality:.3f}
- Clustering coefficient: {self.profile.network_metrics.clustering_coefficient:.3f}
- Research community: {self.profile.research_community}
- Author role: {self.profile.author_role.value}
- Network role: {self.profile.network_role.value}
- Collaboration frequency: {self.profile.network_metrics.collaboration_frequency}
- Network position score: {self.profile.network_metrics.network_position_score}
- Cross-community connections: {self.profile.network_metrics.cross_community_connections}
"""

        # Research specialization
        specialization = f"""
Research Specialization:
- Primary expertise: {', '.join(self.profile.expertise_areas)}
- Research interests: {', '.join(self.profile.research_interests)}
- Preferred journals: {', '.join(self.profile.preferred_journals)}
- Specialization score: {self.profile.specialization_score}
- Influence index: {self.profile.influence_index}
- Collaboration diversity: {self.profile.collaboration_diversity}
"""

        # Recent collaboration history
        collab_history = ""
        if self.profile.collaboration_history:
            collab_history = "\n\nRecent Collaboration Patterns:\n"
            for collab in self.profile.collaboration_history[-3:]:  # Show last 3
                collab_history += f"- {len(collab.publications)} papers with {', '.join(collab.collaborators)} "
                collab_history += f"({collab.collaboration_strength:.1f} strength, {collab.duration_years} years, {collab.institutional_type})\n"
                collab_history += f"  Topics: {', '.join(collab.topic_overlap)}\n"

        # Recent publications
        recent_pubs = ""
        if self.profile.recent_publications:
            recent_pubs = "\n\nRecent Publications:\n"
            for pub in self.profile.recent_publications[-5:]:  # Show last 5
                recent_pubs += f"- {pub.title} ({pub.year}) - {pub.journal}\n"
                recent_pubs += f"  Role: {'First author' if pub.first_author == self.profile.name else 'Corresponding' if pub.corresponding_author == self.profile.name else 'Coauthor'}\n"

        return f"""You are {self.profile.name}, a structural biology researcher from {self.profile.current_affiliation}.

{specialization}

{pub_analysis}

{network_analysis}

{collab_history}

{recent_pubs}

You are participating in a scientific collaboration meeting to discuss and select joint research topics based on bibliographic data analysis and community detection insights.

Behavioral Guidelines Based on Your Profile:
1. **Publication Role Behavior**: As a {self.profile.author_role.value}, {'you often take initiative in proposing and leading research directions' if self.profile.author_role in [AuthorRole.FIRST_AUTHOR_FOCUSED, AuthorRole.CORRESPONDING_AUTHOR_FOCUSED] else 'you prefer to contribute expertise and support collaborative efforts'}

2. **Network Role Behavior**: As a {self.profile.network_role.value}, {'you focus on connecting different research groups and facilitating interdisciplinary collaboration' if self.profile.network_role == NetworkRole.BRIDGE_BUILDER else 'you leverage your strong community connections' if self.profile.network_role == NetworkRole.COMMUNITY_LEADER else 'you contribute specialized knowledge and deep expertise' if self.profile.network_role == NetworkRole.SPECIALIST else 'you seek opportunities to expand your research network and learn from established researchers' if self.profile.network_role == NetworkRole.NEWCOMER else 'you facilitate connections between different research areas and communities'}

3. **Research Collaboration Strategy**: 
   - Draw from your publication pattern insights when evaluating collaboration potential
   - Consider journal preferences and impact factors when proposing research directions
   - Leverage your network position to identify complementary expertise
   - Factor in your collaboration history and successful partnerships

4. **Community Dynamics**: 
   - Represent the interests and expertise of your research community ({self.profile.research_community})
   - Consider cross-community collaboration opportunities
   - Balance community loyalty with innovative interdisciplinary possibilities

5. **Decision Making Criteria**:
   - Feasibility based on your publication track record
   - Alignment with your specialization and journal preferences
   - Potential for high-impact outcomes given your network position
   - Opportunity to strengthen or expand your collaboration network

Use the available tools to:
- Propose research topics that align with your bibliographic profile
- Vote on proposals considering collaboration potential and expertise fit
- Analyze current topics through the lens of your publication and network insights
- Share community perspectives and cross-disciplinary opportunities"""

    @message_handler
    async def handle_message(self, message: BiblioCollaborationMessage, ctx: MessageContext) -> None:
        """Handle incoming collaboration messages with bibliographic context."""
        # Add community context to the message if relevant
        context_info = ""
        if message.community_context:
            context_info = f" [Community: {message.community_context}]"
            
        # Add the message to model context
        await self._model_context.add_message(
            UserMessage(content=f"[{message.message_type.upper()}]{context_info} {message.sender}: {message.content}", 
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
            
        # Publish response with community context
        if messages and isinstance(messages[-1].content, str):
            response = BiblioCollaborationMessage(
                content=messages[-1].content,
                sender=self.profile.name,
                message_type="discussion",
                community_context=self.profile.research_community
            )
            await self.publish_message(response, DefaultTopicId())


def propose_biblio_topic(
    researcher_name: str,
    title: Annotated[str, "Title of the research topic"],
    description: Annotated[str, "Detailed description of the research topic"],
    required_expertise: Annotated[str, "Comma-separated list of expertise areas needed"],
    target_journals: Annotated[str, "Comma-separated list of target journals for publication"],
    interdisciplinary_factors: Annotated[str, "Factors that make this topic interdisciplinary"],
) -> Annotated[str, "Result of the topic proposal"]:
    """Propose a new research topic for bibliographic-based collaboration."""
    # Check if we've reached the maximum number of topics
    if len(biblio_collaboration_state.topics) >= biblio_collaboration_state.max_topics:
        return f"Cannot propose new topic '{title}'. Maximum topic limit ({biblio_collaboration_state.max_topics}) has been reached."
    
    expertise_list = [exp.strip() for exp in required_expertise.split(",")]
    journal_list = [journal.strip() for journal in target_journals.split(",")]
    
    # Get researcher profile for enhanced topic creation
    researcher_profile = biblio_collaboration_state.researchers.get(researcher_name)
    
    topic = ResearchTopic(
        title=title,
        description=description,
        proposer=researcher_name,
        required_expertise=expertise_list,
        target_journals=journal_list,
        feasibility_factors=[interdisciplinary_factors],
    )
    
    # Add potential collaborators based on expertise overlap
    if researcher_profile:
        for name, profile in biblio_collaboration_state.researchers.items():
            if name != researcher_name:
                # Check expertise overlap
                overlap = set(expertise_list) & set(profile.expertise_areas)
                if overlap:
                    topic.potential_collaborators.append(name)
    
    topic_id = biblio_collaboration_state.add_topic(topic)
    
    if topic_id is None:
        return f"Cannot propose new topic '{title}'. Maximum topic limit has been reached."
    
    remaining_slots = biblio_collaboration_state.get_remaining_topic_slots()
    
    print(f"\n🔬 NEW BIBLIOGRAPHIC TOPIC PROPOSED by {researcher_name}")
    print(f"📋 Title: {title}")
    print(f"📝 Description: {description}")
    print(f"🎯 Required expertise: {', '.join(expertise_list)}")
    print(f"📚 Target journals: {', '.join(journal_list)}")
    print(f"🌐 Interdisciplinary factors: {interdisciplinary_factors}")
    print(f"🤝 Potential collaborators: {', '.join(topic.potential_collaborators)}")
    print(f"🆔 Topic ID: {topic_id}")
    print(f"📊 Remaining topic slots: {remaining_slots}")
    
    return f"Successfully proposed bibliographic topic '{title}' (ID: {topic_id}). Target journals: {', '.join(journal_list)}. {remaining_slots} topic slots remaining."


def vote_for_biblio_topic(
    voter_name: str,
    topic_id: Annotated[str, "ID of the topic to vote for"],
    reasoning: Annotated[str, "Reasoning for the vote based on bibliographic insights"],
    collaboration_potential: Annotated[str, "Assessment of collaboration potential"],
) -> Annotated[str, "Result of the vote"]:
    """Vote for a research topic with bibliographic reasoning."""
    if topic_id not in biblio_collaboration_state.topics:
        return f"Topic {topic_id} does not exist."
        
    if voter_name in biblio_collaboration_state.topics[topic_id].supporters:
        return f"You have already voted for topic {topic_id}."
        
    biblio_collaboration_state.vote_for_topic(topic_id, voter_name)
    topic = biblio_collaboration_state.topics[topic_id]
    
    # Get voter profile for enhanced feedback
    voter_profile = biblio_collaboration_state.researchers.get(voter_name)
    community_note = f" [Community: {voter_profile.research_community}]" if voter_profile and voter_profile.research_community else ""
    
    print(f"\n🗳️ BIBLIOGRAPHIC VOTE CAST by {voter_name}{community_note}")
    print(f"📋 Topic: {topic.title}")
    print(f"💭 Reasoning: {reasoning}")
    print(f"🤝 Collaboration potential: {collaboration_potential}")
    print(f"📊 Current votes: {topic.votes}")
    print(f"👥 Supporters: {', '.join(topic.supporters)}")
    
    return f"Successfully voted for '{topic.title}'. Collaboration assessment: {collaboration_potential}. Current votes: {topic.votes}"


def get_current_biblio_topics() -> Annotated[str, "List of all proposed topics with bibliographic context"]:
    """Get information about all currently proposed topics with bibliographic insights."""
    if not biblio_collaboration_state.topics:
        return "No topics have been proposed yet."
        
    topics_info = "Current Bibliographic Research Topics:\n\n"
    for topic_id, topic in biblio_collaboration_state.topics.items():
        topics_info += f"🆔 {topic_id} | 📋 {topic.title}\n"
        topics_info += f"   👤 Proposed by: {topic.proposer}\n"
        topics_info += f"   📝 Description: {topic.description}\n"
        topics_info += f"   🎯 Required expertise: {', '.join(topic.required_expertise)}\n"
        topics_info += f"   📚 Target journals: {', '.join(topic.target_journals)}\n"
        topics_info += f"   🤝 Potential collaborators: {', '.join(topic.potential_collaborators)}\n"
        topics_info += f"   🗳️ Votes: {topic.votes} | 👥 Supporters: {', '.join(topic.supporters)}\n\n"
        
    return topics_info


def get_researcher_biblio_profiles() -> Annotated[str, "Information about all researchers with bibliographic insights"]:
    """Get information about all researchers with their bibliographic profiles."""
    if not biblio_collaboration_state.researchers:
        return "No researchers are currently registered."
        
    profiles_info = "Bibliographic Researcher Profiles:\n\n"
    for name, profile in biblio_collaboration_state.researchers.items():
        profiles_info += f"👤 {profile.name} ({profile.current_affiliation})\n"
        profiles_info += f"   🔬 Expertise: {', '.join(profile.expertise_areas)}\n"
        profiles_info += f"   🎯 Interests: {', '.join(profile.research_interests)}\n"
        profiles_info += f"   📄 Publications: {profile.publication_patterns.total_publications}\n"
        profiles_info += f"   📝 Author role: {profile.author_role.value}\n"
        profiles_info += f"   🌐 Network role: {profile.network_role.value}\n"
        profiles_info += f"   🏛️ Community: {profile.research_community}\n"
        profiles_info += f"   📚 Preferred journals: {', '.join(profile.preferred_journals[:3])}{'...' if len(profile.preferred_journals) > 3 else ''}\n"
        if profile.collaboration_history:
            profiles_info += f"   🤝 Collaborations: {len(profile.collaboration_history)} active partnerships\n"
        profiles_info += "\n"
        
    return profiles_info


def get_community_analysis() -> Annotated[str, "Analysis of research communities and cross-community opportunities"]:
    """Get analysis of research communities and collaboration opportunities."""
    if not biblio_collaboration_state.communities:
        return "No research communities have been identified."
        
    analysis = "Research Community Analysis:\n\n"
    
    for community_id, members in biblio_collaboration_state.communities.items():
        analysis += f"🏛️ Community: {community_id}\n"
        analysis += f"   👥 Members ({len(members)}): {', '.join(members)}\n"
        
        # Analyze community expertise
        all_expertise = set()
        for member in members:
            if member in biblio_collaboration_state.researchers:
                all_expertise.update(biblio_collaboration_state.researchers[member].expertise_areas)
        analysis += f"   🔬 Collective expertise: {', '.join(list(all_expertise)[:5])}{'...' if len(all_expertise) > 5 else ''}\n"
        analysis += "\n"
    
    # Cross-community opportunities
    cross_community_pairs = biblio_collaboration_state.get_cross_community_collaborations()
    if cross_community_pairs:
        analysis += "🌐 Cross-Community Collaboration Opportunities:\n"
        for comm1, comm2 in cross_community_pairs[:3]:  # Show top 3
            analysis += f"   {comm1} ↔ {comm2}: Potential for interdisciplinary research\n"
        analysis += "\n"
    
    return analysis


def get_biblio_collaboration_status() -> Annotated[str, "Current status of the bibliographic collaboration"]:
    """Get the current status of the collaboration process with bibliographic insights."""
    status = f"Bibliographic Collaboration Status:\n"
    status += f"👥 Researchers: {len(biblio_collaboration_state.researchers)}\n"
    status += f"🏛️ Communities: {len(biblio_collaboration_state.communities)}\n"
    status += f"📋 Topics proposed: {len(biblio_collaboration_state.topics)}/{biblio_collaboration_state.max_topics}\n"
    status += f"📊 Remaining topic slots: {biblio_collaboration_state.get_remaining_topic_slots()}\n"
    status += f"🔄 Discussion round: {biblio_collaboration_state.discussion_round}\n"
    
    if biblio_collaboration_state.topics:
        top_topics = biblio_collaboration_state.get_top_topics(3)
        status += f"\n🏆 Top Topics by Votes:\n"
        for i, topic in enumerate(top_topics, 1):
            status += f"   {i}. {topic.title} ({topic.votes} votes)\n"
            status += f"      Target journals: {', '.join(topic.target_journals[:2])}\n"
            
    return status


# Sample data creation functions

def create_sample_publications() -> List[Publication]:
    """Create sample publications for demonstration."""
    return [
        Publication(
            title="Protein Structure Prediction Using Deep Learning",
            journal="Nature Structural & Molecular Biology",
            year=2023,
            authors=["Dr. Sarah Chen", "Dr. James Wilson", "Dr. Maria Garcia"],
            first_author="Dr. Sarah Chen",
            corresponding_author="Dr. James Wilson",
            topic_keywords=["protein folding", "deep learning", "structural biology"],
            impact_factor=15.2,
            citation_count=45
        ),
        Publication(
            title="Cryo-EM Analysis of Membrane Protein Complexes",
            journal="Cell",
            year=2023,
            authors=["Dr. Alex Kim", "Dr. Lisa Zhang", "Dr. Sarah Chen"],
            first_author="Dr. Alex Kim",
            corresponding_author="Dr. Alex Kim",
            topic_keywords=["cryo-EM", "membrane proteins", "structural analysis"],
            impact_factor=41.8,
            citation_count=67
        ),
        Publication(
            title="Machine Learning Approaches to Drug-Target Interactions",
            journal="Journal of Medicinal Chemistry",
            year=2022,
            authors=["Dr. Maria Garcia", "Dr. James Wilson"],
            first_author="Dr. Maria Garcia",
            corresponding_author="Dr. James Wilson",
            topic_keywords=["drug discovery", "machine learning", "protein interactions"],
            impact_factor=7.3,
            citation_count=89
        )
    ]


def create_sample_researcher_profiles() -> List[EnhancedResearcherProfile]:
    """Create sample enhanced researcher profiles with bibliographic data."""
    sample_pubs = create_sample_publications()
    
    researchers = [
        EnhancedResearcherProfile(
            name="Dr. Sarah Chen",
            current_affiliation="MIT Department of Biology",
            expertise_areas=["Structural Biology", "Protein Folding", "Computational Biology"],
            research_interests=["AI-Driven Structure Prediction", "Membrane Protein Dynamics", "Drug Design"],
            publication_patterns=PublicationPatterns(
                total_publications=67,
                first_author_count=23,
                corresponding_author_count=12,
                collaborative_papers=52
            ),
            recent_publications=[sample_pubs[0], sample_pubs[2]],
            preferred_journals=["Nature Structural & Molecular Biology", "PNAS", "Structure"],
            network_metrics=NetworkMetrics(
                degree_centrality=0.75,
                betweenness_centrality=0.45,
                closeness_centrality=0.68,
                clustering_coefficient=0.72,
                community_id="structural_biology_ai"
            ),
            author_role=AuthorRole.FIRST_AUTHOR_FOCUSED,
            network_role=NetworkRole.HUB_CONNECTOR,
            affiliation_history=[
                AffiliationHistory("MIT", 2020, position="Assistant Professor", country="USA"),
                AffiliationHistory("Stanford University", 2017, 2020, "Postdoc", "USA")
            ],
            research_community="Computational Structural Biology",
            community_connections={"Computational Structural Biology": 0.85, "AI in Biology": 0.62}
        ),
        
        EnhancedResearcherProfile(
            name="Dr. James Wilson",
            current_affiliation="Stanford University School of Medicine",
            expertise_areas=["Protein Biochemistry", "Drug Discovery", "Enzyme Kinetics"],
            research_interests=["Therapeutic Target Identification", "Allosteric Regulation", "Protein Engineering"],
            publication_patterns=PublicationPatterns(
                total_publications=89,
                first_author_count=15,
                corresponding_author_count=34,
                collaborative_papers=74
            ),
            recent_publications=[sample_pubs[0], sample_pubs[2]],
            preferred_journals=["Journal of Biological Chemistry", "Biochemistry", "Journal of Medicinal Chemistry"],
            network_metrics=NetworkMetrics(
                degree_centrality=0.82,
                betweenness_centrality=0.35,
                closeness_centrality=0.78,
                clustering_coefficient=0.65,
                community_id="drug_discovery"
            ),
            author_role=AuthorRole.CORRESPONDING_AUTHOR_FOCUSED,
            network_role=NetworkRole.COMMUNITY_LEADER,
            affiliation_history=[
                AffiliationHistory("Stanford University", 2015, position="Associate Professor", country="USA"),
                AffiliationHistory("University of California, San Francisco", 2010, 2015, "Assistant Professor", "USA")
            ],
            research_community="Drug Discovery and Development",
            community_connections={"Drug Discovery and Development": 0.92, "Protein Biochemistry": 0.78}
        ),
        
        EnhancedResearcherProfile(
            name="Dr. Maria Garcia",
            current_affiliation="European Molecular Biology Laboratory",
            expertise_areas=["Cryo-Electron Microscopy", "Protein Complexes", "Structural Analysis"],
            research_interests=["Membrane Protein Structure", "Protein-Protein Interactions", "Method Development"],
            publication_patterns=PublicationPatterns(
                total_publications=45,
                first_author_count=18,
                corresponding_author_count=8,
                collaborative_papers=37
            ),
            recent_publications=[sample_pubs[1], sample_pubs[2]],
            preferred_journals=["eLife", "Nature Communications", "Structure"],
            network_metrics=NetworkMetrics(
                degree_centrality=0.68,
                betweenness_centrality=0.52,
                closeness_centrality=0.71,
                clustering_coefficient=0.69,
                community_id="cryo_em"
            ),
            author_role=AuthorRole.COLLABORATIVE_AUTHOR,
            network_role=NetworkRole.BRIDGE_BUILDER,
            affiliation_history=[
                AffiliationHistory("EMBL", 2019, position="Group Leader", country="Germany"),
                AffiliationHistory("MRC Laboratory of Molecular Biology", 2016, 2019, "Postdoc", "UK")
            ],
            research_community="Structural Biology Methods",
            community_connections={"Structural Biology Methods": 0.88, "Membrane Biology": 0.65}
        ),
        
        EnhancedResearcherProfile(
            name="Dr. Alex Kim",
            current_affiliation="Scripps Research Institute",
            expertise_areas=["Chemical Biology", "Protein Engineering", "Bioorthogonal Chemistry"],
            research_interests=["Protein Modification", "Tool Development", "Chemical Probes"],
            publication_patterns=PublicationPatterns(
                total_publications=52,
                first_author_count=28,
                corresponding_author_count=15,
                collaborative_papers=38
            ),
            recent_publications=[sample_pubs[1]],
            preferred_journals=["Nature Chemical Biology", "ACS Chemical Biology", "Chemical Science"],
            network_metrics=NetworkMetrics(
                degree_centrality=0.71,
                betweenness_centrality=0.38,
                closeness_centrality=0.69,
                clustering_coefficient=0.74,
                community_id="chemical_biology"
            ),
            author_role=AuthorRole.FIRST_AUTHOR_FOCUSED,
            network_role=NetworkRole.SPECIALIST,
            affiliation_history=[
                AffiliationHistory("Scripps Research", 2021, position="Assistant Professor", country="USA"),
                AffiliationHistory("Harvard University", 2018, 2021, "Postdoc", "USA")
            ],
            research_community="Chemical Biology",
            community_connections={"Chemical Biology": 0.91, "Bioorthogonal Chemistry": 0.83}
        ),
        
        EnhancedResearcherProfile(
            name="Dr. Lisa Zhang",
            current_affiliation="University of Cambridge",
            expertise_areas=["Single Molecule Biophysics", "Fluorescence Microscopy", "Protein Dynamics"],
            research_interests=["Real-time Protein Folding", "Molecular Motors", "Super-resolution Imaging"],
            publication_patterns=PublicationPatterns(
                total_publications=38,
                first_author_count=16,
                corresponding_author_count=9,
                collaborative_papers=29
            ),
            recent_publications=[sample_pubs[1]],
            preferred_journals=["Nature Methods", "Biophysical Journal", "PNAS"],
            network_metrics=NetworkMetrics(
                degree_centrality=0.64,
                betweenness_centrality=0.28,
                closeness_centrality=0.66,
                clustering_coefficient=0.81,
                community_id="biophysics"
            ),
            author_role=AuthorRole.COLLABORATIVE_AUTHOR,
            network_role=NetworkRole.SPECIALIST,
            affiliation_history=[
                AffiliationHistory("University of Cambridge", 2020, position="Lecturer", country="UK"),
                AffiliationHistory("University of Oxford", 2017, 2020, "Postdoc", "UK")
            ],
            research_community="Single Molecule Biophysics",
            community_connections={"Single Molecule Biophysics": 0.89, "Protein Dynamics": 0.76}
        )
    ]
    
    # Add collaboration histories
    researchers[0].collaboration_history = [
        BiblioCollaborationHistory(
            collaborators=["Dr. James Wilson"],
            publications=[sample_pubs[0], sample_pubs[2]],
            collaboration_strength=2.0,
            topic_overlap=["protein folding", "machine learning"],
            duration_years=3,
            institutional_type="cross_institution"
        )
    ]
    
    researchers[1].collaboration_history = [
        BiblioCollaborationHistory(
            collaborators=["Dr. Sarah Chen", "Dr. Maria Garcia"],
            publications=[sample_pubs[0], sample_pubs[2]],
            collaboration_strength=2.0,
            topic_overlap=["drug discovery", "structural biology"],
            duration_years=2,
            institutional_type="cross_institution"
        )
    ]
    
    return researchers


async def setup_biblio_collaboration(runtime: AgentRuntime, model_client: ChatCompletionClient) -> None:
    """Set up the bibliographic scientific collaboration with enhanced researcher profiles."""
    
    # Create sample researcher profiles with bibliographic data
    researchers = create_sample_researcher_profiles()
    
    # Add researchers to collaboration state
    for researcher in researchers:
        biblio_collaboration_state.add_researcher(researcher)
    
    # Register researcher agents and their specific tool agents
    for researcher in researchers:
        researcher_safe_name = researcher.name.replace(' ', '_').replace('.', '')
        
        # Create tools specific to this researcher
        def make_researcher_tools(researcher_name: str) -> List[Tool]:
            def propose_topic_for_researcher(title: str, description: str, required_expertise: str, target_journals: str, interdisciplinary_factors: str) -> str:
                return propose_biblio_topic(researcher_name, title, description, required_expertise, target_journals, interdisciplinary_factors)
            
            def vote_for_topic_for_researcher(topic_id: str, reasoning: str, collaboration_potential: str) -> str:
                return vote_for_biblio_topic(researcher_name, topic_id, reasoning, collaboration_potential)
            
            return [
                FunctionTool(
                    propose_topic_for_researcher,
                    name="propose_topic",
                    description="Propose a new research topic with bibliographic context",
                ),
                FunctionTool(
                    vote_for_topic_for_researcher,
                    name="vote_for_topic", 
                    description="Vote for a research topic with bibliographic reasoning and collaboration assessment",
                ),
                FunctionTool(
                    get_current_biblio_topics,
                    name="get_current_topics",
                    description="Get information about all proposed topics with bibliographic insights",
                ),
                FunctionTool(
                    get_researcher_biblio_profiles,
                    name="get_researcher_profiles",
                    description="Get information about all researchers with bibliographic profiles",
                ),
                FunctionTool(
                    get_community_analysis,
                    name="get_community_analysis",
                    description="Get analysis of research communities and cross-community collaboration opportunities",
                ),
                FunctionTool(
                    get_biblio_collaboration_status,
                    name="get_collaboration_status", 
                    description="Get the current status of the bibliographic collaboration",
                ),
            ]
        
        researcher_tools = make_researcher_tools(researcher.name)
        
        # Register tool agent for this researcher
        await ToolAgent.register(
            runtime,
            f"BiblioCollaborationToolAgent_{researcher_safe_name}",
            lambda tools=researcher_tools: ToolAgent(description="Tool agent for bibliographic scientific collaboration.", tools=tools),
        )
        
        # Register researcher agent
        await BiblioResearcherAgent.register(
            runtime,
            f"BiblioResearcher_{researcher_safe_name}",
            lambda r=researcher, safe_name=researcher_safe_name, tools=researcher_tools: BiblioResearcherAgent(
                profile=r,
                model_client=model_client,
                model_context=BufferedChatCompletionContext(buffer_size=20),
                tool_schema=[tool.schema for tool in tools],
                tool_agent_type=f"BiblioCollaborationToolAgent_{safe_name}",
            ),
        )


async def run_biblio_collaboration_round(
    runtime: AgentRuntime, 
    round_num: int, 
    phase: str,
    researchers: List[str]
) -> None:
    """Run a single round of bibliographic collaboration discussion."""
    
    print(f"\n{'='*60}")
    print(f"🔄 BIBLIOGRAPHIC COLLABORATION ROUND {round_num}: {phase.upper()}")
    print(f"{'='*60}")
    
    biblio_collaboration_state.discussion_round = round_num
    
    # Different prompts for different phases with bibliographic context
    if phase == "introduction":
        prompt = ("Please introduce yourself, highlighting your publication patterns, network position, "
                 "and research community. Share insights from your bibliographic profile and mention "
                 "potential collaboration opportunities based on your past publication and network data.")
    elif phase == "community_analysis":
        prompt = ("Analyze the research communities represented here and identify cross-community "
                 "collaboration opportunities. Consider the publication patterns, journal preferences, "
                 "and network positions of participants from different communities.")
    elif phase == "proposal":
        prompt = ("Based on the community analysis and bibliographic insights, propose 1-2 specific "
                 "research topics that would benefit from your expertise and publication track record. "
                 "Consider target journals, collaboration potential, and interdisciplinary factors.")
    elif phase == "evaluation":
        prompt = ("Review the proposed topics through the lens of your bibliographic profile. "
                 "Consider feasibility based on publication patterns, journal fit, collaboration "
                 "potential with other participants, and alignment with your research trajectory. "
                 "Vote for the most promising topics.")
    elif phase == "consensus":
        prompt = ("Review the voting results and discuss the selected topics. Consider the "
                 "bibliographic evidence for successful collaboration, publication strategies, "
                 "and next steps for implementation based on community strengths.")
    else:
        prompt = f"Continue the bibliographic collaboration discussion for phase: {phase}"
    
    # Send messages to researchers, considering community representation
    communities = list(biblio_collaboration_state.communities.keys())
    researchers_by_community = {}
    
    for researcher in researchers:
        if researcher in biblio_collaboration_state.researchers:
            community = biblio_collaboration_state.researchers[researcher].research_community
            if community not in researchers_by_community:
                researchers_by_community[community] = []
            researchers_by_community[community].append(researcher)
    
    # Rotate through communities to ensure balanced participation
    all_researchers = []
    max_community_size = max(len(members) for members in researchers_by_community.values()) if researchers_by_community else 0
    
    for i in range(max_community_size):
        for community in communities:
            if community in researchers_by_community and i < len(researchers_by_community[community]):
                all_researchers.append(researchers_by_community[community][i])
    
    for researcher in all_researchers:
        message = BiblioCollaborationMessage(
            content=prompt,
            sender="Moderator",
            message_type=phase,
            community_context=biblio_collaboration_state.researchers[researcher].research_community
        )
        
        researcher_id = AgentId(f"BiblioResearcher_{researcher.replace(' ', '_').replace('.', '')}", "default")
        await runtime.send_message(message, researcher_id)
        
        # Add delay between messages to avoid overwhelming
        await asyncio.sleep(1)


async def main_biblio(config_file: str = ".server_deployed_LLMs", config_section: str = "ali_official", num_rounds: int = 4) -> None:
    """Main entry point for the bibliographic scientific collaboration simulation."""
    
    print("🧬 BIBLIOGRAPHIC SCIENTIFIC COLLABORATION SIMULATION")
    print("=" * 60)
    print("Simulating academic research collaboration based on:")
    print("• Bibliographic data and publication patterns")
    print("• Community detection from coauthorship networks") 
    print("• Author roles and network positions")
    print("• Journal preferences and collaboration history")
    print()
    
    # Initialize runtime and model
    runtime = SingleThreadedAgentRuntime()
    model_client = create_model_client_from_config(config_file, config_section)
    
    # Set up the collaboration
    await setup_biblio_collaboration(runtime, model_client)
    
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
        phases = ["introduction", "community_analysis", "proposal", "evaluation", "consensus"]
        
        for i, phase in enumerate(phases, 1):
            await run_biblio_collaboration_round(runtime, i, phase, researcher_names)
            
            # Wait for discussions to settle
            await asyncio.sleep(5)
            
            # Show status after key phases
            if phase in ["community_analysis", "proposal", "evaluation"]:
                print(f"\n📊 STATUS AFTER {phase.upper()} PHASE:")
                if phase == "community_analysis":
                    print(get_community_analysis())
                else:
                    print(get_biblio_collaboration_status())
                print()
        
        # Final summary
        print(f"\n🎯 FINAL BIBLIOGRAPHIC COLLABORATION RESULTS")
        print("=" * 60)
        print(get_current_biblio_topics())
        print(get_biblio_collaboration_status())
        print(get_community_analysis())
        
        top_topics = biblio_collaboration_state.get_top_topics(3)
        if top_topics:
            print("\n🏆 SELECTED RESEARCH DIRECTIONS:")
            for i, topic in enumerate(top_topics, 1):
                print(f"{i}. {topic.title} ({topic.votes} votes)")
                print(f"   Proposed by: {topic.proposer}")
                print(f"   Target journals: {', '.join(topic.target_journals)}")
                print(f"   Supporters: {', '.join(topic.supporters)}")
                print(f"   Potential collaborators: {', '.join(topic.potential_collaborators)}")
                print()
                
    except KeyboardInterrupt:
        print("\n⚠️ Collaboration interrupted by user")
    finally:
        # Clean shutdown
        await runtime.stop_when_idle()
        await model_client.close()
        print("\n✅ Bibliographic collaboration simulation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a bibliographic scientific collaboration simulation.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--config-file", type=str, help="Path to the model configuration file.", default=".server_deployed_LLMs"
    )
    parser.add_argument(
        "--config-section", type=str, help="Configuration section to use.", default="ali_official"
    )
    parser.add_argument(
        "--num-rounds", type=int, help="Number of discussion rounds.", default=5
    )
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("autogen_core").setLevel(logging.DEBUG)
        handler = logging.FileHandler("bibliographic_collaboration.log")
        logging.getLogger("autogen_core").addHandler(handler)

    try:
        asyncio.run(main_biblio(args.config_file, args.config_section, args.num_rounds))
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Please ensure the configuration file '{args.config_file}' exists and contains the section '{args.config_section}'.")
        print("\n💡 Tip: To see how this bibliographic collaboration works without an API key, run:")
        print("    python biblio_demo.py")
        sys.exit(1)
    except Exception as e:
        if "api_key" in str(e).lower() or "authentication" in str(e).lower() or "connection" in str(e).lower():
            print(f"❌ Error: {e}")
            print("\n💡 This appears to be an API key or connection issue.")
            print("To test the bibliographic collaboration system without an API key, run:")
            print("    python biblio_demo.py")
            sys.exit(1)
        else:
            print(f"❌ Error: {e}")
            raise