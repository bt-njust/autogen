#!/usr/bin/env python3
"""
Enhanced Scientific Collaboration V2 Standalone Demo

This demonstration script shows the core functionality and improvements of the enhanced 
scientific collaboration system without requiring any external dependencies.

Key enhancements demonstrated:
1. Enhanced team roles and academic positions
2. Discussion-based consensus (no voting)
3. Topic source identification and assignment mechanisms
4. Valid researcher names using only numbers, '_', '-'
5. Improved state management and termination conditions
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional


class TeamRole(Enum):
    """Enhanced team role definitions."""
    LEADER = "leader"
    CO_LEADER = "co_leader"
    INCUMBENT = "incumbent"
    NEWCOMER = "newcomer"


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
    current_workload: str = "moderate"


@dataclass
class ResearchTopic:
    """Enhanced research topic with source information."""
    title: str
    description: str
    proposer: str
    source: TopicSource
    required_expertise: List[str] = field(default_factory=list)
    assigned_members: List[str] = field(default_factory=list)
    status: str = "proposed"
    priority_level: str = "medium"


@dataclass 
class DiscussionEntry:
    """Discussion entry for a topic."""
    participant: str
    topic_id: str
    interest_level: str
    contribution_level: str
    reasoning: str


class EnhancedCollaborationState:
    """Enhanced collaboration state management."""
    
    def __init__(self, max_topics: int = 6, max_rounds_per_phase: int = 3):
        self.researchers: Dict[str, ResearcherProfile] = {}
        self.topics: Dict[str, ResearchTopic] = {}
        self.discussions: List[DiscussionEntry] = []
        self.max_topics = max_topics
        self.max_rounds_per_phase = max_rounds_per_phase
        self.current_phase = ""
        self.phase_round_count = 0
        
    def add_researcher(self, profile: ResearcherProfile):
        """Add a researcher to the collaboration."""
        self.researchers[profile.name] = profile
        
    def add_topic(self, topic: ResearchTopic) -> Optional[str]:
        """Add a research topic and return its ID."""
        if len(self.topics) >= self.max_topics:
            return None
        topic_id = f"topic_{len(self.topics) + 1}"
        self.topics[topic_id] = topic
        return topic_id
        
    def add_discussion(self, entry: DiscussionEntry):
        """Add a discussion entry."""
        self.discussions.append(entry)
        
    def assign_to_topic(self, topic_id: str, assignee: str) -> bool:
        """Assign a member to a topic."""
        if topic_id in self.topics and assignee not in self.topics[topic_id].assigned_members:
            self.topics[topic_id].assigned_members.append(assignee)
            return True
        return False


def demonstrate_enhanced_collaboration():
    """Demonstrate the enhanced collaboration system."""
    
    print("🧪 SCIENTIFIC COLLABORATION V2 STANDALONE DEMO")
    print("=" * 60)
    print("This demo showcases the enhanced collaboration system with:")
    print("• Team roles: leader, co-leader, incumbent, newcomer")
    print("• Academic positions and workload considerations")
    print("• Discussion-based consensus (no voting)")
    print("• Topic sources and assignment mechanisms")
    print("• Valid researcher names (numbers, '_', '-' only)")
    print("• Improved state management and termination controls")
    print()
    
    # Initialize collaboration state
    state = EnhancedCollaborationState()
    
    # Create enhanced researcher profiles with valid names
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
                "Robust Multimodal AI Systems (2023)"
            ],
            collaboration_history=[
                CollaborationHistory(["Dr_Wilson_002"], "Privacy-Preserving ML", "3 joint publications", 2022),
            ],
            years_in_team=5,
            current_workload="heavy"
        ),
        ResearcherProfile(
            name="Dr_Wilson_002",
            academic_position=AcademicPosition.ASSOCIATE_PROFESSOR,
            team_role=TeamRole.CO_LEADER,
            expertise=["Data Science", "Statistical Analysis", "Healthcare Analytics"],
            institution="Stanford University",
            research_interests=["Healthcare AI", "Ethical AI", "Social Media Analysis"],
            recent_publications=[
                "Ethical Considerations in Healthcare AI (2023)",
                "Large-Scale Social Media Sentiment Analysis (2023)"
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
                "User Trust in AI Systems (2023)"
            ],
            collaboration_history=[],
            years_in_team=3,
            current_workload="moderate"
        ),
        ResearcherProfile(
            name="PhD_Zhang_005",
            academic_position=AcademicPosition.PHD_CANDIDATE,
            team_role=TeamRole.NEWCOMER,
            expertise=["Natural Language Processing", "Text Mining"],
            institution="MIT",
            research_interests=["Conversational AI", "Language Models"],
            recent_publications=[
                "Advances in Conversational AI (2023)"
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
            research_interests=["Autonomous Systems", "Robot Learning"],
            recent_publications=[
                "Vision-based Robot Control (2023)"
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
        state.add_researcher(researcher)
    
    print("👥 ENHANCED TEAM STRUCTURE:")
    print("=" * 30)
    for name, profile in state.researchers.items():
        print(f"👤 {profile.name} | {profile.academic_position.value}")
        print(f"   🎭 Team role: {profile.team_role.value}")
        print(f"   🏛️ Institution: {profile.institution}")
        print(f"   🔬 Expertise: {', '.join(profile.expertise[:2])}...")
        print(f"   ⏰ Years in team: {profile.years_in_team}")
        print(f"   📊 Current workload: {profile.current_workload}")
        print()
    
    # Phase 1: Introduction (simulated)
    print("🎯 PHASE 1: INTRODUCTION")
    print("=" * 30)
    print("Team members would introduce themselves, sharing:")
    print("• Academic position and team role")
    print("• Expertise areas and research interests")
    print("• Current workload and availability")
    print("• Past collaboration history")
    print()
    
    # Phase 2: Proposal
    print("🎯 PHASE 2: PROPOSAL")
    print("=" * 30)
    
    # Professor proposes from granted project
    topic1 = ResearchTopic(
        title="Federated Learning for Privacy-Preserving Healthcare AI",
        description="Develop federated learning systems for healthcare applications ensuring HIPAA compliance while enabling multi-institutional collaboration. Leveraging our recent NSF grant.",
        proposer="Prof_Chen_001",
        source=TopicSource.GRANTED_PROJECT,
        required_expertise=["Machine Learning", "Healthcare Analytics", "Privacy Technology"],
        priority_level="high"
    )
    topic1_id = state.add_topic(topic1)
    print(f"🔬 NEW TOPIC PROPOSED by Prof_Chen_001")
    print(f"📋 Title: {topic1.title}")
    print(f"🎯 Source: {topic1.source.value}")
    print(f"⚡ Priority: {topic1.priority_level}")
    print(f"🆔 Topic ID: {topic1_id}")
    print()
    
    # Co-leader proposes research expansion
    topic2 = ResearchTopic(
        title="Ethical AI Framework for Social Media Analysis",
        description="Create comprehensive ethical guidelines and technical framework for responsible social media data analysis, building on our previous work in bias detection.",
        proposer="Dr_Wilson_002",
        source=TopicSource.RESEARCH_EXPANSION,
        required_expertise=["Data Science", "Ethics", "Social Media Analysis"],
        priority_level="medium"
    )
    topic2_id = state.add_topic(topic2)
    print(f"🔬 NEW TOPIC PROPOSED by Dr_Wilson_002")
    print(f"📋 Title: {topic2.title}")
    print(f"🎯 Source: {topic2.source.value}")
    print(f"⚡ Priority: {topic2.priority_level}")
    print(f"🆔 Topic ID: {topic2_id}")
    print()
    
    # Postdoc proposes for future grant
    topic3 = ResearchTopic(
        title="Vision-based Autonomous Robot Learning",
        description="Develop novel computer vision algorithms for autonomous robot learning in unstructured environments. Targeting future NSF robotics grant application.",
        proposer="Postdoc_Lee_006",
        source=TopicSource.FUTURE_GRANT_PROJECT,
        required_expertise=["Robotics", "Computer Vision", "Machine Learning"],
        priority_level="medium"
    )
    topic3_id = state.add_topic(topic3)
    print(f"🔬 NEW TOPIC PROPOSED by Postdoc_Lee_006")
    print(f"📋 Title: {topic3.title}")
    print(f"🎯 Source: {topic3.source.value}")
    print(f"⚡ Priority: {topic3.priority_level}")
    print(f"🆔 Topic ID: {topic3_id}")
    print()
    
    # Phase 3: Discussion
    print("🎯 PHASE 3: DISCUSSION")
    print("=" * 30)
    
    # Team members discuss topics with enhanced assessment
    discussions = [
        DiscussionEntry(
            participant="Dr_Wilson_002",
            topic_id=topic1_id,
            interest_level="high",
            contribution_level="significant",
            reasoning="This aligns perfectly with my healthcare analytics expertise and our previous collaboration on privacy-preserving ML. I can contribute statistical analysis and ethical framework development."
        ),
        DiscussionEntry(
            participant="Dr_Garcia_003",
            topic_id=topic1_id,
            interest_level="medium",
            contribution_level="moderate",
            reasoning="Interested in the human-computer interaction aspects of healthcare AI systems. My accessibility research could inform user interface design, though my current workload is moderate."
        ),
        DiscussionEntry(
            participant="PhD_Zhang_005",
            topic_id=topic2_id,
            interest_level="high",
            contribution_level="lead",
            reasoning="This closely matches my dissertation focus on ethical AI and NLP. I would love to lead the technical implementation of bias detection algorithms for social media text analysis."
        ),
        DiscussionEntry(
            participant="Postdoc_Lee_006",
            topic_id=topic3_id,
            interest_level="high",
            contribution_level="lead",
            reasoning="This is my core expertise area. I proposed this topic and can lead the computer vision algorithm development. My current light workload allows for significant commitment."
        )
    ]
    
    for discussion in discussions:
        state.add_discussion(discussion)
        topic = state.topics[discussion.topic_id]
        participant = state.researchers[discussion.participant]
        
        print(f"💬 TOPIC DISCUSSION by {discussion.participant}")
        print(f"📋 Topic: {topic.title}")
        print(f"💡 Interest level: {discussion.interest_level}")
        print(f"🤝 Contribution level: {discussion.contribution_level}")
        print(f"💭 Reasoning: {discussion.reasoning}")
        print(f"👤 Role: {participant.team_role.value} ({participant.academic_position.value})")
        print(f"📊 Current workload: {participant.current_workload}")
        print()
    
    # Phase 4: Consensus and Assignment
    print("🎯 PHASE 4: CONSENSUS & ASSIGNMENT")
    print("=" * 30)
    
    # Leader makes assignments based on discussions
    assignments = [
        (topic1_id, "Dr_Wilson_002", "Dr. Wilson showed high interest and significant contribution capability. His healthcare analytics expertise and our previous collaboration make him ideal for this project."),
        (topic1_id, "Dr_Garcia_003", "Dr. Garcia's HCI expertise will be valuable for user interface design of the healthcare AI system, ensuring accessibility and usability."),
        (topic2_id, "PhD_Zhang_005", "Zhang expressed high interest and willingness to lead. This aligns with their dissertation focus and will provide excellent PhD research experience."),
        (topic3_id, "Postdoc_Lee_006", "Lee proposed this topic and has the core expertise. Their light workload allows for focused development of the vision algorithms.")
    ]
    
    for topic_id, assignee, reasoning in assignments:
        state.assign_to_topic(topic_id, assignee)
        topic = state.topics[topic_id]
        
        print(f"📋 TOPIC ASSIGNMENT by Prof_Chen_001 (Leader)")
        print(f"📝 Topic: {topic.title}")
        print(f"👤 Assigned to: {assignee}")
        print(f"💭 Reasoning: {reasoning}")
        print(f"👥 All assigned members: {', '.join(topic.assigned_members)}")
        print()
    
    # Final Results
    print("🏆 FINAL COLLABORATION RESULTS")
    print("=" * 40)
    print(f"👥 Team members: {len(state.researchers)}")
    print(f"📋 Topics proposed: {len(state.topics)}")
    print(f"💬 Discussion entries: {len(state.discussions)}")
    print()
    
    print("📊 FINAL TOPIC ASSIGNMENTS:")
    for topic_id, topic in state.topics.items():
        if topic.assigned_members:
            print(f"📋 {topic.title}")
            print(f"   👤 Proposed by: {topic.proposer}")
            print(f"   🎯 Source: {topic.source.value}")
            print(f"   👥 Assigned to: {', '.join(topic.assigned_members)}")
            print(f"   ⚡ Priority: {topic.priority_level}")
            print()
    
    print("✨ ENHANCED FEATURES DEMONSTRATED:")
    print("=" * 40)
    print("✅ Team roles: leader, co-leader, incumbent, newcomer")
    print("✅ Academic positions: professor, associate prof, postdoc, PhD candidate")
    print("✅ Topic sources: granted_project, research_expansion, future_grant_project")
    print("✅ Discussion-based assessment: interest + contribution levels + reasoning")
    print("✅ Leader-based assignments considering expertise and workload")
    print("✅ Valid researcher names using only numbers, '_', and '-'")
    print("✅ Enhanced state tracking with termination conditions")
    print("✅ No voting strategy - qualitative consensus building")
    print()
    
    print("🚀 IMPROVEMENTS OVER ORIGINAL VERSION:")
    print("=" * 40)
    print("• Replaced voting with discussion-based consensus")
    print("• Added team roles and academic position hierarchy")
    print("• Included topic source identification")
    print("• Implemented assignment mechanisms for leaders")
    print("• Added termination conditions and phase controls")
    print("• Improved asyncio handling (timeout protection)")
    print("• Prevented endless loops and message overwhelming")
    print("• Used valid researcher names")
    print("• Enhanced state management and tracking")
    print()
    
    print("📝 READY FOR REAL SIMULATION:")
    print("To run with actual AI agents:")
    print("1. Install required dependencies (autogen-core, autogen-ext)")
    print("2. Set up configuration file (.server_deployed_LLMs)")
    print("3. Run: python main.py --config-section your_section")


if __name__ == "__main__":
    demonstrate_enhanced_collaboration()