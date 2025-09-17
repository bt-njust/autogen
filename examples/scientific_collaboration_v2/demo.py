#!/usr/bin/env python3
"""
Enhanced Scientific Collaboration V2 Demo

This demonstration script shows the structure and capabilities of the enhanced scientific 
collaboration simulation without requiring an OpenAI API key. It demonstrates:

1. Enhanced researcher profiles with team roles and academic positions
2. Topic proposal with source information
3. Discussion-based assessment instead of voting
4. Assignment mechanisms for leaders
5. Termination conditions and improved state tracking

Run this to understand how the enhanced simulation works before setting up with a real API key.
"""

import sys
import os

# Add the current directory to Python path to import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    ResearcherProfile, 
    CollaborationHistory, 
    ResearchTopic,
    CollaborationState,
    TeamRole,
    AcademicPosition,
    TopicSource,
    propose_topic,
    discuss_topic,
    assign_to_topic,
    get_current_topics,
    get_researcher_profiles,
    get_collaboration_status
)

def demonstrate_enhanced_collaboration():
    """Demonstrate the enhanced collaboration system without requiring API calls."""
    
    print("🧪 SCIENTIFIC COLLABORATION V2 DEMO")
    print("=" * 50)
    print("This demo showcases the enhanced collaboration system with:")
    print("• Team roles: leader, co-leader, incumbent, newcomer")
    print("• Academic positions and workload considerations")
    print("• Discussion-based consensus (no voting)")
    print("• Topic sources and assignment mechanisms")
    print("• Valid researcher names (numbers, '_', '-' only)")
    print()
    
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
    
    # Initialize global collaboration state
    import main
    main.collaboration_state = CollaborationState(max_topics=6, max_rounds_per_phase=3)
    
    # Add researchers to collaboration
    for researcher in researchers:
        main.collaboration_state.add_researcher(researcher)
    
    print("👥 ENHANCED TEAM STRUCTURE:")
    print("=" * 30)
    print(get_researcher_profiles())
    
    # Simulate topic proposal phase
    print("\n📋 TOPIC PROPOSAL PHASE:")
    print("=" * 30)
    
    # Professor proposes from granted project
    result1 = propose_topic(
        "Prof_Chen_001",
        "Federated Learning for Privacy-Preserving Healthcare AI",
        "Develop federated learning systems for healthcare applications ensuring HIPAA compliance while enabling multi-institutional collaboration. Leveraging our recent NSF grant.",
        "granted_project",
        "Machine Learning, Healthcare Analytics, Privacy Technology",
        "high"
    )
    print(f"📝 {result1}\n")
    
    # Co-leader proposes research expansion
    result2 = propose_topic(
        "Dr_Wilson_002", 
        "Ethical AI Framework for Social Media Analysis",
        "Create comprehensive ethical guidelines and technical framework for responsible social media data analysis, building on our previous work in bias detection.",
        "research_expansion",
        "Data Science, Ethics, Social Media Analysis",
        "medium"
    )
    print(f"📝 {result2}\n")
    
    # Postdoc proposes for future grant
    result3 = propose_topic(
        "Postdoc_Lee_006",
        "Vision-based Autonomous Robot Learning",
        "Develop novel computer vision algorithms for autonomous robot learning in unstructured environments. Targeting future NSF robotics grant application.",
        "future_grant_project",
        "Robotics, Computer Vision, Machine Learning",
        "medium"
    )
    print(f"📝 {result3}\n")
    
    print("📊 CURRENT TOPICS:")
    print(get_current_topics())
    
    # Simulate discussion phase
    print("\n💬 DISCUSSION PHASE:")
    print("=" * 30)
    
    # Team members discuss topics
    discussion1 = discuss_topic(
        "Dr_Wilson_002",
        "topic_1",
        "high",
        "significant",
        "This aligns perfectly with my healthcare analytics expertise and our previous collaboration on privacy-preserving ML. I can contribute statistical analysis and ethical framework development."
    )
    print(f"💭 {discussion1}\n")
    
    discussion2 = discuss_topic(
        "Dr_Garcia_003",
        "topic_1",
        "medium",
        "moderate",
        "Interested in the human-computer interaction aspects of healthcare AI systems. My accessibility research could inform user interface design, though my current workload is moderate."
    )
    print(f"💭 {discussion2}\n")
    
    discussion3 = discuss_topic(
        "PhD_Zhang_005",
        "topic_2",
        "high",
        "lead",
        "This closely matches my dissertation focus on ethical AI and NLP. I would love to lead the technical implementation of bias detection algorithms for social media text analysis."
    )
    print(f"💭 {discussion3}\n")
    
    discussion4 = discuss_topic(
        "Postdoc_Lee_006",
        "topic_3",
        "high",
        "lead",
        "This is my core expertise area. I proposed this topic and can lead the computer vision algorithm development. My current light workload allows for significant commitment."
    )
    print(f"💭 {discussion4}\n")
    
    # Simulate consensus and assignment phase
    print("\n🎯 CONSENSUS & ASSIGNMENT PHASE:")
    print("=" * 30)
    
    # Leader makes assignments based on discussions
    assignment1 = assign_to_topic(
        "Prof_Chen_001",
        "topic_1",
        "Dr_Wilson_002",
        "Dr. Wilson showed high interest and significant contribution capability. His healthcare analytics expertise and our previous collaboration make him ideal for this project."
    )
    print(f"📋 {assignment1}\n")
    
    assignment2 = assign_to_topic(
        "Prof_Chen_001",
        "topic_1",
        "Dr_Garcia_003",
        "Dr. Garcia's HCI expertise will be valuable for user interface design of the healthcare AI system, ensuring accessibility and usability."
    )
    print(f"📋 {assignment2}\n")
    
    # Co-leader makes assignments for their proposed topic
    assignment3 = assign_to_topic(
        "Dr_Wilson_002",
        "topic_2",
        "PhD_Zhang_005",
        "Zhang expressed high interest and willingness to lead. This aligns with their dissertation focus and will provide excellent PhD research experience."
    )
    print(f"📋 {assignment3}\n")
    
    # Postdoc gets self-assigned to their topic (via leader)
    assignment4 = assign_to_topic(
        "Prof_Chen_001",
        "topic_3",
        "Postdoc_Lee_006",
        "Lee proposed this topic and has the core expertise. Their light workload allows for focused development of the vision algorithms."
    )
    print(f"📋 {assignment4}\n")
    
    # Final status
    print("📊 FINAL COLLABORATION STATUS:")
    print("=" * 30)
    print(get_collaboration_status())
    
    print("\n🏆 FINAL RESULTS:")
    print("=" * 30)
    print(get_current_topics())
    
    # Show enhanced features
    print("\n✨ ENHANCED FEATURES DEMONSTRATED:")
    print("=" * 30)
    print("• Team roles: leader, co-leader, incumbent, newcomer")
    print("• Academic positions: professor, associate prof, postdoc, PhD candidate")
    print("• Topic sources: granted_project, research_expansion, future_grant_project")
    print("• Discussion-based assessment: interest + contribution levels + reasoning")
    print("• Leader-based assignments considering expertise and workload")
    print("• Valid researcher names using only numbers, '_', and '-'")
    print("• Improved state tracking with phase rounds and termination conditions")
    
    print("\n🚀 READY FOR REAL SIMULATION:")
    print("To run with actual AI agents, ensure you have:")
    print("1. Configuration file (.server_deployed_LLMs)")
    print("2. API access to a compatible language model")
    print("3. Run: python main.py --config-section your_section")


if __name__ == "__main__":
    demonstrate_enhanced_collaboration()