#!/usr/bin/env python3
"""
Demonstration script for the Bibliographic Scientific Collaboration Example

This script shows the structure and capabilities of the bibliographic collaboration
simulation without requiring an OpenAI API key. It demonstrates:

1. Enhanced researcher profiles with bibliographic data
2. Publication pattern analysis and author roles  
3. Network-based role definitions from coauthorship analysis
4. Community detection integration
5. Template-based prompts with placeholder metrics
6. Cross-community collaboration opportunities

Run this to understand how the bibliographic simulation works before setting up with a real API key.
"""

import sys
import os

# Add the current directory to Python path to import bibliographic_collaboration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bibliographic_collaboration import (
    EnhancedResearcherProfile,
    BiblioCollaborationHistory, 
    ResearchTopic,
    BiblioCollaborationState,
    propose_biblio_topic,
    vote_for_biblio_topic,
    get_current_biblio_topics,
    get_researcher_biblio_profiles,
    get_community_analysis,
    get_biblio_collaboration_status,
    create_sample_researcher_profiles,
    AuthorRole,
    NetworkRole
)


def demonstrate_biblio_collaboration():
    """Demonstrate the bibliographic collaboration system without requiring API calls."""
    
    print("🧬 BIBLIOGRAPHIC SCIENTIFIC COLLABORATION DEMONSTRATION")
    print("=" * 70)
    print("This demonstrates bibliographic collaboration mechanics based on:")
    print("• Publication patterns and author roles")
    print("• Coauthorship network analysis")
    print("• Community detection insights")
    print("• Journal preferences and collaboration history")
    print()
    
    # Create a fresh collaboration state for this demo
    import bibliographic_collaboration
    bibliographic_collaboration.biblio_collaboration_state = BiblioCollaborationState()
    
    # Add sample researchers with enhanced bibliographic profiles
    researchers = create_sample_researcher_profiles()
    
    # Add researchers to collaboration
    for researcher in researchers:
        bibliographic_collaboration.biblio_collaboration_state.add_researcher(researcher)
    
    print("👥 PARTICIPATING RESEARCHERS WITH BIBLIOGRAPHIC PROFILES:")
    print("=" * 60)
    print(get_researcher_biblio_profiles())
    
    print("\n🏛️ RESEARCH COMMUNITY ANALYSIS:")
    print("=" * 40)
    print(get_community_analysis())
    
    # Simulate topic proposals based on bibliographic insights
    print("\n📋 BIBLIOGRAPHIC TOPIC PROPOSAL PHASE:")
    print("=" * 45)
    
    # Dr. Chen (Hub Connector, First Author Focused) proposes interdisciplinary topic
    result1 = propose_biblio_topic(
        "Dr. Sarah Chen",
        "AI-Driven Protein Structure Prediction for Drug Discovery",
        "Develop advanced deep learning models that integrate structural biology data with drug discovery pipelines, leveraging community detection insights to identify optimal collaboration networks among computational biologists and medicinal chemists.",
        "Machine Learning, Structural Biology, Drug Discovery, Computational Chemistry",
        "Nature Machine Intelligence, Nature Structural & Molecular Biology, Cell",
        "Cross-community collaboration between AI/ML researchers and structural biologists, with potential for high-impact publications in top-tier journals"
    )
    print(f"📝 {result1}\n")
    
    # Dr. Wilson (Community Leader, Corresponding Author Focused) proposes biochemistry-focused topic
    result2 = propose_biblio_topic(
        "Dr. James Wilson", 
        "Allosteric Drug Design Using Network-Based Target Identification",
        "Establish a systematic approach to identify allosteric drug targets using protein interaction networks and bibliometric analysis of successful drug discovery collaborations.",
        "Protein Biochemistry, Drug Discovery, Network Analysis, Allosteric Regulation",
        "Journal of Medicinal Chemistry, Nature Reviews Drug Discovery, Drug Discovery Today",
        "Leverages extensive collaboration networks in drug discovery community, builds on corresponding author expertise in target identification"
    )
    print(f"📝 {result2}\n")
    
    # Dr. Garcia (Bridge Builder, Collaborative Author) proposes methodological topic
    result3 = propose_biblio_topic(
        "Dr. Maria Garcia",
        "Cryo-EM Method Development for Membrane Protein-Drug Complexes",
        "Develop next-generation cryo-EM techniques specifically optimized for visualizing membrane protein-drug interactions, enabling structure-based drug design for challenging targets.",
        "Cryo-Electron Microscopy, Membrane Proteins, Method Development, Structural Biology",
        "Nature Methods, eLife, Structure, Journal of Structural Biology",
        "Bridges structural biology methods and drug discovery communities, leverages strong cross-institutional collaboration networks"
    )
    print(f"📝 {result3}\n")
    
    # Dr. Kim (Specialist, First Author Focused) proposes chemical biology topic
    result4 = propose_biblio_topic(
        "Dr. Alex Kim",
        "Chemical Biology Tools for Real-Time Protein Folding Analysis",
        "Design bioorthogonal chemical probes that enable real-time monitoring of protein folding dynamics in living cells, combining chemical biology expertise with biophysics insights.",
        "Chemical Biology, Protein Folding, Bioorthogonal Chemistry, Live Cell Imaging",
        "Nature Chemical Biology, ACS Chemical Biology, Chemical Science",
        "Combines chemical biology specialization with protein dynamics expertise, high potential for methodological innovation"
    )
    print(f"📝 {result4}\n")
    
    print("\n📊 STATUS AFTER PROPOSAL PHASE:")
    print("=" * 35)
    print(get_biblio_collaboration_status())
    print()
    
    # Simulate voting phase with bibliographic reasoning
    print("\n🗳️ BIBLIOGRAPHIC VOTING PHASE:")
    print("=" * 35)
    
    # Dr. Chen votes for Dr. Wilson's topic (cross-community collaboration)
    vote1 = vote_for_biblio_topic(
        "Dr. Sarah Chen", 
        "topic_2",
        "This aligns with my AI/ML expertise and Dr. Wilson's strong network in drug discovery. The bibliographic data shows successful cross-community collaborations between computational and experimental researchers lead to high-impact publications.",
        "High potential - combines my computational background with Dr. Wilson's extensive drug discovery network. Past publication patterns show 3.2x higher citation rates for such interdisciplinary collaborations."
    )
    print(f"🗳️ {vote1}\n")
    
    # Dr. Wilson votes for Dr. Chen's topic (recognizing AI potential)
    vote2 = vote_for_biblio_topic(
        "Dr. James Wilson",
        "topic_1", 
        "Dr. Chen's first-author publication pattern in AI and my corresponding-author experience in drug discovery create complementary strengths. The target journals align with both our publication histories.",
        "Excellent match - Dr. Chen's 67 publications with 23 first-author papers in AI/ML perfectly complement my 34 corresponding-author papers in drug discovery. Journal overlap indicates feasible publication strategy."
    )
    print(f"🗳️ {vote2}\n")
    
    # Dr. Garcia votes for her own topic but explains network reasoning
    vote3 = vote_for_biblio_topic(
        "Dr. Maria Garcia",
        "topic_3",
        "My bridge-builder network role and cross-institutional collaboration experience make this methodological development feasible. The bibliographic analysis shows method papers have strong cross-community impact.",
        "Strong foundation - my 0.52 betweenness centrality and collaboration with 3 different research communities provide the network foundation needed for widespread method adoption."
    )
    print(f"🗳️ {vote3}\n")
    
    # Dr. Kim votes for Dr. Garcia's topic (methodological synergy)
    vote4 = vote_for_biblio_topic(
        "Dr. Alex Kim",
        "topic_3",
        "Dr. Garcia's cryo-EM expertise complements my chemical biology tools. Our publication patterns show 85% overlap in target journals, indicating strong collaboration potential.",
        "Synergistic collaboration - our combined 52+45=97 publications with overlapping journal preferences in Nature Methods and ACS Chemical Biology suggest high success probability."
    )
    print(f"🗳️ {vote4}\n")
    
    # Dr. Zhang votes for Dr. Kim's topic (biophysics connection)
    vote5 = vote_for_biblio_topic(
        "Dr. Lisa Zhang",
        "topic_4",
        "Dr. Kim's chemical biology tools would revolutionize my single-molecule biophysics research. Our network analysis shows we're 2 degrees separated with 4 mutual collaborators.",
        "Perfect methodological fit - Dr. Kim's bioorthogonal chemistry expertise directly addresses the probe development needs in my protein folding dynamics research."
    )
    print(f"🗳️ {vote5}\n")
    
    # Cross-community votes showing network effects
    vote6 = vote_for_biblio_topic(
        "Dr. Lisa Zhang",
        "topic_1",
        "As a single-molecule biophysicist, I see huge potential in AI-driven structure prediction. The cross-community nature spans my biophysics background and Dr. Chen's computational expertise.",
        "Network analysis shows this collaboration would create new bridges between biophysics and AI communities, with potential for groundbreaking structural dynamics insights."
    )
    print(f"🗳️ {vote6}\n")
    
    vote7 = vote_for_biblio_topic(
        "Dr. Alex Kim",
        "topic_2",
        "Dr. Wilson's network-based target identification could benefit from chemical biology validation tools. Our publication histories show complementary methodological approaches.",
        "Methodological synergy - my chemical probe development experience (28 first-author papers) provides validation tools for Dr. Wilson's target identification approach."
    )
    print(f"🗳️ {vote7}\n")
    
    # Show final results with bibliographic context
    print("\n🏆 FINAL BIBLIOGRAPHIC COLLABORATION RESULTS:")
    print("=" * 50)
    print(get_current_biblio_topics())
    print(get_biblio_collaboration_status())
    
    # Analyze top topics with bibliographic insights
    top_topics = bibliographic_collaboration.biblio_collaboration_state.get_top_topics(4)
    print("\n🎯 SELECTED RESEARCH DIRECTIONS WITH BIBLIOGRAPHIC ANALYSIS:")
    print("=" * 60)
    for i, topic in enumerate(top_topics, 1):
        print(f"{i}. 📋 {topic.title}")
        print(f"   👤 Proposed by: {topic.proposer}")
        print(f"   🗳️ Votes: {topic.votes}")
        print(f"   👥 Supporters: {', '.join(topic.supporters)}")
        print(f"   📚 Target journals: {', '.join(topic.target_journals)}")
        print(f"   🤝 Potential collaborators: {', '.join(topic.potential_collaborators)}")
        print(f"   🌐 Interdisciplinary factors: {', '.join(topic.feasibility_factors)}")
        
        # Analyze supporter bibliographic profiles
        supporter_analysis = []
        for supporter in topic.supporters:
            if supporter in bibliographic_collaboration.biblio_collaboration_state.researchers:
                profile = bibliographic_collaboration.biblio_collaboration_state.researchers[supporter]
                supporter_analysis.append(f"{supporter} ({profile.author_role.value}, {profile.network_role.value})")
        print(f"   📊 Supporter profiles: {', '.join(supporter_analysis)}")
        print()
    
    # Community collaboration analysis
    print("\n🌐 CROSS-COMMUNITY COLLABORATION INSIGHTS:")
    print("=" * 45)
    
    # Analyze voting patterns by community
    community_votes = {}
    for topic_id, topic in bibliographic_collaboration.biblio_collaboration_state.topics.items():
        for supporter in topic.supporters:
            if supporter in bibliographic_collaboration.biblio_collaboration_state.researchers:
                community = bibliographic_collaboration.biblio_collaboration_state.researchers[supporter].research_community
                if community not in community_votes:
                    community_votes[community] = []
                community_votes[community].append(topic.title)
    
    for community, voted_topics in community_votes.items():
        print(f"🏛️ {community}:")
        print(f"   Voted for: {', '.join(set(voted_topics))}")
    
    print("\n📈 BIBLIOGRAPHIC SUCCESS INDICATORS:")
    print("=" * 40)
    print("✅ Cross-community collaboration achieved:")
    print("   • AI/ML + Drug Discovery (topics 1 & 2)")
    print("   • Methods Development + Multiple Communities (topic 3)")
    print("   • Chemical Biology + Biophysics (topic 4)")
    print("\n✅ Publication pattern complementarity:")
    print("   • First-author focused researchers leading novel topics")
    print("   • Corresponding-author researchers providing network support")
    print("   • Bridge-builders facilitating cross-community connections")
    print("\n✅ Journal alignment analysis:")
    print("   • 85% overlap in preferred journals among collaborators")
    print("   • Target journals match historical publication patterns")
    print("   • Impact factor alignment with researcher productivity scores")
    
    print("\n✅ DEMONSTRATION COMPLETED!")
    print("\nThis bibliographic collaboration framework shows:")
    print("• How publication patterns inform collaboration potential")
    print("• Network roles guide research topic selection and voting")
    print("• Community detection enables strategic cross-disciplinary partnerships")
    print("• Template placeholders allow integration of computed bibliometric data")
    print("• Journal preferences and collaboration history drive realistic outcomes")
    print("\nTo run with real AI agents and computed metrics, set up your configuration")
    print("and run: python bibliographic_collaboration.py")


if __name__ == "__main__":
    demonstrate_biblio_collaboration()