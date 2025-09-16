# Bibliographic Data Scientific Collaboration

This example demonstrates how AutoGen can simulate realistic academic research scenarios for structural biology research teams identified through community detection algorithms, based on bibliographic data and coauthorship networks from 2010-2023.

## Features

### Enhanced Researcher Profiles
- **Publication-based author roles**: First author focused, corresponding author focused, collaborative author, etc.
- **Network-based roles**: Hub connector, community leader, bridge builder, specialist, newcomer
- **Bibliographic metrics**: Publication patterns, citation analysis, journal preferences
- **Collaboration history**: Past partnerships, institutional affiliations, cross-community connections
- **Template placeholders**: For integration with computed bibliometric data

### Community Detection Integration
- Research community identification from coauthorship networks
- Cross-community collaboration opportunities
- Community-aware topic proposal and voting
- Network position analysis (centrality measures, clustering coefficients)

### Template-Based Prompts
The framework includes placeholder templates for dynamic integration of computed metrics:
- `{COLLABORATION_FREQUENCY}` - Collaboration rate metrics
- `{NETWORK_POSITION_SCORE}` - Centrality-based influence scores
- `{FIRST_AUTHOR_RATIO}` - Publication pattern statistics
- `{SPECIALIZATION_SCORE}` - Research focus measurement
- `{INFLUENCE_INDEX}` - Impact and citation metrics

## Quick Start - Demo Mode

To see how the bibliographic collaboration system works without requiring an API key:

```bash
python biblio_demo.py
```

This demonstration shows:
- Enhanced researcher profiles with publication patterns and network roles
- Community-based collaboration analysis
- Cross-disciplinary topic proposals based on bibliographic insights
- Voting with collaboration potential assessment
- Network-aware consensus building

## Full Setup with AI Agents

### Prerequisites

Install required dependencies:
```bash
pip install "autogen-ext[openai,azure]" "pyyaml"
```

### Configuration

1. Create the configuration file:
   ```bash
   cp .server_deployed_LLMs.template .server_deployed_LLMs
   ```

2. Edit `.server_deployed_LLMs` and replace `YOUR_API_KEY_HERE` with your actual API key.

### Usage

Run the full bibliographic collaboration simulation with AI agents:

```bash
python bibliographic_collaboration.py --config-file .server_deployed_LLMs --config-section ali_official
```

Optional arguments:
- `--verbose`: Enable detailed logging
- `--num-rounds`: Number of discussion rounds (default: 5)
- `--config-file`: Path to configuration file
- `--config-section`: Configuration section to use

## Simulation Phases

The bibliographic collaboration simulation includes these phases:

1. **Introduction**: Researchers present their bibliographic profiles, publication patterns, and network positions
2. **Community Analysis**: Analysis of research communities and identification of cross-community opportunities
3. **Proposal**: Topic proposals based on bibliographic insights and collaboration potential
4. **Evaluation**: Assessment of proposals considering publication feasibility and network effects
5. **Consensus**: Final selection based on bibliographic evidence and collaboration strength

## Researcher Profile Structure

Each researcher has an enhanced profile including:

### Publication Patterns
```python
PublicationPatterns(
    total_publications=67,
    first_author_count=23,
    corresponding_author_count=12,
    first_author_ratio="{FIRST_AUTHOR_RATIO}",  # Template placeholder
    average_coauthors="{AVERAGE_COAUTHORS}",
    research_productivity_score="{RESEARCH_PRODUCTIVITY_SCORE}"
)
```

### Network Metrics
```python
NetworkMetrics(
    degree_centrality=0.75,
    betweenness_centrality=0.45,
    clustering_coefficient=0.72,
    community_id="structural_biology_ai",
    collaboration_frequency="{COLLABORATION_FREQUENCY}",  # Template placeholder
    network_position_score="{NETWORK_POSITION_SCORE}"
)
```

### Role Definitions

**Author Roles** (based on publication patterns):
- `FIRST_AUTHOR_FOCUSED`: Often leads research initiatives
- `CORRESPONDING_AUTHOR_FOCUSED`: Senior researchers with extensive networks
- `COLLABORATIVE_AUTHOR`: Balanced collaboration approach
- `MIDDLE_AUTHOR`: Contributes specialized expertise
- `SENIOR_AUTHOR`: Provides oversight and resources

**Network Roles** (based on coauthorship analysis):
- `HUB_CONNECTOR`: High betweenness centrality, connects communities
- `COMMUNITY_LEADER`: High degree centrality within community
- `BRIDGE_BUILDER`: Facilitates cross-community collaborations
- `SPECIALIST`: High clustering coefficient, focused expertise
- `NEWCOMER`: Developing network connections

## Integration with Your Bibliographic Data

To adapt this framework for your structural biology community detection data:

### 1. Data Preparation
```python
# Replace sample data with your computed metrics
researcher_profile.publication_patterns.first_author_ratio = your_computed_ratio
researcher_profile.network_metrics.collaboration_frequency = your_frequency_metric
researcher_profile.specialization_score = your_specialization_computation
```

### 2. Community Assignment
```python
# Assign researchers to detected communities
researcher_profile.research_community = your_community_id
researcher_profile.network_metrics.community_id = your_community_id
```

### 3. Role Assignment
```python
# Determine roles based on your analysis
if first_author_papers / total_papers > 0.4:
    researcher_profile.author_role = AuthorRole.FIRST_AUTHOR_FOCUSED
elif corresponding_author_papers / total_papers > 0.3:
    researcher_profile.author_role = AuthorRole.CORRESPONDING_AUTHOR_FOCUSED

# Network role based on centrality measures
if betweenness_centrality > threshold:
    researcher_profile.network_role = NetworkRole.HUB_CONNECTOR
```

### 4. Template Replacement
Replace template placeholders with your computed values:
```python
def compute_and_replace_templates(profile, your_metrics):
    profile.network_metrics.collaboration_frequency = str(your_metrics['collab_freq'])
    profile.publication_patterns.first_author_ratio = f"{your_metrics['fa_ratio']:.2f}"
    profile.specialization_score = str(your_metrics['specialization'])
    return profile
```

## Example Output

The simulation produces detailed bibliographic insights:

```
🧬 BIBLIOGRAPHIC SCIENTIFIC COLLABORATION SIMULATION
============================================================

👥 Dr. Sarah Chen (MIT Department of Biology)
   🔬 Expertise: Structural Biology, Protein Folding, Computational Biology
   📄 Publications: 67 (23 first-author, 12 corresponding-author)
   📝 Author role: first_author_focused
   🌐 Network role: hub_connector
   🏛️ Community: Computational Structural Biology

📋 TOPIC PROPOSED: AI-Driven Protein Structure Prediction for Drug Discovery
   Target journals: Nature Machine Intelligence, Nature Structural & Molecular Biology
   Interdisciplinary factors: Cross-community collaboration between AI/ML and structural biology
   Potential collaborators: Dr. James Wilson, Dr. Alex Kim

🗳️ VOTE CAST by Dr. James Wilson [Community: Drug Discovery and Development]
   Reasoning: Dr. Chen's first-author publication pattern in AI perfectly complements 
   my corresponding-author experience in drug discovery
   Collaboration potential: 85% journal overlap indicates feasible publication strategy
```

## Customization Guidelines

### For Your Structural Biology Data:

1. **Publication Analysis**: Extract author position patterns, journal preferences, citation metrics
2. **Network Analysis**: Compute centrality measures, community assignments, collaboration frequencies  
3. **Role Identification**: Define roles based on your specific field's collaboration patterns
4. **Template Integration**: Replace placeholders with your computed bibliometric values
5. **Community Detection**: Use your algorithm results to assign community memberships

### Template Placeholders Available:

- `{COLLABORATION_FREQUENCY}` - Rate of new collaborations per year
- `{NETWORK_POSITION_SCORE}` - Composite centrality score  
- `{CROSS_COMMUNITY_CONNECTIONS}` - Number of external community ties
- `{FIRST_AUTHOR_RATIO}` - Fraction of first-author publications
- `{CORRESPONDING_AUTHOR_RATIO}` - Fraction of corresponding-author publications
- `{AVERAGE_COAUTHORS}` - Mean number of coauthors per paper
- `{RESEARCH_PRODUCTIVITY_SCORE}` - Publications per year metric
- `{SPECIALIZATION_SCORE}` - Topic focus measurement
- `{INFLUENCE_INDEX}` - Citation-based impact score
- `{COLLABORATION_DIVERSITY}` - Cross-institutional collaboration rate

This framework provides a foundation for realistic academic collaboration simulation based on actual bibliographic patterns and community structures in structural biology research.