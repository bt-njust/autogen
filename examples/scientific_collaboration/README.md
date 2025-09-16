# Scientific Collaboration Example

This example demonstrates how AutoGen can simulate realistic academic research scenarios where multiple researchers discuss and select joint research topics based on their profiles and collaboration history.

## Available Scenarios

### 1. General Scientific Collaboration (`main.py`)
The original example with general academic researchers from various fields.

### 2. Bibliographic Data Collaboration (`bibliographic_collaboration.py`)
**NEW**: Enhanced scenario designed for structural biology research teams identified through community detection algorithms, based on bibliographic data and coauthorship networks from 2010-2023.

**Key Features**:
- Publication-based author roles (first author focused, corresponding author focused, etc.)
- Network-based roles from coauthorship analysis (hub connector, bridge builder, etc.)
- Community detection integration for team formation
- Template placeholders for computed bibliometric metrics
- Cross-community collaboration opportunities

## Features

- Multiple researcher agents with distinct expertise profiles
- Collaboration history tracking
- Interactive topic proposal and discussion
- Consensus-building mechanisms
- Realistic academic conversation patterns

## Quick Start - Demo Mode

### General Collaboration Demo
To see how the general collaboration system works without requiring an API key:

```bash
python demo.py
```

### Bibliographic Collaboration Demo
To see the enhanced bibliographic collaboration system:

```bash
python biblio_demo_standalone.py
```

This demonstrates:
- Enhanced researcher profiles with publication patterns and network roles
- Community-based collaboration analysis  
- Cross-disciplinary topic proposals based on bibliographic insights
- Voting with collaboration potential assessment
- Template placeholders for integration with computed metrics

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

2. Edit `.server_deployed_LLMs` and replace `YOUR_API_KEY_HERE` with your actual API key for the desired endpoint.

The configuration file format is:
```ini
[ali_official]
base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
api_key = YOUR_API_KEY_HERE

[openai_compatible]
base_url = https://api.openai.com/v1
api_key = YOUR_OPENAI_API_KEY_HERE
```

### Usage

#### Run General Scientific Collaboration
```bash
python main.py --config-file .server_deployed_LLMs --config-section ali_official
```

#### Run Bibliographic Scientific Collaboration
```bash
python bibliographic_collaboration.py --config-file .server_deployed_LLMs --config-section ali_official
```

Optional arguments for both:
- `--verbose`: Enable detailed logging
- `--num-rounds`: Number of discussion rounds
- `--config-file`: Path to configuration file (default: .server_deployed_LLMs)
- `--config-section`: Configuration section to use (default: ali_official)

## Bibliographic Collaboration Integration

For detailed information about integrating the bibliographic collaboration scenario with your community detection data from structural biology research, see [BIBLIOGRAPHIC_README.md](BIBLIOGRAPHIC_README.md).

**Template Placeholders Available**:
- `{COLLABORATION_FREQUENCY}` - Rate of new collaborations per year
- `{NETWORK_POSITION_SCORE}` - Centrality-based influence scores  
- `{FIRST_AUTHOR_RATIO}` - Publication pattern statistics
- `{SPECIALIZATION_SCORE}` - Research focus measurement
- `{INFLUENCE_INDEX}` - Impact and citation metrics

## Example Output

The simulation will show researchers:
1. Introducing themselves and their expertise
2. Proposing research topics
3. Discussing the merits of each proposal
4. Reaching consensus on joint research directions

### Bibliographic Example
```
👤 Dr. Sarah Chen (MIT Department of Biology)
   📄 Publications: 67 total (23 first-author, 12 corresponding-author)
   🌐 Network Role: hub_connector | Author Role: first_author_focused
   🏛️ Community: Computational Structural Biology
   📊 Metrics: Specialization 7.8/10, Influence 142.5

📋 TOPIC PROPOSED: AI-Driven Protein Structure Prediction for Drug Discovery
   Target journals: Nature Machine Intelligence, Nature Structural & Molecular Biology
   Potential collaborators: Dr. James Wilson, Dr. Alex Kim
```

## Researcher Profiles

### General Collaboration
The example includes researchers with expertise in:
- Machine Learning & AI
- Data Science & Analytics  
- Human-Computer Interaction
- Computational Biology
- Cybersecurity

### Bibliographic Collaboration
Enhanced profiles for structural biology including:
- Publication patterns and author roles
- Network centrality and community positions
- Collaboration history and institutional affiliations
- Journal preferences and impact metrics
- Template placeholders for computed bibliometric data

Each researcher has:
- Detailed expertise areas
- Research interests and recent publications
- Collaboration history with other researchers
- Institutional affiliations