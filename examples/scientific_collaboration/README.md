# Scientific Collaboration Example

This example demonstrates how AutoGen can simulate realistic academic research scenarios where multiple researchers discuss and select joint research topics based on their profiles and collaboration history.

## Features

- Multiple researcher agents with distinct expertise profiles
- Collaboration history tracking
- Interactive topic proposal and discussion
- Consensus-building mechanisms
- Realistic academic conversation patterns

## Quick Start - Demo Mode

To see how the collaboration system works without requiring an API key:

```bash
python demo.py
```

This will run a demonstration showing:
- Researcher profile management
- Topic proposal mechanics
- Voting and consensus building
- Real-time collaboration status tracking

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

Run the full scientific collaboration simulation with AI agents:

```bash
python main.py --config-file .server_deployed_LLMs --config-section ali_official
```

Optional arguments:
- `--verbose`: Enable detailed logging
- `--num-rounds`: Number of discussion rounds (default: 4)
- `--config-file`: Path to configuration file (default: .server_deployed_LLMs)
- `--config-section`: Configuration section to use (default: ali_official)

## Example Output

The simulation will show researchers:
1. Introducing themselves and their expertise
2. Proposing research topics
3. Discussing the merits of each proposal
4. Reaching consensus on joint research directions

## Researcher Profiles

The example includes researchers with expertise in:
- Machine Learning & AI
- Data Science & Analytics  
- Human-Computer Interaction
- Computational Biology
- Cybersecurity

Each researcher has:
- Detailed expertise areas
- Research interests and recent publications
- Collaboration history with other researchers
- Institutional affiliations