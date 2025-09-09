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

1. Copy the model configuration template:
   ```bash
   cp model_config_template.yml model_config.yml
   ```

2. Edit `model_config.yml` and replace `REPLACE_WITH_YOUR_API_KEY` with your actual OpenAI API key.

### Usage

Run the full scientific collaboration simulation with AI agents:

```bash
python main.py --model-config model_config.yml
```

Optional arguments:
- `--verbose`: Enable detailed logging
- `--num-rounds`: Number of discussion rounds (default: 4)
- `--model-config`: Path to model configuration file

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