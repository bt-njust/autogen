# Scientific Collaboration Example

This example demonstrates how AutoGen can simulate realistic academic research scenarios where multiple researchers discuss and select joint research topics based on their profiles and collaboration history.

## Features

- Multiple researcher agents with distinct expertise profiles
- Collaboration history tracking
- Interactive topic proposal and discussion
- Consensus-building mechanisms
- Realistic academic conversation patterns

## Setup

1. Copy the model configuration template:
   ```bash
   cp model_config_template.yml model_config.yml
   ```

2. Edit `model_config.yml` and replace `REPLACE_WITH_YOUR_API_KEY` with your actual OpenAI API key.

## Usage

Run the scientific collaboration simulation:

```bash
python main.py --model-config model_config.yml
```

Optional arguments:
- `--verbose`: Enable detailed logging
- `--num-rounds`: Number of discussion rounds (default: 3)
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