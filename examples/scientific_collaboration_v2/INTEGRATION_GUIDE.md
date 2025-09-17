# Integration Guide for Scientific Collaboration V2

This guide helps you integrate and customize the enhanced scientific collaboration example for your specific needs.

## Quick Start

1. **Test without API**: Run the standalone demo first
   ```bash
   python demo_standalone.py
   ```

2. **Set up configuration**: Copy and edit the template
   ```bash
   cp .server_deployed_LLMs.template .server_deployed_LLMs
   # Edit .server_deployed_LLMs with your API details
   ```

3. **Run with AI agents**:
   ```bash
   python main.py --config-section your_section
   ```

## Customization Options

### Modify Team Structure

Edit the `researchers` list in `setup_collaboration()` in `main.py`:

```python
researchers = [
    ResearcherProfile(
        name="Your_Researcher_001",  # Use only numbers, '_', '-'
        academic_position=AcademicPosition.PROFESSOR,
        team_role=TeamRole.LEADER,
        expertise=["Your", "Expertise", "Areas"],
        institution="Your Institution",
        research_interests=["Your", "Research", "Interests"],
        recent_publications=["Your Recent Publications"],
        years_in_team=5,
        current_workload="moderate"  # light, moderate, heavy
    ),
    # Add more researchers...
]
```

### Adjust Collaboration Parameters

Modify the `CollaborationState` initialization:

```python
collaboration_state = CollaborationState(
    max_topics=8,           # Maximum topics allowed
    max_rounds_per_phase=4  # Maximum rounds per phase
)
```

### Customize Topic Sources

Add new topic sources in the `TopicSource` enum:

```python
class TopicSource(Enum):
    GRANTED_PROJECT = "granted_project"
    FUTURE_GRANT_PROJECT = "future_grant_project"
    RESEARCH_EXPANSION = "research_expansion"
    EXPLORE_NEW_DIRECTIONS = "explore_new_directions"
    ASSIGNMENT_FROM_PROFESSOR = "assignment_from_professor"
    PHD_INITIATIVE = "phd_initiative"
    # Add your custom sources here
    INDUSTRY_COLLABORATION = "industry_collaboration"
    CONFERENCE_FOLLOWUP = "conference_followup"
```

### Modify Phase Prompts

Edit the prompts in `run_collaboration_round()`:

```python
if phase == "introduction":
    prompt = "Your custom introduction prompt..."
elif phase == "proposal":
    prompt = "Your custom proposal prompt..."
# etc.
```

### Add New Tools

Create additional tools for researchers:

```python
def new_tool_function(
    researcher_name: str,
    param1: Annotated[str, "Description"],
    param2: Annotated[str, "Description"],
) -> Annotated[str, "Result description"]:
    """Your new tool function."""
    # Implementation here
    return "Result"

# Add to the tools list in make_researcher_tools()
tools.append(FunctionTool(
    new_tool_function,
    name="new_tool",
    description="Description of new tool",
))
```

### Adjust Timeout Settings

Modify timeout values in `ResearcherAgent.handle_message()`:

```python
messages = await asyncio.wait_for(
    tool_agent_caller_loop(...),
    timeout=60.0  # Increase timeout if needed
)
```

### Configure Logging

Enable detailed logging for debugging:

```python
python main.py --verbose
```

Or customize logging in code:

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("autogen_core").setLevel(logging.DEBUG)
```

## Integration with Other Systems

### Database Integration

Add database logging for collaboration results:

```python
import sqlite3

def save_collaboration_results():
    conn = sqlite3.connect('collaboration.db')
    # Save topics, assignments, discussions to database
    conn.close()
```

### Web Interface

Create a web interface using FastAPI or Flask:

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/start_collaboration")
async def start_collaboration(config: CollaborationConfig):
    # Run collaboration and return results
    pass
```

### Notification System

Add notifications for assignments:

```python
def notify_assignment(assignee: str, topic: str):
    # Send email, Slack message, etc.
    pass
```

## Best Practices

### 1. Researcher Names
- Always use only numbers, underscores, and hyphens
- Examples: `Prof_Smith_001`, `Dr_Lee_002`, `PhD_Wang_003`

### 2. Error Handling
- Wrap API calls in try-catch blocks
- Implement fallback behaviors for timeouts
- Log errors for debugging

### 3. Performance
- Adjust buffer sizes based on your needs
- Monitor memory usage with large teams
- Use appropriate timeout values

### 4. Testing
- Test with `demo_standalone.py` first
- Verify all customizations work
- Test with different team sizes

### 5. Configuration Management
- Keep sensitive API keys secure
- Use environment variables for production
- Document configuration options

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install autogen packages
   ```bash
   pip install autogen-core autogen-ext
   ```

2. **API Timeout**: Increase timeout values or check network

3. **Endless Loops**: Check message handling logic

4. **Memory Issues**: Reduce buffer sizes or team size

### Debug Mode

Run with verbose logging to see detailed information:

```bash
python main.py --verbose --config-section your_section
```

## Example Configurations

### Small Team (3-4 members)
```python
max_topics=4
max_rounds_per_phase=2
buffer_size=5
```

### Large Team (8+ members)
```python
max_topics=10
max_rounds_per_phase=4
buffer_size=15
```

### Quick Testing
```python
max_topics=3
max_rounds_per_phase=1
timeout=15.0
```

## Support

- Review the README.md for feature explanations
- Check demo_standalone.py for working examples
- Examine the original examples/scientific_collaboration for comparison
- Use verbose logging for debugging issues