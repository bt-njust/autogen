# Scientific Collaboration V2 Example

This enhanced example demonstrates how AutoGen can simulate realistic academic research scenarios with improved team structures and role-based collaboration. It addresses the requirements specified in the problem statement with significant improvements over the original version.

## Enhanced Features

### Team Role System
- **Leader**: Single team leader with assignment authority
- **Co-leader**: Multiple leaders scenario with shared responsibilities  
- **Incumbent**: Established team member (3+ years) with experience
- **Newcomer**: New team member (1-2 collaborations) with fresh perspectives

### Academic Position Hierarchy
- **Professor**: Often has grant projects, can assign research directions
- **Associate Professor**: Executive leadership responsibilities
- **Assistant Professor**: Early career faculty with research independence
- **Postdoc**: Specialized skills, significant project responsibility
- **PhD Candidate**: May propose dissertation topics or receive assignments

### Discussion-Based Consensus
- **No voting strategy**: Replaced with discussion-based consensus
- **Interest levels**: high, medium, low, none
- **Contribution levels**: lead, significant, moderate, minimal
- **Reasoning**: Detailed explanations for decisions
- **Assignment mechanism**: Leaders assign based on expertise and workload

### Topic Source System
- **Granted Project**: From existing funded research
- **Future Grant Project**: For upcoming funding applications
- **Research Expansion**: Building on previous work
- **Explore New Directions**: Novel research areas
- **Assignment from Professor**: Top-down research direction
- **PhD Initiative**: Student-driven research proposal

### Improved Asyncio Handling
- **Timeout protection**: Prevents hanging on model calls
- **Message limits**: Prevents endless loops and overwhelming
- **Phase termination**: Maximum rounds per phase with early termination
- **Clean shutdown**: Proper runtime and client cleanup
- **Loop prevention**: Agents don't respond to their own messages

### Valid Researcher Names
All researcher names use only valid characters (numbers, '_', '-'):
- `Prof_Chen_001` (Professor, Leader)
- `Dr_Wilson_002` (Associate Professor, Co-leader)
- `Dr_Garcia_003` (Assistant Professor, Incumbent)
- `Dr_Kim_004` (Postdoc, Incumbent)
- `PhD_Zhang_005` (PhD Candidate, Newcomer)
- `Postdoc_Lee_006` (Postdoc, Newcomer)

## 4-Phase Collaboration Process

### 1. Introduction Phase
- Team members introduce themselves
- Share expertise, role, and current workload
- Suggest potential collaboration areas
- Build understanding of team dynamics

### 2. Proposal Phase
- Propose research topics based on role and position
- Include topic source and required expertise
- Consider team structure and capabilities
- Leaders may propose grant-based projects

### 3. Discussion Phase
- Assess topics based on expertise and interests
- Indicate interest and contribution levels
- Consider current workload and capacity
- Leaders evaluate assignment possibilities

### 4. Consensus Phase
- Finalize topic assignments
- Leaders make decisions based on discussions
- Establish next steps and responsibilities
- Set timelines and collaboration structures

## Usage

### Quick Demo (No API Key Required)
```bash
python demo.py
```

### Full Simulation
```bash
# With default configuration
python main.py

# With custom configuration
python main.py --config-file your_config.ini --config-section your_section

# With verbose logging
python main.py --verbose

# With custom phase rounds
python main.py --num-rounds 2
```

### Configuration File Format
Create a `.server_deployed_LLMs` file:
```ini
[ali_official]
base_url = https://your-api-endpoint.com/v1
api_key = your-api-key-here
```

## Key Improvements Over Original Version

### 1. Enhanced Role System
- Clear hierarchy and responsibilities
- Role-based tool access (only leaders can assign)
- Newcomers may receive assignments rather than propose

### 2. No Voting Strategy
- Discussion-based consensus prevents tied votes
- Qualitative assessment with reasoning
- Leader-driven final decisions

### 3. Topic Source Information
- Clear origin of research ideas
- Different proposal styles based on source
- Realistic academic collaboration scenarios

### 4. Termination Controls
- Maximum rounds per phase (default: 3)
- Early termination conditions
- Prevents endless discussions

### 5. Asyncio Improvements
- Timeout protection (30s for model calls, 10s for messages)
- Message count limits per round
- Clean shutdown procedures
- Loop prevention mechanisms

### 6. Realistic Team Dynamics
- Workload considerations (heavy, moderate, light)
- Years in team affecting behavior
- Position-appropriate proposal patterns
- Assignment delegation from leaders

## Example Output

```
🧪 SCIENTIFIC COLLABORATION V2 SIMULATION
============================================================
Enhanced academic research collaboration with:
• Team roles: leader, co-leader, incumbent, newcomer
• Academic positions: professor, associate prof, assistant prof, postdoc, PhD
• Discussion-based consensus (no voting)
• Topic sources and assignment mechanisms
• Improved asyncio handling and termination conditions

🎯 STARTING PHASE: INTRODUCTION
============================================================
🔄 COLLABORATION ROUND 1: INTRODUCTION
============================================================

🔬 NEW TOPIC PROPOSED by Prof_Chen_001
📋 Title: Federated Learning for Privacy-Preserving Healthcare AI
📝 Description: Develop federated learning systems for healthcare applications...
🎯 Source: granted_project
🔧 Required expertise: Machine Learning, Healthcare Analytics, Privacy Technology
⚡ Priority: high
🆔 Topic ID: topic_1
📊 Remaining topic slots: 5

💬 TOPIC DISCUSSION by Dr_Wilson_002
📋 Topic: Federated Learning for Privacy-Preserving Healthcare AI
💡 Interest level: high
🤝 Contribution level: significant
💭 Reasoning: This aligns perfectly with my healthcare analytics expertise...
👤 Role: co_leader (associate_professor)
📊 Current workload: moderate

📋 TOPIC ASSIGNMENT by Prof_Chen_001
📝 Topic: Federated Learning for Privacy-Preserving Healthcare AI
👤 Assigned to: Dr_Wilson_002
💭 Reasoning: Dr. Wilson showed high interest and significant capability...
👥 All assigned members: Dr_Wilson_002

🏆 FINAL TOPIC ASSIGNMENTS:
📋 Federated Learning for Privacy-Preserving Healthcare AI
   👤 Proposed by: Prof_Chen_001
   🎯 Source: granted_project
   👥 Assigned to: Dr_Wilson_002, Dr_Garcia_003
   ⚡ Priority: high
```

## Files

- `main.py`: Enhanced main simulation with all improvements
- `demo.py`: Demonstration script (no API key required)
- `README.md`: This documentation file

## Technical Details

### State Management
- `CollaborationState`: Enhanced state tracking
- Phase round counting with termination
- Topic status tracking (proposed, discussed, accepted, declined)
- Researcher workload and role management

### Agent Architecture
- `ResearcherAgent`: Enhanced with role-based behavior
- Message count limiting and timeout protection
- Tool access based on team role and position
- Loop prevention and clean shutdown

### Tool System
- `propose_topic`: Include source and priority
- `discuss_topic`: Interest/contribution assessment
- `assign_to_topic`: Leader-only assignment capability
- Status and profile checking tools

This enhanced version provides a more realistic and robust simulation of academic research collaboration while addressing all the identified issues from the original implementation.