"""
Tests for agent naming restrictions.
"""
import pytest
import sys
import os

# Add the source path to test the actual implementation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autogen_core._agent_id import AgentId, is_valid_agent_type
from autogen_core._agent_type import AgentType


class TestAgentNamingRestrictions:
    """Test the updated agent naming rules."""
    
    def test_valid_agent_names(self):
        """Test that valid agent names are accepted."""
        valid_names = [
            "simple_agent",
            "agent-name", 
            "agent123",
            "Agent_With_Underscores",
            "a",  # Single character
            "Agent_123-test",  # Mixed case with numbers and valid symbols
            "weather_agent",  # Realistic examples
            "PlayerBlack",
            "PlayerWhite"
        ]
        
        for name in valid_names:
            # Should not raise an exception
            agent_id = AgentId(name, "default")
            assert agent_id.type == name
            assert agent_id.key == "default"
            assert str(agent_id) == f"{name}/default"

    def test_invalid_agent_names_with_dots(self):
        """Test that agent names with dots are now rejected."""
        invalid_names_with_dots = [
            "agent.with.dots",
            "my.dotted.agent", 
            "service.v1.agent",
            "old.style.name",
            "a.b"
        ]
        
        for name in invalid_names_with_dots:
            with pytest.raises(ValueError, match=r"Invalid agent type.*[\w\-]+"):
                AgentId(name, "default")

    def test_invalid_agent_names_other(self):
        """Test that other invalid agent names continue to be rejected."""
        invalid_names = [
            "agent with spaces",
            "agent@symbol",
            "",  # Empty string
            "agent/slash",
            "agent+plus",
            "agent*asterisk",
            "agent(parentheses)",
            "agent[brackets]",
            "agent{braces}",
            "agent=equals",
            "agent:colon",
            "agent;semicolon",
            "agent,comma"
        ]
        
        for name in invalid_names:
            with pytest.raises(ValueError, match=r"Invalid agent type.*[\w\-]+"):
                AgentId(name, "default")

    def test_agent_type_wrapper_validation(self):
        """Test that AgentType wrapper also applies validation."""
        # Valid AgentType should work
        agent_type = AgentType("valid_agent")
        agent_id = AgentId(agent_type, "default")
        assert agent_id.type == "valid_agent"
        
        # Invalid AgentType should fail
        with pytest.raises(ValueError):
            invalid_type = AgentType("invalid.agent")
            AgentId(invalid_type, "default")

    def test_is_valid_agent_type_function(self):
        """Test the validation function directly."""
        assert is_valid_agent_type("valid_agent") == True
        assert is_valid_agent_type("agent-name") == True
        assert is_valid_agent_type("agent123") == True
        
        assert is_valid_agent_type("agent.with.dots") == False
        assert is_valid_agent_type("agent with spaces") == False
        assert is_valid_agent_type("agent@symbol") == False
        assert is_valid_agent_type("") == False

    def test_regex_pattern_documentation(self):
        """Test that the error message reflects the new regex pattern."""
        try:
            AgentId("invalid.name", "default")
        except ValueError as e:
            assert r"[\w\-]+" in str(e)
            # Should not contain the old pattern with dots
            assert r"[\w\-\.]+" not in str(e)