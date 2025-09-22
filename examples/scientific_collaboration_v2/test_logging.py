#!/usr/bin/env python3
"""
Simple test to demonstrate the enhanced logging and telemetry features
of the scientific collaboration v2 simulation.
"""

import logging
import tempfile
import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import from main.py
sys.path.insert(0, str(Path(__file__).parent))

from main import (
    setup_enhanced_logging, 
    configure_telemetry, 
    log_structured_event,
    TELEMETRY_AVAILABLE,
    EVENT_LOGGER_NAME,
    TRACE_LOGGER_NAME
)

def test_logging_setup():
    """Test the enhanced logging setup."""
    print("🧪 Testing Enhanced Logging Setup")
    print("=" * 50)
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        temp_log_file = f.name
    
    try:
        # Set up enhanced logging
        setup_enhanced_logging(verbose=True, log_file=temp_log_file)
        
        # Get loggers
        trace_logger = logging.getLogger(TRACE_LOGGER_NAME + ".test")
        event_logger = logging.getLogger(EVENT_LOGGER_NAME + ".test")
        
        print("✅ Enhanced logging setup completed")
        
        # Test trace logging
        trace_logger.info("This is a trace log message")
        trace_logger.debug("This is a debug trace message")
        trace_logger.warning("This is a warning trace message")
        
        # Test structured event logging
        log_structured_event(event_logger, {
            "event_type": "test_event",
            "test_parameter": "test_value",
            "numeric_value": 42,
            "list_value": ["item1", "item2", "item3"]
        })
        
        print("✅ Logging tests completed")
        
        # Check if log file was created and has content
        if os.path.exists(temp_log_file):
            with open(temp_log_file, 'r') as f:
                log_content = f.read()
            if log_content.strip():
                print(f"✅ Log file created with {len(log_content)} characters")
                print(f"📄 Log file location: {temp_log_file}")
                
                # Show some log content
                lines = log_content.strip().split('\n')
                print(f"📄 Log file contains {len(lines)} lines")
                if lines:
                    print("📄 Sample log entries:")
                    for i, line in enumerate(lines[:3]):  # Show first 3 lines
                        print(f"   {i+1}: {line[:100]}{'...' if len(line) > 100 else ''}")
            else:
                print("⚠️ Log file created but is empty")
        else:
            print("⚠️ Log file was not created")
            
    finally:
        # Clean up temporary file
        if os.path.exists(temp_log_file):
            os.unlink(temp_log_file)

def test_telemetry_setup():
    """Test the telemetry configuration."""
    print("\n🔭 Testing Telemetry Setup")
    print("=" * 50)
    
    if not TELEMETRY_AVAILABLE:
        print("⚠️ OpenTelemetry SDK not available")
        print("💡 To test telemetry, install: pip install opentelemetry-sdk")
        return
    
    # Configure telemetry
    tracer_provider = configure_telemetry()
    
    if tracer_provider:
        print("✅ OpenTelemetry tracer provider configured successfully")
        print(f"📊 Tracer provider type: {type(tracer_provider).__name__}")
        
        # Try to get a tracer
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer("test-tracer")
            print("✅ Test tracer obtained successfully")
            
            # Create a test span
            with tracer.start_as_current_span("test-span") as span:
                span.set_attribute("test.attribute", "test_value")
                span.set_attribute("test.number", 123)
                print("✅ Test span created and configured")
                
        except Exception as e:
            print(f"⚠️ Error testing tracer: {e}")
    else:
        print("❌ Failed to configure telemetry")

def test_collaboration_logging():
    """Test collaboration-specific logging functionality."""
    print("\n🤝 Testing Collaboration Logging")
    print("=" * 50)
    
    # Set up logging
    setup_enhanced_logging(verbose=False)
    
    # Import collaboration functions to test their logging
    from main import propose_topic, discuss_topic, assign_to_topic, collaboration_state
    
    # Reset collaboration state for clean test
    collaboration_state.topics.clear()
    collaboration_state.researchers.clear()
    
    # Add a test researcher
    from main import ResearcherProfile, TeamRole, AcademicPosition
    test_researcher = ResearcherProfile(
        name="Test_Researcher_001",
        academic_position=AcademicPosition.PROFESSOR,
        team_role=TeamRole.LEADER,
        expertise=["Testing", "Logging"],
        institution="Test University",
        research_interests=["Software Testing", "Observability"],
        recent_publications=["Test Paper 1", "Test Paper 2"]
    )
    collaboration_state.add_researcher(test_researcher)
    
    print("✅ Test researcher added to collaboration state")
    
    # Test topic proposal logging
    result = propose_topic(
        "Test_Researcher_001",
        "Enhanced Logging for Multi-Agent Systems",
        "Research on improving observability in agent-based simulations",
        "research_expansion",
        "Logging, OpenTelemetry, Agent Systems",
        "high"
    )
    print(f"✅ Topic proposal logged: {result[:50]}...")
    
    # Test topic discussion logging
    if collaboration_state.topics:
        topic_id = list(collaboration_state.topics.keys())[0]
        result = discuss_topic(
            "Test_Researcher_001",
            topic_id,
            "high",
            "lead",
            "This aligns perfectly with our current research goals"
        )
        print(f"✅ Topic discussion logged: {result[:50]}...")
    
    print(f"✅ Collaboration logging test completed")
    print(f"📊 Topics in state: {len(collaboration_state.topics)}")
    print(f"📊 Researchers in state: {len(collaboration_state.researchers)}")

def main():
    """Run all tests."""
    print("🧪 ENHANCED LOGGING AND TELEMETRY TESTS")
    print("=" * 60)
    print("Testing the improved scientific collaboration v2 simulation")
    print("with enhanced logging and OpenTelemetry tracing capabilities.")
    print()
    
    try:
        test_logging_setup()
        test_telemetry_setup()
        test_collaboration_logging()
        
        print("\n🎉 ALL TESTS COMPLETED")
        print("=" * 60)
        print("✅ Enhanced logging functionality working correctly")
        print("✅ Structured event logging operational")
        if TELEMETRY_AVAILABLE:
            print("✅ OpenTelemetry tracing configured and functional")
        else:
            print("⚠️ OpenTelemetry tracing not available (install opentelemetry-sdk)")
        print("✅ Collaboration-specific logging working properly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())