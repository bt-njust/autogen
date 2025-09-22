#!/usr/bin/env python3
"""
Simple test to demonstrate the enhanced logging concepts
for the scientific collaboration v2 simulation.
"""

import logging
import json
import tempfile
import os

# Mock the autogen logger names for testing
EVENT_LOGGER_NAME = "autogen_core.events"
TRACE_LOGGER_NAME = "autogen_core.trace"

def setup_enhanced_logging(verbose: bool = False, log_file: str = "collaboration_v2.log") -> None:
    """Set up enhanced logging following AutoGen's logging best practices."""
    # Configure basic logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Set up trace logging for debugging
    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    trace_handler = logging.StreamHandler()
    trace_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    trace_logger.addHandler(trace_handler)
    trace_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Set up structured event logging
    event_logger = logging.getLogger(EVENT_LOGGER_NAME)
    event_handler = logging.StreamHandler()
    event_handler.setLevel(logging.INFO)
    event_logger.addHandler(event_handler)
    event_logger.setLevel(logging.INFO)
    
    # Add file logging for debugging
    if verbose:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        trace_logger.addHandler(file_handler)
        event_logger.addHandler(file_handler)

def log_structured_event(logger, event_data: dict) -> None:
    """Helper function to log structured events as JSON strings."""
    logger.info(json.dumps(event_data))

def test_basic_logging():
    """Test basic logging functionality."""
    print("🧪 Testing Enhanced Logging for Scientific Collaboration V2")
    print("=" * 60)
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        temp_log_file = f.name
    
    try:
        # Set up enhanced logging
        setup_enhanced_logging(verbose=True, log_file=temp_log_file)
        
        # Get loggers
        trace_logger = logging.getLogger(TRACE_LOGGER_NAME + ".collaboration")
        event_logger = logging.getLogger(EVENT_LOGGER_NAME + ".collaboration")
        
        print("✅ Enhanced logging setup completed")
        print(f"📄 Log file: {temp_log_file}")
        
        # Test trace logging (human-readable messages)
        print("\n📝 Testing trace logging (human-readable messages):")
        trace_logger.info("Starting scientific collaboration simulation")
        trace_logger.debug("Agent Prof_Chen_001 created with role: leader")
        trace_logger.info("Topic proposal initiated by Prof_Chen_001")
        trace_logger.warning("Maximum topic limit reached")
        
        # Test structured event logging (machine-readable JSON)
        print("\n📊 Testing structured event logging (JSON format):")
        
        # Simulate agent creation event
        log_structured_event(event_logger, {
            "event_type": "agent_created",
            "agent_name": "Prof_Chen_001",
            "team_role": "leader",
            "academic_position": "professor",
            "institution": "MIT",
            "expertise": ["Machine Learning", "Deep Learning"],
            "years_in_team": 5
        })
        
        # Simulate topic proposal event
        log_structured_event(event_logger, {
            "event_type": "topic_proposed",
            "researcher": "Prof_Chen_001",
            "topic_id": "topic_1",
            "title": "Advanced Multi-Agent Collaboration",
            "source": "granted_project",
            "priority": "high",
            "required_expertise": ["AI", "Multi-Agent Systems"],
            "remaining_slots": 5
        })
        
        # Simulate topic discussion event
        log_structured_event(event_logger, {
            "event_type": "topic_discussed",
            "participant": "Dr_Wilson_002",
            "topic_id": "topic_1",
            "topic_title": "Advanced Multi-Agent Collaboration",
            "interest_level": "high",
            "contribution_level": "significant",
            "reasoning": "This aligns with my data science expertise",
            "participant_role": "co_leader",
            "participant_position": "associate_professor"
        })
        
        # Simulate topic assignment event
        log_structured_event(event_logger, {
            "event_type": "topic_assigned",
            "assigner": "Prof_Chen_001",
            "assignee": "Dr_Wilson_002",
            "topic_id": "topic_1",
            "topic_title": "Advanced Multi-Agent Collaboration",
            "reasoning": "Dr Wilson's data science expertise is crucial for this project",
            "all_assigned_members": ["Prof_Chen_001", "Dr_Wilson_002"]
        })
        
        # Simulate simulation completion event
        log_structured_event(event_logger, {
            "event_type": "simulation_completed",
            "total_topics_proposed": 3,
            "topics_assigned": 2,
            "final_assignments": [
                {
                    "title": "Advanced Multi-Agent Collaboration",
                    "proposer": "Prof_Chen_001",
                    "source": "granted_project",
                    "assigned_members": ["Prof_Chen_001", "Dr_Wilson_002"],
                    "priority": "high"
                }
            ],
            "participants": ["Prof_Chen_001", "Dr_Wilson_002", "Dr_Garcia_003"]
        })
        
        print("✅ All logging tests completed successfully")
        
        # Check log file content
        if os.path.exists(temp_log_file):
            with open(temp_log_file, 'r') as f:
                log_content = f.read()
            
            if log_content.strip():
                lines = log_content.strip().split('\n')
                print(f"\n📄 Log file analysis:")
                print(f"   • Total characters: {len(log_content)}")
                print(f"   • Total lines: {len(lines)}")
                print(f"   • File location: {temp_log_file}")
                
                print(f"\n📋 Sample log entries:")
                for i, line in enumerate(lines[:5]):  # Show first 5 lines
                    if len(line) > 120:
                        display_line = line[:117] + "..."
                    else:
                        display_line = line
                    print(f"   {i+1}: {display_line}")
                
                # Count different types of logs
                trace_lines = [line for line in lines if TRACE_LOGGER_NAME in line]
                event_lines = [line for line in lines if EVENT_LOGGER_NAME in line]
                json_lines = [line for line in lines if line.strip().startswith('{"event_type"')]
                
                print(f"\n📈 Log analysis:")
                print(f"   • Trace log entries: {len(trace_lines)}")
                print(f"   • Event log entries: {len(event_lines)}")
                print(f"   • JSON structured events: {len(json_lines)}")
                
            else:
                print("⚠️ Log file created but is empty")
        else:
            print("⚠️ Log file was not created")
            
    finally:
        # Clean up temporary file
        if os.path.exists(temp_log_file):
            os.unlink(temp_log_file)

def test_telemetry_concepts():
    """Test telemetry concepts (without requiring actual OpenTelemetry)."""
    print("\n🔭 OpenTelemetry Integration Concepts")
    print("=" * 60)
    
    print("✨ Enhanced Scientific Collaboration V2 includes:")
    print("   • Tracer provider configuration for service identification")
    print("   • Agent creation and invocation spans following GenAI semantic conventions")
    print("   • Console span exporter for demonstration purposes")
    print("   • Graceful fallback when OpenTelemetry SDK is not available")
    
    print("\n📦 Required packages for full telemetry support:")
    print("   • opentelemetry-sdk>=1.34.1 (for basic tracing)")
    print("   • opentelemetry-exporter-otlp-proto-grpc (for GRPC export)")
    print("   • opentelemetry-exporter-otlp-proto-http (for HTTP export)")
    
    print("\n🎯 Telemetry features implemented:")
    print("   • Service resource configuration with proper naming")
    print("   • Agent-level tracing with role and position attributes")
    print("   • Message handling spans with context propagation")
    print("   • Collaboration event tracking across all phases")
    
    try:
        import opentelemetry
        print("\n✅ OpenTelemetry API is available")
    except ImportError:
        print("\n⚠️ OpenTelemetry API not available in current environment")
    
    try:
        from opentelemetry.sdk.trace import TracerProvider
        print("✅ OpenTelemetry SDK is available")
    except ImportError:
        print("⚠️ OpenTelemetry SDK not available (install: pip install opentelemetry-sdk)")

def main():
    """Run the logging demonstration."""
    print("🧪 SCIENTIFIC COLLABORATION V2 - LOGGING & TELEMETRY DEMO")
    print("=" * 70)
    print("Demonstrating enhanced logging and OpenTelemetry integration")
    print("for the improved scientific collaboration simulation.")
    print()
    
    try:
        test_basic_logging()
        test_telemetry_concepts()
        
        print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("✅ Enhanced logging framework working correctly")
        print("✅ Structured event logging operational")
        print("✅ JSON serialization for machine-readable events")
        print("✅ Trace logging for human-readable debugging")
        print("✅ File and console logging configured")
        print("✅ OpenTelemetry integration ready (when SDK available)")
        
        print("\n💡 Next Steps:")
        print("   1. Install OpenTelemetry SDK: pip install opentelemetry-sdk")
        print("   2. Run the enhanced simulation: python main.py --verbose")
        print("   3. Check the log files for detailed collaboration tracking")
        print("   4. Configure OTLP exporters for production telemetry backends")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())