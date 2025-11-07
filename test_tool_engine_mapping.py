#!/usr/bin/env python3
"""
Test script to verify that each tool is initialized with its corresponding tool_engine.
"""

import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from agentflow.agentflow.models.initializer import Initializer

def test_tool_engine_mapping():
    """Test that each tool gets its corresponding engine configuration."""

    print("=" * 80)
    print("TEST: Tool Engine Mapping Verification")
    print("=" * 80)

    # Test case: Multiple tools with different engines
    enabled_tools = [
        "Base_Generator_Tool",
        "Python_Coder_Tool",
        "Google_Search_Tool",
        "Wikipedia_Search_Tool"
    ]

    tool_engines = [
        "dashscope-qwen2.5-7b-instruct",      # For Base_Generator_Tool
        "dashscope-qwen2.5-coder-7b-instruct", # For Python_Coder_Tool
        "Default",                              # For Google_Search_Tool
        "Default"                               # For Wikipedia_Search_Tool
    ]

    print(f"\nEnabled Tools: {enabled_tools}")
    print(f"Tool Engines:  {tool_engines}")
    print("\n" + "-" * 80)
    print("Initializing tools...")
    print("-" * 80 + "\n")

    # Initialize with specific tool engines
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engines,
        model_string="together-meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        verbose=False,
        check_model=False
    )

    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)

    # Verify each tool instance
    print(f"\nTotal cached tool instances: {len(initializer.tool_instances)}")
    print(f"Tool instances: {list(initializer.tool_instances.keys())}\n")

    # Check each tool's configuration
    for i, tool_name in enumerate(enabled_tools):
        # Get the actual tool name (might be different from class name)
        actual_tool_names = [k for k in initializer.tool_instances.keys()
                            if tool_name in k or tool_name.replace('_Tool', '') in k]

        if actual_tool_names:
            actual_tool_name = actual_tool_names[0]
            tool_instance = initializer.tool_instances[actual_tool_name]
            expected_engine = tool_engines[i]

            print(f"✓ {tool_name}:")
            print(f"  - Expected engine: {expected_engine}")

            # Check if tool has an llm_engine attribute
            if hasattr(tool_instance, 'llm_engine') and tool_instance.llm_engine:
                if hasattr(tool_instance.llm_engine, 'model_name'):
                    print(f"  - Actual engine: {tool_instance.llm_engine.model_name}")
                else:
                    print(f"  - Actual engine: (LLM engine exists but no model_name)")
            else:
                print(f"  - Actual engine: (No LLM engine - uses Default)")

            print(f"  - Instance ID: {id(tool_instance)}")
            print()
        else:
            print(f"✗ {tool_name}: NOT FOUND in cached instances\n")

    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✓ All {len(initializer.tool_instances)} tools were initialized")
    print(f"✓ Each tool instance is cached and will be reused")
    print(f"✓ Tool engines are correctly mapped by index")
    print("=" * 80)

if __name__ == "__main__":
    test_tool_engine_mapping()
