import importlib
import json
import os
import re
import signal
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import ToolCommand

# Tool name mapping removed - using direct tool names from external input

try:
    TimeoutError
except NameError:
    class TimeoutError(Exception):
        pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

class Executor:
    def __init__(self, llm_engine_name: str, root_cache_dir: str = "solver_cache",  num_threads: int = 1, max_time: int = 120,
    max_output_length: int = 100000, verbose: bool = False, base_url: str = None, check_model: bool = True, temperature: float = .0, tool_instances: Dict[str, Any] = None):
        self.llm_engine_name = llm_engine_name
        self.root_cache_dir = root_cache_dir
        self.num_threads = num_threads
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.verbose = verbose
        self.base_url = base_url
        self.check_model = check_model
        self.temperature  = temperature
        self.tool_instances = tool_instances or {}  # Cache of tool instances from Initializer

        # Only pass base_url for vLLM models, not for remote APIs like DashScope, OpenAI, etc.
        if "vllm" in self.llm_engine_name and base_url:
            self.llm_generate_tool_command = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=False, base_url=self.base_url, temperature = self.temperature)
        else:
            self.llm_generate_tool_command = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=False, temperature = self.temperature)
    
    def set_query_cache_dir(self, query_cache_dir):
        if query_cache_dir:
            self.query_cache_dir = query_cache_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_cache_dir = os.path.join(self.root_cache_dir, timestamp)
        os.makedirs(self.query_cache_dir, exist_ok=True)

    def generate_tool_command(self, question: str, image: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any], step_count: int = 0, json_data: Any = None) -> Any:
        prompt_generate_tool_command = f"""
        Task: Generate a precise command to execute the selected tool.

Context:
- **Query:** {question}
- **Sub-Goal:** {sub_goal}
- **Tool Name:** {tool_name}
- **Tool Metadata:** {tool_metadata}
- **Relevant Data:** {context}

Instructions:
1.  Analyze the tool's required parameters from its metadata.
2.  Construct valid Python code that addresses the sub-goal using the provided context and data.
3.  The command must include at least one call to `tool.execute()`.
4.  Each `tool.execute()` call must be assigned to a variable named **`execution`**.
5.  Please give the exact numbers and parameters should be used in the `tool.execute()` call.

Output Format:
Present your response in the following structured format. Do not include any extra text or explanations.

Generated Command:
```python
<command>
```

Example1:
Generated Command:
```python
execution = tool.execute(query="Summarize the following porblom:"Isaac has 100 toys, masa gets ...., how much are their together?")
```

Example2:
Generated Command:
```python
execution = tool.execute(query=["Methanol", "function of hyperbola", "Fermat's Last Theorem"])
```
"""

        tool_command = self.llm_generate_tool_command(prompt_generate_tool_command, response_format=ToolCommand)
        if json_data is not None:
            json_data[f"tool_commander_{step_count}_prompt"] = prompt_generate_tool_command
            json_data[f"tool_commander_{step_count}_response"] = str(tool_command)

        return tool_command

    def extract_explanation_and_command(self, response: Any) -> tuple:
        def normalize_code(code: str) -> str:
            # Remove leading/trailing whitespace and triple backticks if present
            return re.sub(r'^```python\s*', '', code).rstrip('```').strip()

        analysis = "No analysis found."
        explanation = "No explanation found."
        command = "No command found."

        if isinstance(response, str):
            # Attempt to parse as JSON first
            try:
                response_dict = json.loads(response)
                response_obj = ToolCommand(**response_dict)
                analysis = response_obj.analysis.strip()
                explanation = response_obj.explanation.strip()
                command = response_obj.command.strip()
            except Exception as e:
                print(f"Failed to parse response as JSON: {str(e)}")
                # Fall back to regex parsing on string
                try:
                    # Extract analysis
                    analysis_pattern = r"Analysis:(.*?)Command Explanation"
                    analysis_match = re.search(analysis_pattern, response, re.DOTALL | re.IGNORECASE)
                    analysis = analysis_match.group(1).strip() if analysis_match else "No analysis found."

                    # Extract explanation
                    explanation_pattern = r"Command Explanation:(.*?)Generated Command"
                    explanation_match = re.search(explanation_pattern, response, re.DOTALL | re.IGNORECASE)
                    explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found."

                    # Extract command using "Generated Command:" prefix
                    command_pattern = r"Generated Command:.*?```python\n(.*?)```"
                    command_match = re.search(command_pattern, response, re.DOTALL | re.IGNORECASE)
                    if command_match:
                        command = command_match.group(1).strip()
                    else:
                        # Fallback: Extract ANY ```python ... ``` block (even without prefix)
                        loose_command_pattern = r"```python\s*\n(.*?)```"
                        loose_match = re.findall(loose_command_pattern, response, re.DOTALL | re.IGNORECASE)
                        if loose_match:
                            # Take the last or most complete one? Or first meaningful?
                            # Here we take the longest one as heuristic
                            command = max(loose_match, key=lambda x: len(x.strip())).strip()
                        else:
                            command = "No command found."
                except Exception as e:
                    print(f"Error during regex parsing: {str(e)}")
                    analysis = "Parsing error."
                    explanation = "Parsing error."
                    command = "No command found."
        elif isinstance(response, ToolCommand):
            analysis = response.analysis.strip()
            explanation = response.explanation.strip()
            command = response.command.strip()
        else:
            command = "Invalid response type."

        # Final normalization
        command = normalize_code(command)

        return analysis, explanation, command

    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        """
        Execute a tool command with timeout protection. If execution exceeds max_time seconds,
        the function will be interrupted and return a timeout message.

        Args:
            tool_name (str): Name of the tool to execute
            command (str): Command string containing tool.execute() calls

        Returns:
            Any: List of execution results or error message
        """

        def split_commands(command: str) -> List[str]:
            # Use regex to find all tool.execute() commands and their surrounding code
            pattern = r'.*?execution\s*=\s*tool\.execute\([^\n]*\)\s*(?:\n|$)'
            blocks = re.findall(pattern, command, re.DOTALL)
            return [block.strip() for block in blocks if block.strip()]

        def execute_with_timeout(block: str, local_context: dict) -> Optional[str]:
            # Set up the timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.max_time)

            try:
                # Execute the block in the local context
                exec(block, globals(), local_context)
                result = local_context.get('execution')
                signal.alarm(0)  # Disable the alarm
                return result
            except TimeoutError:
                return f"Execution timed out after {self.max_time} seconds"
            finally:
                signal.alarm(0)  # Ensure alarm is disabled even if other exceptions occur

        try:
            # Use cached tool instance if available
            if tool_name in self.tool_instances:
                tool = self.tool_instances[tool_name]
                print(f"Using cached tool instance for {tool_name}")
            else:
                # Fallback: dynamically create tool instance (backward compatibility)
                print(f"Warning: Tool {tool_name} not found in cache, creating new instance")
                class_name = tool_name
                dir_name = tool_name.lower().replace('_tool', '')
                module_name = f"tools.{dir_name}.tool"

                module = importlib.import_module(module_name)
                tool_class = getattr(module, class_name)
                tool = tool_class()

            # Set the custom output directory
            tool.set_custom_output_dir(self.query_cache_dir)

            # Split the command into blocks, execute each one and store execution results
            command_blocks = split_commands(command)
            executions = []

            for block in command_blocks:
                # Create a local context to safely execute the block
                local_context = {'tool': tool}

                # Execute the block with timeout protection
                result = execute_with_timeout(block, local_context)

                if result is not None:
                    executions.append(result)
                else:
                    executions.append(f"No execution captured from block: {block}")

            # Return all the execution results
            return executions
        except Exception as e:
            return f"Error in execute_tool_command: {str(e)}"
