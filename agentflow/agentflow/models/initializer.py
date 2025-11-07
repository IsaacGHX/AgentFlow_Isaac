import os
import sys
import importlib
import inspect
import traceback
from typing import Dict, Any, List, Tuple
import time

class Initializer:
    def __init__(self,
    enabled_tools: List[str] = [],
    tool_engine: List[str] = [],
    model_string: str = None,
    verbose: bool = False,
    vllm_config_path: str = None,
    base_url: str = None,
    check_model: bool = True):

        self.toolbox_metadata = {}
        self.tool_instances = {}  # Cache tool instances to avoid re-initialization
        self.available_tools = []
        self.enabled_tools = enabled_tools
        self.tool_engine = tool_engine
        self.load_all = self.enabled_tools == ["all"]
        self.model_string = model_string
        self.verbose = verbose
        self.vllm_server_process = None
        self.vllm_config_path = vllm_config_path
        self.base_url = base_url
        self.check_model = check_model
        print("\n==> Initializing agentflow...")
        print(f"Enabled tools: {self.enabled_tools} with {self.tool_engine}")
        print(f"LLM engine name: {self.model_string}")
        self._set_up_tools()
        
        # if vllm, set up the vllm server
        # if model_string.startswith("vllm-"):
        #     self.setup_vllm_server()

    def get_project_root(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir != '/':
            if os.path.exists(os.path.join(current_dir, 'agentflow')):
                return os.path.join(current_dir, 'agentflow')
            current_dir = os.path.dirname(current_dir)
        raise Exception("Could not find project root")

    def get_tool_directory_mapping(self, tools_dir: str) -> Dict[str, str]:
        """
        Build a simple mapping from tool class names to their directory names.

        Returns:
            Dict mapping class names to directory names (e.g., Base_Generator_Tool -> base_generator)
        """
        class_to_dir = {}

        for root, dirs, files in os.walk(tools_dir):
            if 'tool.py' in files:
                dir_name = os.path.basename(root)
                tool_file_path = os.path.join(root, 'tool.py')

                try:
                    # Read the tool.py file and extract class name
                    with open(tool_file_path, 'r') as f:
                        content = f.read()

                    # Find the class name from the file
                    for line in content.split('\n'):
                        if 'class ' in line and 'BaseTool' in line:
                            class_name = line.split('class ')[1].split('(')[0].strip()
                            class_to_dir[class_name] = dir_name
                            print(f"Mapped class: {class_name} -> directory: {dir_name}")
                            break
                except Exception as e:
                    print(f"Warning: Could not process {tool_file_path}: {str(e)}")
                    continue

        return class_to_dir

    def load_tools_and_get_metadata(self) -> Dict[str, Any]:
        # Implementation of load_tools_and_get_metadata function
        print("Loading tools and getting metadata...")
        self.toolbox_metadata = {}
        agentflow_dir = self.get_project_root()
        tools_dir = os.path.join(agentflow_dir, 'tools')
        # print(f"agentflow directory: {agentflow_dir}")
        # print(f"Tools directory: {tools_dir}")

        # Add the agentflow directory and its parent to the Python path
        sys.path.insert(0, agentflow_dir)
        sys.path.insert(0, os.path.dirname(agentflow_dir))
        print(f"Updated Python path: {sys.path}")

        if not os.path.exists(tools_dir):
            print(f"Error: Tools directory does not exist: {tools_dir}")
            return self.toolbox_metadata

        # Build tool directory mapping if not already built
        if not hasattr(self, 'tool_dir_mapping'):
            self.tool_dir_mapping = self.get_tool_directory_mapping(tools_dir)
        print(f"\n==> Tool directory mapping: {self.tool_dir_mapping}")

        for root, dirs, files in os.walk(tools_dir):
            # print(f"\nScanning directory: {root}")
            if 'tool.py' in files and (self.load_all or os.path.basename(root) in self.available_tools):
                file = 'tool.py'
                module_path = os.path.join(root, file)
                module_name = os.path.splitext(file)[0]
                relative_path = os.path.relpath(module_path, agentflow_dir)
                import_path = '.'.join(os.path.split(relative_path)).replace(os.sep, '.')[:-3]

                print(f"\n==> Attempting to import: {import_path}")
                try:
                    module = importlib.import_module(import_path)
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and name.endswith('Tool') and name != 'BaseTool':
                            print(f"Found tool class: {name}")
                            try:
                                # Check if the tool requires specific llm engine
                                tool_index = -1
                                current_dir_name = os.path.basename(root)
                                for i, tool_name in enumerate(self.enabled_tools):
                                    # Check if tool name matches this class/directory
                                    if tool_name == name or tool_name.lower().replace('_tool', '') == current_dir_name:
                                        tool_index = i
                                        break

                                if tool_index >= 0 and tool_index < len(self.tool_engine):
                                    engine = self.tool_engine[tool_index]
                                    print(f"  → Tool '{name}' at index {tool_index} using engine: '{engine}'")
                                    if engine == "Default":
                                        tool_instance = obj()
                                    elif engine == "self":
                                        tool_instance = obj(model_string=self.model_string)
                                    else:
                                        tool_instance = obj(model_string=engine)
                                else:
                                    print(f"  → Tool '{name}' using default engine (no index match)")
                                    tool_instance = obj()
                                # Use the external tool name (from TOOL_NAME) as the key
                                metadata_key = getattr(tool_instance, 'tool_name', name)

                                # Cache the tool instance for reuse
                                self.tool_instances[metadata_key] = tool_instance

                                self.toolbox_metadata[metadata_key] = {
                                    'tool_name': getattr(tool_instance, 'tool_name', 'Unknown'),
                                    'tool_description': getattr(tool_instance, 'tool_description', 'No description'),
                                    'tool_version': getattr(tool_instance, 'tool_version', 'Unknown'),
                                    'input_types': getattr(tool_instance, 'input_types', {}),
                                    'output_type': getattr(tool_instance, 'output_type', 'Unknown'),
                                    'demo_commands': getattr(tool_instance, 'demo_commands', []),
                                    'user_metadata': getattr(tool_instance, 'user_metadata', {}), # This is a placeholder for user-defined metadata
                                    'require_llm_engine': getattr(obj, 'require_llm_engine', False),
                                }
                                print(f"Metadata for {metadata_key}: {self.toolbox_metadata[metadata_key]}")
                            except Exception as e:
                                print(f"Error instantiating {name}: {str(e)}")
                except Exception as e:
                    print(f"Error loading module {module_name}: {str(e)}")
                    
        print(f"\n==> Total number of tools imported: {len(self.toolbox_metadata)}")

        return self.toolbox_metadata

    def run_demo_commands(self) -> List[str]:
        print("\n==> Running demo commands for each tool...")
        self.available_tools = []

        for tool_name, tool_data in self.toolbox_metadata.items():
            print(f"Checking availability of {tool_name}...")

            try:
                # Use cached tool instance instead of creating a new one
                if tool_name in self.tool_instances:
                    # Tool instance already exists and cached
                    self.available_tools.append(tool_name)
                else:
                    print(f"Warning: Tool {tool_name} not found in cached instances")

            except Exception as e:
                print(f"Error checking availability of {tool_name}: {str(e)}")
                print(traceback.format_exc())

        # Update the toolmetadata and tool_instances with the available tools
        self.toolbox_metadata = {tool: self.toolbox_metadata[tool] for tool in self.available_tools}
        self.tool_instances = {tool: self.tool_instances[tool] for tool in self.available_tools}
        print("\n✅ Finished running demo commands for each tool.")
        # print(f"Updated total number of available tools: {len(self.toolbox_metadata)}")
        # print(f"Available tools: {self.available_tools}")
        return self.available_tools
    
    def _set_up_tools(self) -> None:
        print("\n==> Setting up tools...")

        # Build directory mapping by scanning all tools
        agentflow_dir = self.get_project_root()
        tools_dir = os.path.join(agentflow_dir, 'tools')
        self.tool_dir_mapping = self.get_tool_directory_mapping(tools_dir) if os.path.exists(tools_dir) else {}

        # Map input tool names to directory names for filtering
        mapped_tools = []
        for tool in self.enabled_tools:
            # Use tool_dir_mapping to get directory name, or derive it
            if tool in self.tool_dir_mapping:
                mapped_tools.append(self.tool_dir_mapping[tool])
            else:
                # Fallback: derive directory name from tool name
                mapped_tools.append(tool.lower().replace('_tool', ''))

        self.available_tools = mapped_tools

        # Now load tools and get metadata
        self.load_tools_and_get_metadata()

        # Run demo commands to determine available tools
        # This will update self.available_tools to contain tool names
        self.run_demo_commands()

        # available_tools is now updated by run_demo_commands
        print("✅ Finished setting up tools.")
        print(f"✅ Total number of final available tools: {len(self.available_tools)}")
        print(f"✅ Final available tools: {self.available_tools}")

if __name__ == "__main__":
    enabled_tools = ["Base_Generator_Tool", "Python_Coder_Tool"]
    tool_engine = ["Default", "Default"]
    initializer = Initializer(enabled_tools=enabled_tools,tool_engine=tool_engine)

    print("\nAvailable tools:")
    print(initializer.available_tools)

    print("\nToolbox metadata for available tools:")
    print(initializer.toolbox_metadata)
    