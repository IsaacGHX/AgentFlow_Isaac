import os
from agentflow.tools.base import BaseTool

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Math_Calculator_Tool"

LIMITATION = f"""
The {TOOL_NAME} executes Python code and may:
1. Raise exceptions if the code is invalid or contains errors.
2. Be limited to mathematical operations and basic Python expressions.
3. Not support imports or file I/O operations for security reasons.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Provide valid Python mathematical expressions (e.g., "9**3", "2 + 2 * 3").
2. Use Python syntax for operations (** for power, not ^).
3. You can use math functions like abs(), round(), min(), max().
4. For complex expressions, ensure proper parentheses for order of operations.
5. Keep expressions simple and focused on calculations.
"""

class Math_Calculator_Tool(BaseTool):
    require_llm_engine = False

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A tool that evaluates mathematical expressions or simple Python code snippets and returns the result.",
            tool_version="1.0.0",
            input_types={
                "expression": "str - A mathematical expression or Python code snippet to evaluate (e.g., '9**3', '2 + 2 * 3').",
            },
            output_type="str - The evaluated result of the expression or an error message",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(expression="9**3")',
                    "description": "Calculate 9 to the power of 3."
                },
                {
                    "command": 'execution = tool.execute(expression="(5 + 3) * 2")',
                    "description": "Evaluate a mathematical expression with parentheses."
                },
                {
                    "command": 'execution = tool.execute(expression="abs(-42)")',
                    "description": "Calculate the absolute value of -42."
                },
                {
                    "command": 'execution = tool.execute(expression="round(3.14159, 2)")',
                    "description": "Round a number to 2 decimal places."
                }
            ],

            user_metadata = {
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }

        )
        print(f"Initializing {TOOL_NAME}")


    def execute(self, expression):
        """
        Evaluates a mathematical expression or simple Python code snippet.
        
        Args:
            expression (str): The expression to evaluate
            
        Returns:
            str: The result of the evaluation or an error message
        """
        try:
            # Create a safe environment with only basic math functions
            safe_dict = {
                '__builtins__': {
                    'abs': abs,
                    'round': round,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'pow': pow,
                    'len': len,
                }
            }
            
            # Evaluate the expression
            result = eval(expression, safe_dict, {})
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd agentflow/tools/math_caculator
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # Example usage of the Math_Calculator_Tool
    tool = Math_Calculator_Tool()

    # Get tool metadata
    metadata = tool.get_metadata()
    print("\n=== Tool Metadata ===")
    print(metadata)

    # Test expressions
    test_expressions = [
        "9**3",
        "(5 + 3) * 2",
        "abs(-42)",
        "round(3.14159, 2)",
        "sum([1, 2, 3, 4, 5])",
        "max(10, 20, 30)"
    ]

    print("\n=== Testing Math Calculator Tool ===")
    for expr in test_expressions:
        try:
            result = tool.execute(expression=expr)
            print(f"{expr} = {result}")
        except Exception as e: 
            print(f"Execution failed for '{expr}': {e}")

    print("\nDone!")
