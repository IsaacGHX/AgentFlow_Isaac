import os
from agentflow.tools.base import BaseTool
from agentflow.engine.factory import create_llm_engine

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Visual_Generator_Tool"

LIMITATION = f"""
The {TOOL_NAME} may provide hallucinated or incorrect responses about image content.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Provide a clear, specific query about what you want to know from the image.
2. Ensure the image path is valid and accessible.
3. Use descriptive queries like "Describe this image in detail", "What objects are in this image?", or "Explain the mood of this scene."
4. For object detection queries, be specific about the format you need (e.g., bounding box coordinates).
5. Verify important information from its responses, especially for critical applications.
"""

class Visual_Generator_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o-mini"):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A tool that analyzes images and generates descriptions, captions, or answers questions about the image content. It uses a multimodal LLM to understand and describe visual information.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The query or instruction for what to analyze in the image (Examples: 'Describe this image in detail', 'What objects are in this image?', 'Explain the mood of this scene').",
                "image": "str - The path to the image file to analyze (can be absolute or relative to current directory).",
            },
            output_type="str - The generated description, caption, or answer based on the image and query",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Describe this image in detail", image="path/to/image.png")',
                    "description": "Generate a detailed description of the image."
                },
                {
                    "command": 'execution = tool.execute(query="Explain the mood of this scene.", image="path/to/image1.png")',
                    "description": "Generate a caption focusing on the mood using a specific query and image."
                },
                {
                    "command": 'execution = tool.execute(query="What are the main objects in this image?", image="path/to/image2.png")',
                    "description": "Identify and list the main objects present in the image."
                },
                {
                    "command": 'execution = tool.execute(query="Is the number of tiny objects that are behind the small metal jet less than the number of tiny things left of the tiny sedan?", image="path/to/image2.png")',
                    "description": "Answer a complex question step by step given the image."
                }
            ],

            user_metadata = {
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }

        )
        self.model_string = model_string  
        print(f"Initializing Visual Generator Tool with model: {self.model_string}")
        
        # Enable multimodal mode for image processing
        multimodal = True
        
        # NOTE: deterministic mode
        self.llm_engine = create_llm_engine(
            model_string=self.model_string, 
            is_multimodal=multimodal, 
            temperature=0.0, 
            top_p=1.0, 
            frequency_penalty=0.0, 
            presence_penalty=0.0
            )


    def execute(self, query, image):
        """
        Execute the visual generator tool to answer queries about an image.
        
        Args:
            query (str): The query or instruction for analyzing the image
            image (str): The path to the image file (absolute or relative)
            
        Returns:
            str: The generated description, caption, or answer based on the image and query
        """
        try:
            # Validate query parameter
            if not query:
                return "Error: No query provided. Please provide a query to analyze the image."
            
            # Validate image parameter
            if not image:
                return "Error: No image path provided. Please provide a valid image path."
            
            # Resolve the image path (handles both absolute and relative paths)
            image_path = os.path.abspath(os.path.expanduser(image))
            
            # Check if image file exists
            if not os.path.exists(image_path):
                return f"Error: Image file not found at path: {image_path}"
            
            # Read the image file as bytes
            with open(image_path, 'rb') as img_file:
                image_bytes = img_file.read()
            
            # Prepare input data as a list with query text and image bytes
            # The OpenAI engine expects: List[Union[str, bytes]]
            input_data = [query, image_bytes]
            
            # Call the multimodal LLM engine with the query and image
            response = self.llm_engine(input_data)
            
            # Return the response directly
            return response
            
        except Exception as e:
            return f"Error generating response for query: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd agentflow/tools/visual_generator
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # Example usage of the Visual_Generator_Tool
    tool = Visual_Generator_Tool(model_string="gpt-4o-mini") # NOTE: strong multimodal LLM for tool
    # tool = Visual_Generator_Tool(model_string="gemini-1.5-flash") # NOTE: alternative multimodal model

    # Get tool metadata
    metadata = tool.get_metadata()
    print("\n=== Tool Metadata ===")
    print(metadata)

    # Test with example images - use absolute paths
    examples = [
        {
            "query": "How many baseballs are in the image?",
            "image": os.path.join(script_dir, "examples/baseball.png")
        },
        {
            "query": "What is the name of the store in the image?",
            "image": os.path.join(script_dir, "examples/store.jpg")
        }
    ]

    for idx, example in enumerate(examples, 1):
        print(f"\n=== Example {idx} ===")
        print(f"Query: {example['query']}")
        print(f"Image: {example['image']}")
        
        try:
            execution = tool.execute(**example)
            print("\nGenerated Response:")
            print(execution)
            print("\n")
        except Exception as e:
            print(f"Execution failed: {e}")

    print("\nDone!")
