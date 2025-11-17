import os
from agentflow.tools.base import BaseTool
from agentflow.engine.factory import create_llm_engine

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Image_Captioner_Tool"

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

class Image_Captioner_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o-mini"):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A tool that analyzes images and generates descriptions, captions, or answers questions about the image content. It uses a multimodal LLM to understand and describe visual information.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The query or instruction for what to analyze in the image (Examples: 'Describe this image in detail', 'What objects are in this image?', 'Explain the mood of this scene').",
                "image": "str - The path to the image file to analyze.",
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
        print(f"Initializing Image Captioner Tool with model: {self.model_string}")
        
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
        Execute the image captioner tool.
        
        Args:
            query (str): The query or instruction for analyzing the image
            image (str): The path to the image file
            
        Returns:
            str: The generated description, caption, or answer based on the image
        """
        try:
            # Validate image parameter
            if not image:
                return "Error: No image path provided. Please provide a valid image path."
            
            # Check if image file exists
            if not os.path.exists(image):
                return f"Error: Image file not found at path: {image}"
            
            # Prepare input data with both query and image for multimodal LLM
            input_data = {
                "text": query,
                "image": image
            }
            
            # Call the multimodal LLM engine with the query and image
            response = self.llm_engine(input_data)
            return response
            
        except Exception as e:
            return f"Error generating image description: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd agentflow/tools/image_captioner
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # Example usage of the Image_Captioner_Tool
    tool = Image_Captioner_Tool(model_string="gpt-4o-mini") # NOTE: strong multimodal LLM for tool
    # tool = Image_Captioner_Tool(model_string="gemini-1.5-flash") # NOTE: alternative multimodal model

    # Get tool metadata
    metadata = tool.get_metadata()
    print("\n=== Tool Metadata ===")
    print(metadata)

    # Test with example image if available
    example_image_path = os.path.join(script_dir, "examples", "baseball.png")
    
    if os.path.exists(example_image_path):
        query = "Describe this image in detail."
        print(f"\n=== Testing with image: {example_image_path} ===")
        print(f"Query: {query}")
        
        try:
            execution = tool.execute(query=query, image=example_image_path)
            print("\nGenerated Image Description:")
            print(execution)
        except Exception as e: 
            print(f"Execution failed: {e}")
    else:
        print(f"\n=== Note: Example image not found at {example_image_path} ===")
        print("To test this tool, provide a valid image path:")
        print('tool.execute(query="Describe this image", image="/path/to/your/image.png")')

    print("\nDone!")
