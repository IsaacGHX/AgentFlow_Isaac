import os
from agentflow.tools.base import BaseTool
from agentflow.engine.factory import create_llm_engine

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "OCR_Extractor_Tool"

LIMITATION = f"""
The {TOOL_NAME} may occasionally miss or misread text in images with poor quality, unusual fonts, or complex layouts.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Ensure the image path is valid and accessible.
2. Use high-quality, clear images for better OCR accuracy.
3. Images with good contrast and lighting will produce better results.
4. The tool will extract all visible text from the image.
5. Verify important information from its responses, especially for critical applications.
"""

class OCR_Extractor_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o-mini"):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A tool that extracts text from images using Optical Character Recognition (OCR). It uses a multimodal LLM to detect and extract all visible text content from the provided image.",
            tool_version="1.0.0",
            input_types={
                "image": "str - The path to the image file to extract text from.",
            },
            output_type="str - The extracted text from the image",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/document.png")',
                    "description": "Extract all text from a document image."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/receipt.jpg")',
                    "description": "Extract text from a receipt or invoice."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/sign.png")',
                    "description": "Extract text from a sign or poster."
                }
            ],

            user_metadata = {
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }

        )
        self.model_string = model_string  
        print(f"Initializing OCR Extractor Tool with model: {self.model_string}")
        
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


    def execute(self, image):
        """
        Execute the OCR extractor tool.
        
        Args:
            image (str): The path to the image file
            
        Returns:
            str: The extracted text from the image
        """
        try:
            # Validate image parameter
            if not image:
                return "Error: No image path provided. Please provide a valid image path."
            
            # Check if image file exists
            if not os.path.exists(image):
                return f"Error: Image file not found at path: {image}"
            
            # Read the image file as bytes
            with open(image, 'rb') as img_file:
                image_bytes = img_file.read()
            
            # Prepare the OCR extraction prompt
            ocr_prompt = "Extract all text from this image. Return only the text content without any additional explanation or commentary."
            
            # Prepare input data as a list with prompt text and image bytes
            # The OpenAI engine expects: List[Union[str, bytes]]
            input_data = [ocr_prompt, image_bytes]
            
            # Call the multimodal LLM engine with the OCR prompt and image
            response = self.llm_engine(input_data)
            return response
            
        except Exception as e:
            return f"Error extracting text from image: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd agentflow/tools/ocr_extractor
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # Example usage of the OCR_Extractor_Tool
    tool = OCR_Extractor_Tool(model_string="gpt-4o-mini") # NOTE: strong multimodal LLM for tool
    # tool = OCR_Extractor_Tool(model_string="gemini-1.5-flash") # NOTE: alternative multimodal model

    # Get tool metadata
    metadata = tool.get_metadata()
    print("\n=== Tool Metadata ===")
    print(metadata)

    # Test with example image if available
    example_image_path = os.path.join(script_dir, "examples", "store.jpg")
    
    if os.path.exists(example_image_path):
        print(f"\n=== Testing OCR extraction with image: {example_image_path} ===")
        
        try:
            execution = tool.execute(image=example_image_path)
            print("\nExtracted Text:")
            print(execution)
        except Exception as e: 
            print(f"Execution failed: {e}")
    else:
        print(f"\n=== Note: Example image not found at {example_image_path} ===")
        print("To test this tool, provide a valid image path:")
        print('tool.execute(image="/path/to/your/image.png")')

    print("\nDone!")
