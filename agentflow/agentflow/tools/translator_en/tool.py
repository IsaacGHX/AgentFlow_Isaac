import os
from agentflow.tools.base import BaseTool
from agentflow.engine.factory import create_llm_engine

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Translator_EN_Tool"

LIMITATION = f"""
The {TOOL_NAME} may occasionally:
1. Misinterpret context or nuances in complex texts
2. Struggle with idiomatic expressions or cultural references
3. Have difficulty with very technical or domain-specific terminology
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Provide clear text in any language that needs to be translated to English
2. Break down very long texts into smaller chunks for more accurate translation
3. Verify translations of critical or technical content
4. The tool will preserve the original meaning and tone as much as possible
"""

class Translator_EN_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o-mini"):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A translation tool that translates text from any language to English. It preserves the original meaning, tone, and context while providing accurate English translations.",
            tool_version="1.0.0",
            input_types={
                "text": "str - The text in any language that needs to be translated to English.",
            },
            output_type="str - The translated English text",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(text="Bonjour, comment allez-vous?")',
                    "description": "Translate French text to English."
                },
                {
                    "command": 'execution = tool.execute(text="こんにちは、元気ですか？")',
                    "description": "Translate Japanese text to English."
                },
                {
                    "command": 'execution = tool.execute(text="Esta es una oración técnica sobre inteligencia artificial.")',
                    "description": "Translate Spanish technical text to English."
                }
            ],

            user_metadata = {
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }

        )
        self.model_string = model_string  
        print(f"Initializing Translator EN Tool with model: {self.model_string}")
        multimodal = False
        
        # NOTE: deterministic mode for consistent translations
        self.llm_engine = create_llm_engine(
            model_string=self.model_string, 
            is_multimodal=multimodal, 
            temperature=0.0, 
            top_p=1.0, 
            frequency_penalty=0.0, 
            presence_penalty=0.0
            )


    def execute(self, text):
        """
        Translate non-English text to English.
        
        Args:
            text: The text in any language to translate to English
            
        Returns:
            str: The translated English text
        """
        try:
            # Build the translation prompt
            prompt = f"""You are a professional translator. Translate the following text to English.

Text to translate:
{text}

Provide ONLY the English translation, without any explanations or additional text."""
            
            response = self.llm_engine(prompt)
            return response
        except Exception as e:
            return f"Error translating text: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd agentflow/tools/translator_en
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # Example usage of the Translator_EN_Tool
    tool = Translator_EN_Tool(model_string="gpt-4o-mini") # NOTE: strong LLM for tool
    # tool = Translator_EN_Tool(model_string="gemini-1.5-flash") # NOTE: alternative model
    # tool = Translator_EN_Tool(model_string="dashscope") # NOTE: Qwen2.5-7B model for tool


    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Test translations from different languages
    test_texts = [
        ("Bonjour, comment allez-vous?", "French"),
        ("こんにちは、元気ですか？", "Japanese"),
        ("Esta es una oración sobre inteligencia artificial.", "Spanish"),
        ("Guten Tag, wie geht es Ihnen?", "German"),
    ]

    for text, lang in test_texts:
        print(f"\n{'='*60}")
        print(f"Testing {lang}:")
        print(f"Original: {text}")
        
        try:
            translation = tool.execute(text=text)
            print(f"Translation: {translation}")
        except Exception as e: 
            print(f"Execution failed: {e}")

    print("\n" + "="*60)
    print("Done!")
