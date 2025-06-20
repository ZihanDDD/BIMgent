import os
import re
import json
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple, Optional
from typing import Tuple, List, Dict
from openai import OpenAI

from conf.config import Config

load_dotenv()
config = Config()
api_key = os.getenv('OA_OPENAI_KEY')

class OpenAIProvider():

    def __init__(self, model):

        self.client = OpenAI(api_key=api_key)
        self.llm_model = model


    def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = config.temperature,
        seed: int = config.seed,
        max_tokens: int = 20000,
    ) -> Tuple[str, Dict[str, int]]:
        """Create a chat completion using the OpenAI API

        Supports both GPT-4 and GPT-4V).

        Example Usage:
        image_path = "path_to_your_image.jpg"
        base64_image = encode_image(image_path)
        response, info = self.create_completion(
            model="gpt-4-vision-preview",
            messages=[
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": "What’s in this image?"
                  },
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                  }
                ]
              }
            ],
        )
        """

        if model is None:
            model = self.llm_model

        def _generate_response_with_retry(
            messages: List[Dict[str, str]],
            model: str,
            temperature: float,
            seed: int = None,
            max_tokens: int = 20000,
        ) -> Tuple[str, Dict[str, int]]:

            """Send a request to the OpenAI API."""

            
            if model == 'o4-mini':
                response = self.client.chat.completions.create(model=model,
                messages=messages)
            
            else:
                response = self.client.chat.completions.create(model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_completion_tokens=max_tokens,)

            if response is None:
                print("Failed to get a response from OpenAI. Try again.")

            message = response.choices[0].message.content

            info = {
                "prompt_tokens" : response.usage.prompt_tokens,
                "completion_tokens" : response.usage.completion_tokens,
                "total_tokens" : response.usage.total_tokens,
                "system_fingerprint" : response.system_fingerprint,
            }

            print(f'Response received from {model}.')


            return message

        return _generate_response_with_retry(
            messages,
            model,
            temperature,
            seed,
            max_tokens,
        )
    def assemble_prompt(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Assemble a prompt with two parts:
        
        1. System message: The first paragraph of the template.
        2. User message: All subsequent paragraphs merged together after processing placeholders.
        """
        if params is None:
            params = {}

        paragraphs = re.split(r"\n\n+", template_str or "")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            raise ValueError("Template string is empty or only contains whitespace.")

        # Create the system message from the first paragraph
        system_content = paragraphs[0]
        system_message = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_content
                }
            ]
        }

        # Process all remaining paragraphs for the user message
        user_paragraphs = paragraphs[1:]
        
        def process_paragraphs(paragraph_list: List[str]) -> str:
            """Replace placeholders in each paragraph and join them."""
            processed = []
            placeholder_pattern = re.compile(r"<\$[^\$]+\$>")
            for paragraph in paragraph_list:
                # Replace all occurrences of placeholders in the paragraph
                for placeholder in re.findall(placeholder_pattern, paragraph):
                    placeholder_name = placeholder.replace("<$", "").replace("$>", "")
                    replacement = params.get(placeholder_name, "")
                    
                    # JSON-serialize any non-string placeholders, including dict
                    if isinstance(replacement, (list, bool, int, float, dict)):
                        replacement = json.dumps(replacement)
                    elif not isinstance(replacement, str):
                        raise ValueError(
                            f"Unexpected input type for placeholder '{placeholder_name}': {type(replacement)}"
                        )
                    
                    paragraph = paragraph.replace(placeholder, replacement)
                processed.append(paragraph)
            return "\n\n".join(processed)

        user_content = process_paragraphs(user_paragraphs)
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_content
                }
            ]
        }

        return [system_message, user_message]