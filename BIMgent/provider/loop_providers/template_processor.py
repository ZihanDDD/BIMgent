import os 
import re
import json
from termcolor import colored
from typing import Dict, Any, List, Tuple, Optional
from BIMgent.utils.json_utils import parse_semi_formatted_text
from BIMgent.provider.screenshots_processor import encode_data_to_base64_path
from conf.config import Config

config = Config()



def extract_keys_from_template(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Template file {path} does not exist")

    with open(path, "r", encoding="utf-8") as fd:
        file = fd.read()

    parse_input_keys = re.findall(r'<\$(.*?)\$>', file)
    input_keys = [key.strip() for key in parse_input_keys]
    #print(f"Recommended input parameters: {input_keys}")

    start_output_line_index = file.find('You should respond')
    output_text = file[start_output_line_index + 1:]
    output = parse_semi_formatted_text(output_text)
    output_keys = list(output.keys())
    #print(f"Recommended output parameters: {output_keys}")

    return file, input_keys, output_keys


def check_input_keys(params: Dict[str, Any], input_keys):
    for key in input_keys:
        if key not in params:
            print(colored(f"Key {key} is not in the input parameters", "red"))
            params[key] = None

def check_output_keys(response: Dict[str, Any], output_keys):
    for key in output_keys:
        if key not in response:
            print(colored(f"Key {key} is not in the response", "red"))
            response[key] = None


def assemble_prompt(template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Assemble a prompt with two parts:
    
    1. System message: The first paragraph of the template.
    2. User message: All subsequent paragraphs merged together after processing placeholders.
    """
    if params is None:
        params = {}

    # Split the template into paragraphs using two or more newlines
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
                if isinstance(replacement, (list, bool, int, float)):
                    replacement = json.dumps(replacement)
                elif not isinstance(replacement, str):
                    raise ValueError(f"Unexpected input type for placeholder '{placeholder_name}': {type(replacement)}")
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

def _is_line_key_candidate(line: str) -> Tuple[bool, Optional[str]]:

    result = False
    likely_key = None

    if line.endswith(':'):

        # Cannot have other previous punctuation, except if it's a numbered bullet list item
        num_idx = is_numbered_bullet_list_item(line)

        post_num_idx = 0
        if num_idx > -1:
            post_num_idx = num_idx

        likely_key = line[post_num_idx:-1].strip()
        result = not contains_punctuation(likely_key)

    return result, likely_key

def is_numbered_bullet_list_item(s: str) -> int:
    # Regular expression to match the series of numbers followed by a dot
    pattern = r'^\d+\.'

    match = re.match(pattern, s)
    if match:
        # If there's a match, return the index right after the matched pattern
        return match.end()
    else:
        # If no match, return an invalid value
        return -1
    

def contains_punctuation(s: str) -> bool:
    # Pattern to match punctuation characters
    punctuation_pattern = r'[^\w\s_]'

    # Search for punctuation characters in the string
    return re.search(punctuation_pattern, s)


### Parses the semi-formatted text from model response
def parse_semi_formatted_text(text):

    lines = text.split('\n')

    lines = [line.rstrip() for line in lines if line.rstrip()]
    result_dict = {}
    current_key = None
    current_value = []
    parsed_data = []
    in_code_flag = False

    for line in lines:

        line = line.replace("**", "").replace("###", "").replace("##", "") # Remove unnecessary in Markdown formatting

        is_key, key_candidate = _is_line_key_candidate(line)

        # Check if the line indicates a new key
        if  is_key and in_code_flag == False:

            # If there's a previous key, process its values
            if current_key and current_key == 'action_guidance':
                result_dict[current_key] = parsed_data
            elif current_key:
                result_dict[current_key] = '\n'.join(current_value).strip()

            try:
                current_key = key_candidate.replace(" ", "_").lower()
            except Exception as e:
                print(f"Response is not in the correct format: {e}\nReceived text was: {text}")
                raise

            current_value = []
            parsed_data = []
        else:
            if current_key == 'action_guidance':
                in_code_flag = True
                if line.strip() == '```':
                    if current_value:  # Process previous code block and description
                        entry = {"code": '\n'.join(current_value[1:])}
                        parsed_data.append(entry)
                        current_value = []
                    in_code_flag = False
                else:
                    current_value.append(line)
                    if line.strip().lower() == 'null':
                        in_code_flag = False
            else:
                in_code_flag = False
                line = line.strip()
                current_value.append(line)

    # Process the last key
    if current_key == 'action_guidance':
        if current_value:  # Process the last code block and description
            entry = {"code": '\n'.join(current_value[:-1]).strip()}
            parsed_data.append(entry)
        result_dict[current_key] = parsed_data
    else:
        result_dict[current_key] = '\n'.join(current_value).strip()

    if "actions" in result_dict:
        actions = result_dict["actions"]
        actions = actions.replace('```python', '').replace('```', '')
        actions = actions.split('\n')

        actions = [action for action in actions if action]

        actions = [action.split('#')[0] if "#" in action else action for action in actions]

        result_dict["actions"] = actions

    if "success" in result_dict:
        result_dict["success"] = result_dict["success"].lower() == "true"

    return result_dict
