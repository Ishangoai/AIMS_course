import json

from pydantic import BaseModel, ValidationError


def parse_json_to_pydantic(json_string: str, pydantic_model: BaseModel):
    """
    Parses a JSON string and attempts to load it into a Pydantic model.

    Args:
        json_string: The string containing JSON data.
        pydantic_model: The Pydantic model class to which the JSON should be mapped.

    Returns:
        An instance of the pydantic_model if successful, otherwise None.
    """
    try:
        # Strip common delimiters like "```json" and "```"
        if json_string.strip().startswith("```json"):
            json_string = json_string.strip()[len("```json"):].strip()
        if json_string.strip().endswith("```"):
            json_string = json_string.strip()[:-len("```")].strip()

        data = json.loads(json_string)
        return pydantic_model.model_validate(data)  # Use model_validate for Pydantic v2
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print(f"Problematic JSON string: {json_string}")
        return None
    except ValidationError as e:
        print(f"Pydantic validation error: {e}")
        print(f"Problematic JSON data: {data}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Problematic JSON string: {json_string}")
        return None
