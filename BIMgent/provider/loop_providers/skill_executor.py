from BIMgent.provider.ui_controller import  MouseController
import re
from collections.abc import Iterable

def flatten_actions(actions):
    """Recursively flattens arbitrarily nested lists of actions."""
    for item in actions:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten_actions(item)
        else:
            yield item


def execute_actions(actions, controller):
    """
    Executes a sequence of actions on the given controller.

    :param actions: List (possibly nested) of action strings to execute.
    :param controller: An instance of MouseController.
    """
    flat_actions = list(flatten_actions(actions))  # Flatten any shape of [[[]]]
    
    for action in flat_actions:
        try:
            # Match function calls like move(x=10, y=20) or click(100, 200)
            matches = re.findall(r"(\w+)\(([^)]*)\)", action)
            
            for method_name, param_str in matches:
                args = []
                kwargs = {}

                # Parse parameters only if exists
                if param_str.strip():
                    params = [p.strip() for p in param_str.split(",") if p.strip()]
                    for p in params:
                        if "=" in p:  # keyword argument
                            key, value = p.split("=", 1)
                            kwargs[key.strip()] = eval(value.strip())
                        else:
                            args.append(eval(p))

                # Call the method dynamically
                method = getattr(controller, method_name, None)
                if not method:
                    print(f"[Warning] Unknown controller method: {method_name}")
                    continue

                method(*args, **kwargs)

        except Exception as e:
            print(f"Error executing action '{action}': {e}")
