from BIMgent.provider.ui_controller import  MouseController
import re

def execute_actions(actions: list, controller: MouseController):
    """
    Executes a sequence of actions on the given controller.

    :param actions: List of action strings to execute.
    :param controller: An instance of MouseController.
    """
    for action in actions:
        try:
            #print(f"Executing action: {action}")
            # Use regex to match method calls with optional parameters
            matches = re.findall(r"(\w+)\(([^)]*)\)", action)
            for method_name, param_str in matches:
                # Parse parameters into positional and keyword arguments
                args = []
                kwargs = {}
                if param_str.strip():  # Check if there are any parameters
                    for param in param_str.split(","):
                        key_value = param.split("=")
                        if len(key_value) == 2:  # Named parameter
                            key, value = key_value
                            kwargs[key.strip()] = eval(value.strip())
                        else:  # Positional parameter
                            args.append(eval(key_value[0].strip()))
                # Dynamically call the method with positional and keyword arguments
                getattr(controller, method_name)(*args, **kwargs)
        except Exception as e:
            print(f"Error executing action '{action}': {e}")
