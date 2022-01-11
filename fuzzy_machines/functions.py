def convert(my_name):
    """
    Print a line about converting a notebook.
    Args:
        my_name (str): person's name
    Returns:
        None
    """
    if not isinstance(my_name, str):
        raise TypeError(f'arg "my_name" expected type str. Received type {type(my_name)}')
    print(f"I'll convert a notebook for you some day, {my_name}.")