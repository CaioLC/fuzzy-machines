""" fuzzy logic functions """


def convert(my_name: str) -> None:
    """
    Print a line about converting a notebook.
    Args:
        my_name (str): person's name
    Returns:
        None
    """
    if not isinstance(my_name, str):
        raise TypeError(
            f'arg "my_name" expected type str. Received type {type(my_name)}'
        )
    print(f"I'll convert a notebook for you some day, {my_name}.")

def a_second_function(a_number: int) -> int:
    """template for a second function

    Args:
        a_number (int): a number

    Returns:
        int: the same number
    """
    return a_number