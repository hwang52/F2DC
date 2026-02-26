import os


def create_if_not_exists(path: str) -> None:
    """
    creates the folder if it does not exist
    param path = the path of the folder
    """
    if not os.path.exists(path):
        os.makedirs(path)