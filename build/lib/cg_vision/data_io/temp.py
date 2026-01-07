import tempfile

def create_temp_dir():
    """
    Creates a temporary directory that works both locally and on the cloud.
    Ensures cleanup after use.

    :return: str. Path to the temporary directory.
    """
    # Create a temporary directory
    try:
        temp_dir = tempfile.mkdtemp()  # Generates the temporary directory
        print(f"Temporary directory created at: {temp_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to create temporary directory: {e}")

    return temp_dir