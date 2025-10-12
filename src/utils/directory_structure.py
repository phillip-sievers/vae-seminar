import os


def get_repo_root():
    """
    Get the repository root directory (vae-seminar folder)
    """
    current_path = os.path.abspath(__file__)
    # Split the path and find 'vae-seminar', then rejoin up to that point
    parts = current_path.split(os.sep)
    vae_index = parts.index("vae-seminar")
    return os.sep.join(parts[: vae_index + 1])
