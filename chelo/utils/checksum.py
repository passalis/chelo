import hashlib


def verify_checksum(file_path: str, checksum: str) -> bool:
    """
    Verify the checksum of a file.
    :param file_path: Path to the file.
    :param checksum: Expected checksum (MD5 or SHA256).
    :return: True if the checksum matches, False otherwise.
    """
    hash_func = hashlib.sha256 if len(checksum) == 64 else hashlib.md5
    with open(file_path, "rb") as file:
        file_hash: str = hash_func(file.read()).hexdigest()
    return file_hash == checksum
