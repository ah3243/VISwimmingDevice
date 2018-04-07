import os

def isPi():
    # Check if code is running on arm based pi
    out = os.uname()[4].startswith("arm")
    return out
