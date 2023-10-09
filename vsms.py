# Very Simple Memory Storage
WORKING_MEMORY = {}

def set(key, value):
    WORKING_MEMORY[key] = value
    return True

def get(key):
    if key in WORKING_MEMORY:
        return WORKING_MEMORY[key]
    return None

def delete(key):
    if key in WORKING_MEMORY:
        del WORKING_MEMORY[key]
        return True
    return False

def clear():
    WORKING_MEMORY.clear()
    return True

def keys():
    return WORKING_MEMORY.keys()

def exists(key):
    return key in WORKING_MEMORY