"""Voice-only preference. Never changes obstacle detection or navigation goals."""
import json

def read_proximity_voice(path):
    try:
        value=json.loads(path.read_text('utf-8')).get('proximity_voice',False)
        return value if isinstance(value,bool) else False
    except (OSError,ValueError,AttributeError):
        return False
