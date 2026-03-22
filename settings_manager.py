import json

_settings = {}

def load_settings():
    global _settings
    with open("settings.json",encoding="utf-8") as f:
        _settings = json.load(f)

def save(key: str, value):
    global _settings
    _settings[key] = value
    with open("settings.json", "w", encoding="utf-8") as f:
        json.dump(_settings, f, ensure_ascii=False, indent=2)

def get(key: str, default=None):
    return _settings.get(key, default)

def get_all() -> dict:
    return _settings.copy()