import json
import os

SETTINGS_PATH = "data/settings.json"

DEFAULT_SETTINGS = {
  "MAX_MESSAGE_MEMORY": 20,
  "DEFAULT_BOT_NAME": "Остап",
  "DEFAULT_TTS_VOICE": "uk-UA-OstapNeural",
  "MIN_QUESTION_LENGTH": 5,
  "MAX_QUESTION_LENGTH": 250,
  "VISION_MODEL_NAME": "qwen3.5:397b-cloud",
  "COMMAND_MODEL_NAME": "ministral-3:14b-cloud",
  "ROUTER_MODEL_NAME": "ministral-3:14b-cloud",
  "ANSWER_TYPE": "voice"
}

_settings = {}

def load_settings():
    global _settings
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(SETTINGS_PATH):
        _settings = DEFAULT_SETTINGS.copy()
        _save_all()
        return
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        _settings = json.load(f)

def _save_all():
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(_settings, f, ensure_ascii=False, indent=2)

def save(key: str, value):
    global _settings
    _settings[key] = value
    _save_all()

def get(key: str, default=None):
    return _settings.get(key, default)

def get_all() -> dict:
    return _settings.copy()