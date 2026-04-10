import json
import os

SETTINGS_PATH = "data/settings.json"

DEFAULT_SETTINGS = {
  "ANSWER_TYPE": "text",
  "MAX_ANSWER_LENGTH": 250,
  "MAX_MESSAGE_MEMORY": 20,
  "DEFAULT_BOT_NAME": "Остап",
  "DEFAULT_TTS_VOICE": "uk-UA-OstapNeural",
  "MIN_QUESTION_LENGTH": 5,
  "MAX_QUESTION_LENGTH": 250,

  "VISION_MODEL_NAME": "google/gemini-3-flash-preview",
  "VISION_FALLBACK1": "qwen/qwen3.5-397b-a17b",
  "VISION_FALLBACK2": "xiaomi/mimo-v2-omni",

  "ROUTER_MODEL_NAME": "xiaomi/mimo-v2-flash",
  "ROUTER_FALLBACK1": "z-ai/glm-4.7-flash",
  "ROUTER_FALLBACK2": "qwen/qwen3-235b-a22b-2507",

  "COMMAND_MODEL_NAME": "xiaomi/mimo-v2-flash",
  "COMMAND_FALLBACK1": "z-ai/glm-4.7-flash",
  "COMMAND_FALLBACK2": "qwen/qwen3-235b-a22b-2507",

    "CALCULATOR_MODEL_NAME": "openai/gpt-oss-120b",
    "CALCULATOR_FALLBACK1": "z-ai/glm-4.7-flash",
    "CALCULATOR_FALLBACK2": "minimax/minimax-m2.5"
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