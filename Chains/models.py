import os
from typing import Literal, Optional

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from groq import Groq

import settings_manager as s

whisper_model = None
vision_model = None
router_model = None
command_model = None


def _make_openrouter_llm(model_name: str, temperature: float = 0, **kwargs) -> ChatOpenAI:
    """Фабрика для створення ChatOpenAI, налаштованого на OpenRouter."""
    return ChatOpenAI(

        model=model_name,
        temperature=temperature,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://eye-help-you.fly.dev",
            "X-Title": "EyeHelpYou_tg_bot",
        },
        **kwargs
    )

class Router(BaseModel):
    """
    task - тип завдання "answer" або "command"
    search_query - пошуковий запит для допомоги
    is_vision_needed - чи потрібно попросити користувача надати зображення
    """
    task: Literal["answer","command"]
    search_query: Optional[str] = None
    is_vision_needed: Optional[bool] = None

class Command(BaseModel):
    """
    command - команда
    command_argument - аргумент команди

    Достпні команди: set_user_name, set_bot_name
    """
    command: Literal["set_user_name","set_bot_name"]
    command_argument: str

def load_models():

    global whisper_model
    global vision_model
    global router_model
    global command_model

    whisper_model = Groq(api_key=os.getenv("GROQ_API_KEY"))

    vision_model = _make_openrouter_llm(s.get("VISION_MODEL_NAME"),reasoning={"effort": "none"},)

    router_model = _make_openrouter_llm(
        model_name=s.get("ROUTER_MODEL_NAME"),
        model_kwargs={"response_format": {"type": "json_object"},},
        reasoning={"effort": "low"}
    )
    router_model = router_model.with_structured_output(Router, method="json_mode")

    command_model = _make_openrouter_llm(
        model_name=s.get("COMMAND_MODEL_NAME"),
        model_kwargs={"response_format": {"type": "json_object"}},
        reasoning={"effort": "low"}
    )
    command_model = command_model.with_structured_output(Command, method="json_mode")
