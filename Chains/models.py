import os
from typing import Literal, Optional

from groq import Groq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

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
    command: Literal["set_user_name","set_bot_name","set_answer_type"]
    command_argument: str

def load_models():

    global whisper_model
    global vision_model
    global router_model
    global command_model

    whisper_model = Groq(api_key=os.getenv("GROQ_API_KEY"))

    vision_model = _make_openrouter_llm(s.get("VISION_MODEL_NAME"),
                                        reasoning={"effort": "none"},
                                        max_tokens=s.get("MAX_ANSWER_LENGTH", 250),
                                        )

    primary_router = _make_openrouter_llm(
        model_name=s.get("ROUTER_MODEL_NAME"),
        model_kwargs={"response_format": {"type": "json_object"}, },
        reasoning={"effort": "none"}
    )
    fallback_router1 = _make_openrouter_llm(
        model_name=s.get("ROUTER_FALLBACK1"),
        model_kwargs={"response_format": {"type": "json_object"}, },
        reasoning={"effort": "none"}
    )
    fallback_router2 = _make_openrouter_llm(
        model_name=s.get("ROUTER_FALLBACK2"),
        model_kwargs={"response_format": {"type": "json_object"}, },
        reasoning={"effort": "none"}
    )
    router_llm = primary_router.with_fallbacks([fallback_router1, fallback_router2])
    router_model = router_llm.with_structured_output(Router, method="json_mode")


    primary_command = _make_openrouter_llm(
        model_name=s.get("COMMAND_MODEL_NAME"),
        model_kwargs={"response_format": {"type": "json_object"}, },
        reasoning={"effort": "none"}
    )
    fallback_command1 = _make_openrouter_llm(
        model_name=s.get("COMMAND_FALLBACK1"),
        model_kwargs={"response_format": {"type": "json_object"}, },
        reasoning={"effort": "none"}
    )
    fallback_command2 = _make_openrouter_llm(
        model_name=s.get("COMMAND_FALLBACK2"),
        model_kwargs={"response_format": {"type": "json_object"}, },
        reasoning={"effort": "none"}
    )
    command_llm = primary_command.with_fallbacks([fallback_command1, fallback_command2])
    command_model = command_llm.with_structured_output(Command, method="json_mode")
