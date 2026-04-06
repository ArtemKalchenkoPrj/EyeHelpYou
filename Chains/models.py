import os
from typing import Literal, Optional

from groq import Groq
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from Chains.command_chain import set_user_name, set_bot_name, set_answer_type
import settings_manager as s

whisper_model = None
vision_model = None
router_model = None
command_model = None


def _validate_output(response):
    content = response.content

    if not content or (isinstance(content, str) and content.strip() == ""):
        raise ValueError("Empty response from model")
    return response


def _make_openrouter_llm(model_name: str, temperature: float = 0, base_url="https://openrouter.ai/api/v1", **kwargs) -> ChatOpenAI:
    """Фабрика для створення ChatOpenAI, налаштованого на OpenRouter."""

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        api_key=os.getenv("OPENROUTER_API_KEY"),
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

def load_models():

    global whisper_model
    global vision_model
    global router_model
    global command_model

    whisper_model = Groq(api_key=os.getenv("GROQ_API_KEY"))


    primary_vision = _make_openrouter_llm(s.get("VISION_MODEL_NAME"),
                                            reasoning={"effort": "none"},
                                            max_tokens=s.get("MAX_ANSWER_LENGTH", 250),
                                            )
    fallback_vision1 = _make_openrouter_llm(s.get("VISION_FALLBACK1"),
                                            reasoning={"effort": "none"},
                                            max_tokens=s.get("MAX_ANSWER_LENGTH", 250),
                                            )
    fallback_vision2 = _make_openrouter_llm(s.get("VISION_FALLBACK2"),
                                            reasoning={"effort": "none"},
                                            max_tokens=s.get("MAX_ANSWER_LENGTH", 250),
                                            )

    primary_chain = primary_vision | RunnableLambda(_validate_output)
    fallback_vision1_chain = fallback_vision1 | RunnableLambda(_validate_output)
    fallback_vision2_chain = fallback_vision2 | RunnableLambda(_validate_output)

    vision_model = primary_chain.with_fallbacks([fallback_vision1_chain, fallback_vision2_chain])


    primary_router = _make_openrouter_llm(
        temperature=0,
        model_name=s.get("ROUTER_MODEL_NAME"),
        reasoning={"effort": "none"}
    )
    fallback_router1 = _make_openrouter_llm(
        temperature=0,
        model_name=s.get("ROUTER_FALLBACK1"),
        reasoning={"effort": "none"}
    )
    fallback_router2 = _make_openrouter_llm(
        temperature=0,
        model_name=s.get("ROUTER_FALLBACK2"),
        reasoning={"effort": "none"}
    )
    router_llm = primary_router.with_fallbacks([fallback_router1, fallback_router2])
    router_model = router_llm.with_structured_output(Router, method="json_mode")


    primary_command = _make_openrouter_llm(
        temperature=0,
        model_name=s.get("COMMAND_MODEL_NAME"),
        reasoning={"effort": "none"}
    )
    fallback_command1 = _make_openrouter_llm(
        temperature=0,
        model_name=s.get("COMMAND_FALLBACK1"),
        reasoning={"effort": "none"}
    )
    fallback_command2 = _make_openrouter_llm(
        temperature=0,
        model_name=s.get("COMMAND_FALLBACK2"),
        reasoning={"effort": "none"}
    )
    command_llm = primary_command.with_fallbacks([fallback_command1, fallback_command2])
    command_tools = [set_bot_name, set_user_name, set_answer_type]
    command_model = command_llm.bind_tools(command_tools)