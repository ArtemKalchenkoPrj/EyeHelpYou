import os
from typing import Literal, Optional

from groq import Groq
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import settings_manager as s

whisper_model: Groq = None
vision_model: ChatOpenAI = None
router_model: ChatOpenAI = None
command_model: ChatOpenAI = None
calculator_model: ChatOpenAI = None

reasoning_needed_models = ['openai/gpt-oss-120b']

def _make_openrouter_vision_llm(postfix:str, **kwargs):
    """Фабрика для створення ChatOpenAI, налаштованого на OpenRouter. для vision моделей"""
    from Chains.processor_chain import search, run_calculator

    def _validate_output(ai_msg: AIMessage) -> AIMessage:
        if isinstance(ai_msg.content, list):
            # витягуємо текст з блоків
            text = " ".join(
                block.get("text", "")
                for block in ai_msg.content
                if isinstance(block, dict)
            )
            ai_msg.content = text
        return ai_msg

    vision_tools = [search, run_calculator]

    return ChatOpenAI(
        model=s.get("VISION_"+postfix),
        max_tokens=s.get("MAX_ANSWER_LENGTH", 250),
        reasoning={"effort": "low"},
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://eye-help-you.fly.dev",
            "X-Title": "EyeHelpYou_tg_bot",
        },
        **kwargs
    ).bind_tools(vision_tools) | RunnableLambda(_validate_output)

def _make_openrouter_command_llm(postfix:str, **kwargs):
    """Фабрика для створення ChatOpenAI, налаштованого на OpenRouter. для command моделей"""

    model_name = s.get("COMMAND_" + postfix)
    if model_name in reasoning_needed_models:
        reasoning = {"effort": "low"}
    else:
        reasoning = {"effort": "none"}
    return ChatOpenAI(
        reasoning=reasoning,
        model=model_name,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://eye-help-you.fly.dev",
            "X-Title": "EyeHelpYou_tg_bot",
        },
        **kwargs
    )

def _make_openrouter_router_llm(postfix:str, **kwargs):
    """Фабрика для створення ChatOpenAI, налаштованого на OpenRouter. для router моделей"""
    model_name = s.get("ROUTER_" + postfix)
    if model_name in reasoning_needed_models:
        reasoning = {"effort": "low"}
    else:
        reasoning = {"effort": "none"}
    return ChatOpenAI(
        reasoning=reasoning,
        model=model_name,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://eye-help-you.fly.dev",
            "X-Title": "EyeHelpYou_tg_bot",
        },
        **kwargs
    )

def _make_openrouter_calculator_llm(postfix:str, **kwargs):
    """Фабрика для створення ChatOpenAI, налаштованого на OpenRouter. для calculator моделей"""
    model_name = s.get("CALCULATOR_" + postfix)
    if model_name in reasoning_needed_models:
        reasoning = {"effort": "low"}
    else:
        reasoning = {"effort": "none"}
    return ChatOpenAI(
        reasoning=reasoning,
        model=model_name,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
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
    is_vision_needed: Optional[bool] = None

def load_models():
    from Chains.command_chain import set_user_name, set_bot_name, set_answer_type

    global whisper_model
    global vision_model
    global router_model
    global command_model
    global calculator_model

    whisper_model = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # vision model init
    primary_vision = _make_openrouter_vision_llm("MODEL_NAME")
    fallback_vision1 = _make_openrouter_vision_llm("FALLBACK1")
    fallback_vision2 = _make_openrouter_vision_llm("FALLBACK2")
    vision_model = primary_vision.with_fallbacks([fallback_vision1, fallback_vision2])

    # router model init
    primary_router = _make_openrouter_router_llm("MODEL_NAME")
    fallback_router1 = _make_openrouter_router_llm("FALLBACK1")
    fallback_router2 = _make_openrouter_router_llm("FALLBACK2")
    router_llm = primary_router.with_fallbacks([fallback_router1, fallback_router2])
    router_model = router_llm.with_structured_output(Router, method="json_mode")

    # command model init
    primary_command = _make_openrouter_command_llm("MODEL_NAME")
    fallback_command1 = _make_openrouter_command_llm("FALLBACK1")
    fallback_command2 = _make_openrouter_command_llm("FALLBACK2")
    command_llm = primary_command.with_fallbacks([fallback_command1, fallback_command2])
    command_tools = [set_bot_name, set_user_name, set_answer_type]
    command_model = command_llm.bind_tools(command_tools)

    # calculator model init
    primary_calculator = _make_openrouter_calculator_llm("MODEL_NAME")
    fallback_calculator1 = _make_openrouter_calculator_llm("FALLBACK1")
    fallback_calculator2 = _make_openrouter_calculator_llm("FALLBACK2")
    calculator_model = primary_calculator.with_fallbacks([fallback_calculator1, fallback_calculator2])