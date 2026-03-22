from typing import Literal, Optional

from pydantic import BaseModel

import settings_manager as s

whisper_model = None
vision_model = None
router_model = None
command_model = None

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
    import whisper
    from langchain_ollama import ChatOllama


    global whisper_model
    global vision_model
    global router_model
    global command_model

    vision_model_name = s.get("VISION_MODEL_NAME")

    vision_model = ChatOllama(model=vision_model_name, temperature=0, format="json", reasoning=False)

    whisper_model = whisper.load_model("small")

    router_model = ChatOllama(model=s.get("ROUTER_MODEL_NAME"), temperature=0, format="json", reasoning=False)
    router_model = router_model.with_structured_output(Router)

    command_model = ChatOllama(model=s.get("COMMAND_MODEL_NAME"), temperature=0, format="json", reasoning=False)
    command_model = command_model.with_structured_output(Command)
