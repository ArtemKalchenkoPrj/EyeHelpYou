import asyncio
import logging
from _contextvars import ContextVar
from urllib.parse import urlparse

from ddgs import DDGS
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable

import settings_manager as s

logger = logging.getLogger("Chains")


current_answer_type: ContextVar[str|None] = ContextVar('answer_type', default=None)


def trim_history(history: list, max_message_memory: int = None) -> list:
    """
    Обрізає історію повідомлень до max_message_memory. Залишає важливі для контексту повідомлення: попереднє повідомлення від
    ШІ, попереднє повідомлення від користувача, попередній AiMessage + ToolMessage з результатами пошуку в мережі,
    попередні AiMessage + ToolMessage із зображенням.
    Залишає 1 місце для SystemMessage.
    :param history: список повідомлень
    :param max_message_memory: максимальна кількість повідомлень в історії
    :return: обрізана історія
    """
    if max_message_memory is None:
        max_message_memory = s.get("MAX_MESSAGE_MEMORY")

    # Шукаємо останнє людське повідомлення, щоб 100% залишити
    human_message = next((m for m in reversed(history) if isinstance(m, HumanMessage)), None)

    ai_message = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)

    # Знаходимо останню пару AIMessage з tool_call + ToolMessage для пошуку та зображення. В історію вони записані саме по-черзі
    vision_pair = []
    search_pair = []
    for i, msg in enumerate(history):
        if isinstance(msg, list):
            ai_msg, *tool_msgs = msg
            if any(tc["name"] == "search" for tc in ai_msg.tool_calls):
                search_pair = msg
            elif any(tc["name"] == "vision" for tc in ai_msg.tool_calls):
                vision_pair = msg

    # Обов'язкові повідомлення які завжди залишаються
    required = []
    if human_message:
        required.append(human_message)
    if ai_message:
        required.append(ai_message)

    if search_pair:
        required.append(search_pair)
    if vision_pair:
        required.append(vision_pair)


    optional = [m for m in history if m not in required]

    max_history = max_message_memory - len(required)
    if max_history <= 0:
        optional = []
    elif len(optional) > max_history:
        optional = optional[-max_history:]

    result = optional + required

    logger.debug(f"Історія: {[msg.content[:20] for msg in result if isinstance(msg, AIMessage)]}")

    # Сортуємо за оригінальним індексом в history
    return sorted(result, key=lambda m: history.index(m))


def unpack_history(history: list) -> list:
    result = []
    for msg in history:
        if isinstance(msg, list):
            result.extend(msg)
        else:
            result.append(msg)
    return result