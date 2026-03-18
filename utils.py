import asyncio
import base64
from typing import Literal
import logging
logger = logging.getLogger("Chains")

from ddgs import DDGS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from constants import MAX_MESSAGE_MEMORY

async def search_web(query: str, max_results: int=5, search_type: Literal['strong','weak']='weak') -> str:
    """
    Функція для пошуку інформації в мережі.
    :param query: Пошуковий запит
    :param max_results: Максимальна кількість паралельних запитів, при search_type='weak' не використовується
    :param search_type: Тип пошуку. strong - сторінка повертається повністю, weak - сторінки групуються
    :return:
    """
    logger.debug("search_web")
    logger.debug(f"Шукаю в інтернеті інформацію за запитом {query}")
    if search_type == 'strong':
        def _search():
            with DDGS() as ddgs:
                res = list(ddgs.text(query, max_results=max_results))
            return res

        results = await asyncio.to_thread(_search)
        full_text = ""
        for r in results:
            full_text += f"Заголовок: {r['title']}\n"
            full_text += f"Текст: {r['body']}\n"
            full_text += f"Посилання: {r['href']}\n\n"


        logger.debug(f"Ось що я знайшов: {full_text}")
        return full_text
    elif search_type == 'weak':
        search = DuckDuckGoSearchRun()
        result = search.invoke(query)

        logger.debug(f"Ось що я знайшов: {result}")
        return result
    else:
        raise ValueError(f"search_type {search_type} is not supported")

def trim_history(history: list, max_message_memory: int = MAX_MESSAGE_MEMORY) -> list:
    """
    Обрізає історію повідомлень до max_message_memory. Залишає важливі для контексту повідомлення: попереднє повідомлення від
    ШІ, попереднє повідомлення від користувача, попередній AiMessage + ToolMessage з результатами пошуку в мережі,
    попередні AiMessage + ToolMessage із зображенням.
    Залишає 1 місце для SystemMessage.
    :param history: список повідомлень
    :param max_message_memory: максимальна кількість повідомлень в історії
    :return: обрізана історія
    """

    # Шукаємо останнє людське повідомлення, щоб 100% залишити
    human_message = next((m for m in reversed(history) if isinstance(m, HumanMessage)), None)

    ai_message = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)

    # Знаходимо останню пару AIMessage з tool_call + ToolMessage для пошуку та зображення. В історю вони записані саме по-черзі
    vision_pair = []
    search_pair = []
    for i, msg in enumerate(history):
        if isinstance(msg, list):
            ai_msg, tool_msg = msg
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