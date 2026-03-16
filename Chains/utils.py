import asyncio
import base64
from typing import Literal
import logging
logger = logging.getLogger("Chains")

from ddgs import DDGS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from constants import MAX_MESSAGE_MEMORY


async def trim_history(history: list, max_message_memory: int = MAX_MESSAGE_MEMORY) -> list:
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
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if any(tc["name"] == "search" for tc in msg.tool_calls):
                if i + 1 < len(history) and isinstance(history[i + 1], ToolMessage):
                    search_pair = [msg, history[i + 1]]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if any(tc["name"] == "vision" for tc in msg.tool_calls):
                if i + 1 < len(history) and isinstance(history[i + 1], ToolMessage):
                    search_pair = [msg, history[i + 1]]

    # Обов'язкові повідомлення які завжди залишаються
    required = [] # system + user question + AiMessage(search call) + ToolMessage(search response) + model response
    if human_message:
        required.append(human_message)
    if ai_message and ai_message not in search_pair:
        required.append(ai_message)

    required.extend(search_pair)
    required.extend(vision_pair)

    optional = [m for m in history if m not in required]

    max_history = max_message_memory - len(required) - 1 # 1 - вільне місце для запису SystemMessage наступного разу
    if len(optional) > max_history:
        optional = optional[-max_history:]

    result = optional + required

    # Сортуємо за оригінальним індексом в history
    return sorted(result, key=lambda m: history.index(m))

async def get_image_from_message(message, bot):
    photo = message.photo[-1].file_id
    buffer = await bot.download(photo)
    image_bytes = buffer.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_base64


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
