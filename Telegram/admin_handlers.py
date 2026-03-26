import os
from functools import wraps

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

import settings_manager as s
from dotenv import load_dotenv

load_dotenv()

admin = Router()
ADMIN_ID = int(os.getenv("ADMIN_ID"))

def admin_only(func):
    @wraps(func)
    async def wrapper(message: Message, *args, **kwargs):
        if message.from_user.id != ADMIN_ID:
            return
        return await func(message, *args, **kwargs)
    return wrapper


@admin.message(Command("show_settings"))
@admin_only
async def show_settings(message: Message):
    allowed_settings = s.get_all()
    res = [f"{settings_name} : {settings_value}" for settings_name, settings_value in allowed_settings.items()]
    res = "\n".join(res)
    await message.reply(res)


@admin.message(Command("set_setting"))
@admin_only
async def set_setting(message: Message):
    # прибираємо команду, лишаємо "KEY value"
    parts = message.text.removeprefix("/set_setting").strip().split(maxsplit=1)

    if len(parts) != 2:
        await message.reply("Формат <code>/set_setting KEY значення</code>", parse_mode="HTML")
        return

    key, value = parts

    if key not in s.get_all():
        await message.reply(f"Налаштування <code>{key}</code> не існує", parse_mode="HTML")
        return

    old_value = s.get(key)
    try:
        value = type(old_value)(value)
    except ValueError:
        await message.reply(f"Очікується тип {type(old_value).__name__}", parse_mode="HTML")
        return

    s.save(key, value)
    await message.reply(f"<code>{key}</code> → <code>{value}</code>", parse_mode="HTML")
