import os
import warnings

from aiogram import Bot, Dispatcher
from aiogram.webhook.aiohttp_server import SimpleRequestHandler
from aiohttp import web

import settings_manager as s
from Chains.models import load_models
from Telegram.admin_handlers import admin
from Telegram.middleware import ThrottlingMiddleware, AnswerTypeMiddleware, AlbumMiddleware
from Telegram.user_handlers import user

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

SECRET = os.getenv("WEBHOOK_SECRET")


async def main():
    import asyncio
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Chains")

    bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
    dp = Dispatcher()
    dp.include_router(admin)
    dp.include_router(user)
    dp.message.middleware(AlbumMiddleware())
    dp.message.middleware(ThrottlingMiddleware())
    dp.message.middleware(AnswerTypeMiddleware())

    async def on_startup(app):
        for attempt in range(10):
            try:
                await bot.set_webhook(
                    "https://eye-help-you.fly.dev/webhook",
                    secret_token=SECRET
                )
                logger.debug("Webhook встановлено!")
                return
            except Exception as e:
                logger.warning(f"Спроба {attempt + 1}/10 не вдалась: {e}")
                await asyncio.sleep(3)
        logger.error("Не вдалось встановити webhook після 10 спроб")

    async def on_shutdown(app):
        await bot.delete_webhook()

    async def health(request):
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/health", health)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
        secret_token=SECRET
    ).register(app, path="/webhook")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    logger.debug("Сервер запущено на порті 8080")

    await asyncio.Event().wait()

if __name__ == '__main__':
    try:
        import asyncio
        from dotenv import load_dotenv

        load_dotenv()
        s.load_settings()
        load_models()
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
