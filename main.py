import asyncio
import logging
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

from api import app


async def _run_bot() -> None:
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        return
    from bot import dp, bot
    while True:
        try:
            await dp.start_polling(bot)
        except Exception as e:
            log.error("Bot polling crashed: %s — restarting in 15s", e)
            await asyncio.sleep(15)


async def main() -> None:
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
    )
    server = uvicorn.Server(config)
    await asyncio.gather(
        server.serve(),
        _run_bot(),
        return_exceptions=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
