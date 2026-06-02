import asyncio
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from api import app


async def main() -> None:
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
    )
    server = uvicorn.Server(config)
    tasks = [server.serve()]

    if os.getenv("TELEGRAM_BOT_TOKEN"):
        from bot import dp, bot
        tasks.append(dp.start_polling(bot))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
