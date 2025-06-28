import os
import io
import asyncio
import base64
import logging
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_KEY = os.getenv("TELEGRAM_BOT_KEY")
TEMPLATE_PATH = os.getenv("TEMPLATE_PATH", "data/template.png")
MODEL_NAME = "gpt-image-1"
if not (OPENAI_API_KEY and TELEGRAM_BOT_KEY):
    raise RuntimeError("OPENAI_API_KEY and TELEGRAM_BOT_KEY error")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

PROMPT = (
    "сделай фотографию этого человека через 15 лет. "
    "сохрани основные черты лица, цвет волос, элементы одежды. "
    "он должен выглядеть успешным, фотография должна быть портретной"
)

OPENAI_SEM = asyncio.Semaphore(4)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

TOTAL_GENERATED = 0
COUNTER_LOCK = Lock()

BORDER_WIDTH = 2
CORNER_RAD = 12


async def age_photo(original: bytes) -> bytes:
    async with OPENAI_SEM:
        buf = io.BytesIO(original)
        buf.name = "input.png"
        buf.seek(0)
        res = await client.images.edit(
            model=MODEL_NAME,
            image=[buf],
            prompt=PROMPT,
            n=1,
            quality="high",
            size="1024x1024",
        )
        return base64.b64decode(res.data[0].b64_json)


def _prepare(img_bytes: bytes, w: int, h: int) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    img = ImageOps.fit(img, (w, h), Image.LANCZOS)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w, h), radius=CORNER_RAD, fill=255)
    img.putalpha(mask)
    return img


def make_collage(before_bytes: bytes, after_bytes: bytes) -> bytes:
    template = Image.open(TEMPLATE_PATH).convert("RGBA")
    left_box = (91, 239, 455, 619)
    right_box = (503, 239, 865, 619)
    w = left_box[2] - left_box[0]
    h = left_box[3] - left_box[1]
    before_img = _prepare(before_bytes, w, h)
    after_img = _prepare(after_bytes, w, h)
    template.paste(before_img, left_box[:2], mask=before_img)
    template.paste(after_img, right_box[:2], mask=after_img)
    draw = ImageDraw.Draw(template)
    for box in (left_box, right_box):
        draw.rounded_rectangle(box, radius=CORNER_RAD, outline="white", width=BORDER_WIDTH)
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rounded_rectangle((0, 0, w, h), radius=CORNER_RAD, fill=(0, 0, 0, 40))
        shadow = shadow.filter(ImageFilter.GaussianBlur(3))
        template.alpha_composite(shadow, (box[0], box[1] + 4))
    buf = io.BytesIO()
    template.convert("RGB").save(buf, "PNG")
    buf.seek(0)
    return buf.getvalue()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Пришлите фото — верну коллаж «сейчас / через 25 лет».")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    before_bytes = await photo_file.download_as_bytearray()
    status_msg = await update.message.reply_text("⏳ Обрабатываю…")

    async def work():
        try:
            after_bytes = await age_photo(before_bytes)
            collage = await asyncio.to_thread(make_collage, before_bytes, after_bytes)
            global TOTAL_GENERATED
            with COUNTER_LOCK:
                TOTAL_GENERATED += 1
                logging.info("Всего сгенерировано фотографий: %d", TOTAL_GENERATED)
            await status_msg.edit_text("Готово!")
            await update.message.reply_photo(collage)
        except Exception as e:
            logging.exception("Ошибка обработки")
            await status_msg.edit_text(f"Не получилось: {e}")

    context.application.create_task(work())


def main():
    application = Application.builder().token(TELEGRAM_BOT_KEY).build()
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=8))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    logging.info("Бот запущен.")
    application.run_polling()


if __name__ == "__main__":
    main()