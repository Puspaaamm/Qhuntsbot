import base64
import logging
import os
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# -------------------------------------------------------------------
# 1. Logging & Environment Setup
# -------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

if not TELEGRAM_TOKEN or not OPENROUTER_KEY:
    raise ValueError(
        "❌ Missing API Keys! Please check your .env file for TELEGRAM_BOT_TOKEN and OPENROUTER_API_KEY."
    )

# Initialize OpenRouter Client
ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

# -------------------------------------------------------------------
# 2. Configuration & State Management
# -------------------------------------------------------------------
AVAILABLE_MODELS = {
    "⚡ Gemini 2.0 Flash (Fast & Free)": "google/gemini-2.0-flash-001",
    "🧠 GPT-4o Mini (Smart)": "openai/gpt-4o-mini",
    "🚀 Claude 3.5 Sonnet (Advanced)": "anthropic/claude-3.5-sonnet",
    "🦙 LLaMA 3.3 70B (Open Source)": "meta-llama/llama-3.3-70b-instruct",
}

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
MAX_HISTORY = 10  # Context history limit per user

user_models = defaultdict(lambda: DEFAULT_MODEL)
user_chat_history = defaultdict(list)

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are an expert AI assistant and tutor integrated into a Telegram bot. "
        "Provide clear, accurate, and structured answers in Hindi, Hinglish, or English based on user query. "
        "When solving math, science problems, or reading text from photos, provide step-by-step solutions."
    ),
}


# -------------------------------------------------------------------
# 3. Command Handlers
# -------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_text = (
        "👋 **Namaste! Main aapka AI Assistant hoon.**\n\n"
        "Main aapke in sabhi kaam mein help kar sakta hoon:\n"
        "• 💬 **Sawal-Jawab & Chat**: Kuch bhi puchiye.\n"
        "• 📷 **Photo Solver**: Homework, Math question, ya Code ki pic bhejiye solution paane ke liye.\n"
        "• 🔀 **Model Switch**: `/model` command se AI model badlein.\n"
        "• 🧹 **Reset Chat**: `/reset` command se purani chat clear karein.\n\n"
        "Shuru karne ke liye koi message ya photo bhejiye!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "💡 **Kaise Use Karein:**\n\n"
        "1. **Text Message**: Apna sawal normal type karke bhejiye.\n"
        "2. **Photo**: Sawaal ki photo bhejiye. Caption mein instruction bhi likh sakte hain.\n"
        "3. **Change Model**: `/model` type karke doosra AI model select karein.\n"
        "4. **Clear History**: Chat reset karne ke liye `/reset` type karein."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_chat_history[user_id].clear()
    await update.message.reply_text("🧹 **Conversation history clear ho gayi hai!**")


async def model_menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    current_model = user_models[user_id]

    keyboard = [
        [InlineKeyboardButton(label, callback_data=f"set_model:{model_id}")]
        for label, model_id in AVAILABLE_MODELS.items()
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"⚙️ **Current Active Model**: `{current_model}`\n\nNeeche se naya model chuniye:",
        reply_markup=reply_markup,
        parse_mode="Markdown",
    )


async def model_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    _, selected_model = query.data.split(":", 1)
    user_id = query.from_user.id
    user_models[user_id] = selected_model

    await query.edit_message_text(
        f"✅ **Active model set to:**\n`{selected_model}`",
        parse_mode="Markdown",
    )


# -------------------------------------------------------------------
# 4. Message Handlers (Text & Image)
# -------------------------------------------------------------------
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_prompt = update.message.text
    active_model = user_models[user_id]

    if not user_chat_history[user_id]:
        user_chat_history[user_id].append(SYSTEM_PROMPT)

    user_chat_history[user_id].append({"role": "user", "content": user_prompt})

    if len(user_chat_history[user_id]) > (MAX_HISTORY + 1):
        user_chat_history[user_id] = [SYSTEM_PROMPT] + user_chat_history[user_id][-(MAX_HISTORY):]

    await update.message.chat.send_action(action="typing")

    try:
        response = ai_client.chat.completions.create(
            model=active_model,
            messages=user_chat_history[user_id],
            extra_headers={
                "HTTP-Referer": "https://telegram-bot.local",
                "X-Title": "Telegram AI Assistant",
            },
        )
        answer = response.choices[0].message.content

        user_chat_history[user_id].append({"role": "assistant", "content": answer})
        await update.message.reply_text(answer)

    except Exception as e:
        logger.error(f"Error for user {user_id}: {e}")
        await update.message.reply_text(f"❌ **Error:** {str(e)}")


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    active_model = user_models[user_id]
    
    caption_text = (
        update.message.caption
        or "Is image ko ache se analyze karein, isme diye sawal ko solve karein aur step-by-step explain karein."
    )

    await update.message.chat.send_action(action="typing")

    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes).decode("utf-8")

        messages = [
            SYSTEM_PROMPT,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": caption_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            },
        ]

        response = ai_client.chat.completions.create(
            model=active_model,
            messages=messages,
            extra_headers={
                "HTTP-Referer": "https://telegram-bot.local",
                "X-Title": "Telegram AI Assistant",
            },
        )

        answer = response.choices[0].message.content
        await update.message.reply_text(answer)

    except Exception as e:
        logger.error(f"Photo Error for user {user_id}: {e}")
        await update.message.reply_text(f"❌ **Image Process Error:** {str(e)}")


# -------------------------------------------------------------------
# 5. Main Execution
# -------------------------------------------------------------------
def main() -> None:
    print("🚀 Initializing Telegram Bot...")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("model", model_menu_command))

    # Callbacks & Messages
    app.add_handler(CallbackQueryHandler(model_callback_handler, pattern="^set_model:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))

    print("🤖 Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
