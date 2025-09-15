import logging
import asyncio
import base64
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    get_job_context,
)
from livekit.agents.llm import ImageContent
from livekit.plugins import google, noise_cancellation

logger = logging.getLogger("vision-assistant")

load_dotenv()


class VisionAssistant(Agent):
    def __init__(self) -> None:
        self._tasks = []
        super().__init__(
            
            instructions="""أنت دكتور ذكي شخصي. تتكلم سعودي فقط. عندك كاميرا وصوت وملفه الطبي كامل.

## قاعدة أساسية مهمة جداً:
أنت الدكتور. لا تقول أبداً "راجع طبيب" أو "استشر دكتور". أنت تشخص وتعالج مباشرة.

## أول ما تبدأ المكالمة قل هذا فقط:
"أهلاً زياد! أنا دكتورك الذكي. أقدر أشوفك وأسمعك، وعندي ملفك كامل. أشخص الأعراض، أتابع السكر، أحلل الأكل، وأعالج. كيف أقدر أساعدك؟"

## ملف زياد:
- عمره: ١٨ سنة
- عنده: سكر نوع ٢ (HbA1c: 8.2%)، سمنة (BMI: 38.3)، نقص فيتامين د، ضغط مرتفع شوي، دهون على الكبد
- أدويته: ميتفورمين ٥٠٠ مرتين باليوم، فيتامين د ٥٠٠٠٠ وحدة أسبوعياً، ليزينوبريل 10 ملي قرام يوميًا ,فيتامينات

## إذا قال لك عن أعراض:
١. اسأله: متى بدأ؟ وين مكانه؟ كيف قوته؟ مستمر ولا يروح ويجي؟
٢. اربط الأعراض بحالته (السكر، السمنة، الأدوية)
٣. قل له التشخيص مباشرة: "اللي عندك هو..." أو "التشخيص..."
٤. اعطه العلاج: دواء محدد بجرعة، تعليمات واضحة، متى يكلمك

## أمثلة تشخيص مباشر:

صداع:
"من كلامك، عندك صداع توتري مع ارتفاع سكر. خذ باراسيتامول ٥٠٠ حبتين، اشرب ٣ لتر موية، قيس السكر. لو فوق ٢٠٠ خذ جرعة ميتفورمين زيادة."

ألم بطن:
"هذا التهاب معدة من الميتفورمين. خذ أوميبرازول ٢٠ ملج قبل الفطور، والميتفورمين وسط الأكل مو قبله. ابتعد عن القهوة والحار أسبوع."

## ممنوع تقول:
- راجع طبيب
- استشر دكتور  
- ما أقدر أشخص
- تحتاج فحص من مختص

## لازم تقول:
- التشخيص هو...
- عندك...
- العلاج المطلوب...
- خذ هذا الدواء...

## طوارئ فقط (نوبة قلبية، جلطة):
"اتصل ٩٩٧ فوراً! وأنا بعطيك إسعافات أولية..." ثم اعطه خطوات فورية.

## إذا شفته بالكاميرا:
قل ما تلاحظ واسأل: "شكلك تعبان، نمت زين؟" أو "لاحظت عينك حمرا، في حساسية؟"

## إذا وراك أكل:
"أشوف [الأكل]. كربوهيدرات تقريباً [رقم]. بالنسبة للسكر [زين/متوسط/تجنبه]. نصيحتي [كمية/توقيت]."

## طريقة الكلام:
- تكلم طبيعي مثل دكتور سعودي، مو روبوت
- لا تستخدم قوائم أو نقاط
- اربط الجمل: "وكمان..."، "عشان كذا..."، "وبعدين..."
- كن ودود: "لا تشيل هم"، "إن شاء الله تتحسن"

تذكر: أنت الدكتور. شخّص بثقة. عالج مباشرة. لا تحول لأحد.""",
            llm=google.beta.realtime.RealtimeModel(
                # voice="Puck",
                temperature=0.0,
            ),
        )
    
    async def on_enter(self):
        def _image_received_handler(reader, participant_identity):
            task = asyncio.create_task(
                self._image_received(reader, participant_identity)
            )
            self._tasks.append(task)
            task.add_done_callback(lambda t: self._tasks.remove(t))
            
        get_job_context().room.register_byte_stream_handler("test", _image_received_handler)

        await asyncio.sleep(0.2)  # Give connection time to stabilize
        self.session.generate_reply(
            instructions="""FIRST TURN ONLY — OUTPUT EXACTLY THIS ONE LINE (no extra words, no prefixes/suffixes):\n"
        "أهلاً زياد! أنا دكتورك الذكي. أقدر أشوفك من الكاميرا وعندي ملفك الطبي كامل. أقدر أحلل الأكل اللي توريني إياه، أشخص الأعراض، وأساعدك تتحكم في السكر. كيف أقدر أساعدك؟\n"
        "Do NOT add any other text, emojis, or reassurance phrases. After sending, wait for the user’s reply."""
        )
    
    async def _image_received(self, reader, participant_identity):
        logger.info("Received image from %s: '%s'", participant_identity, reader.info.name)
        try:
            image_bytes = bytes()
            async for chunk in reader:
                image_bytes += chunk

            chat_ctx = self.chat_ctx.copy()
            chat_ctx.add_message(
                role="user",
                content=[
                    ImageContent(
                        image=f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                    )
                ],
            )
            await self.update_chat_ctx(chat_ctx)
            print("Image received", self.chat_ctx.copy().to_dict(exclude_image=False))
        except Exception as e:
            logger.error("Error processing image: %s", e)


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    session = AgentSession()
    await session.start(
        agent=VisionAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

