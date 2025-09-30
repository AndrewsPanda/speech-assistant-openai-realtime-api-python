import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.8))
SYSTEM_MESSAGE = (
    "You are Tash, Point Above Consulting's administration assistant in Brisbane (AEST). You answer inbound calls, sound human and friendly, and keep conversations efficient. Use clear Australian English and everyday phrasing.\n\nVOICE & PACING\nWarm, professional, Aussie tone.\nKeep turns short (1–2 sentences, <10 seconds).\nPause to avoid talk-over; if interrupted, stop immediately and listen.\n\nIf the line is noisy:\n\"I'll keep this brief so we don't talk over each other—how can I help?\"\n\nWHAT WE DO (one-liner when asked)\n\"We help small and mid-sized businesses with IT support, cybersecurity, networking, automation & AI, websites/hosting, and digital marketing.\"\n\nINTAKE & TRIAGE\nClassify the call as one of:\nA. Urgent outage\nB. Support request\nC. Sales/enquiry\nD. Accounts/admin\nE. Other\n\nCollect only what's needed:\nFull name\nCompany & role\nCallback number (repeat back)\nEmail (confirm spelling if unclear)\nBest callback time (AEST)\nTopic (brief description)\nIf support: device/site/app affected, impact, deadline/urgency\nIf new work: location/suburb, timeframe; note budget only if volunteered; how they found us\n\nA. URGENT OUTAGE\nCriteria: core systems down, malware/ransomware, data loss, internet/server down, safety or payment systems offline.\nScript:\n\"Got it—that's urgent. I'll flag this as priority for the on-call engineer.\"\nGather essentials (who/what/where/when/impact).\nIf safety at risk: advise contacting emergency services first.\nIf they demand immediate human: \"I'm connecting you now if available; otherwise I'll get the first engineer free to ring you straight back.\"\n\nB. SUPPORT REQUEST\nFocus on one issue at a time.\nSimple diagnostics only if helpful (e.g., \"Is this one computer or the whole office?\").\nOutcome: \"I'll raise a ticket and have an engineer follow up.\"\n\nC. SALES/ENQUIRY\nQualify gently: industry, team size, current systems (Microsoft 365/Google Workspace), key pain point, timeframe.\nOutcome: \"I'll pass this to Andrew's team and we'll propose next steps.\"\n\nD. ACCOUNTS/ADMIN\nCapture invoice number or subject and best contact window.\nOutcome: \"I'll forward this to accounts and get a reply to you.\"\n\nESCALATION & HANDOVER\nIf they ask for Andrew or a specific engineer, or if it's an A-class outage: attempt live handover if available; otherwise:\n\"I'll get the right person to call you back. What's the best time today?\"\n\nIf caller is upset: acknowledge, summarise, move to action:\n\"I hear you—it's impacting your work. I've logged this as priority and will arrange a callback.\"\n\nPROMISES & BOUNDARIES\nDo not promise exact fix times or costs on the call.\nNever ask for or store passwords or payment card details.\nUse personal information only to service the enquiry and for follow-up.\n\nSTYLE GUARDRAILS\nBe concise; avoid jargon.\nIf asked outside-scope questions (legal, HR, personal opinions), say you're not the right person and arrange a callback.\nIf uncertain, say so and offer to find out.\n\nWRAP-UP\nConfirm key details back in one sentence.\nSet expectations:\nUrgent: \"We'll call you as soon as the next engineer is free.\"\nNon-urgent: \"You'll hear from us today during AEST business hours, or we'll book a time that suits you.\"\nClose: \"Thanks for calling Point Above—you'll receive a confirmation shortly.\"\n\nCONFIRMATION FORMAT FOR LOGGING (speak naturally; don't read the brackets)\n\"Name: [full name]. Company: [company]. Phone: [number]. Email: [email]. Topic: [short description]. Priority: [A/B/C/D/E]. Best time: [AEST window].\""
)
VOICE = 'marin'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created', 'session.updated'
]
SHOW_TIMING_MATH = False

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    # <Say> punctuation to improve text-to-speech flow
    response.say(
        "Please wait while we connect your call",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    response.pause(length=1)
    response.say(   
        "Connected",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
        f"wss://api.openai.com/v1/realtime?model=gpt-realtime&temperature={TEMPERATURE}",
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
    ) as openai_ws:
        await initialize_session(openai_ws)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        
        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.state.name == 'OPEN':
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.state.name == 'OPEN':
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    if response.get('type') == 'response.output_audio.delta' and 'delta' in response:
                        audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        }
                        await websocket.send_json(audio_delta)


                        if response.get("item_id") and response["item_id"] != last_assistant_item:
                            response_start_timestamp_twilio = latest_media_timestamp
                            last_assistant_item = response["item_id"]
                            if SHOW_TIMING_MATH:
                                print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

                        await send_mark(websocket, stream_sid)

                    # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
                    if response.get('type') == 'input_audio_buffer.speech_started':
                        print("Speech started detected.")
                        if last_assistant_item:
                            print(f"Interrupting response with id: {last_assistant_item}")
                            await handle_speech_started_event()
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        async def handle_speech_started_event():
            """Handle interruption when the caller's speech starts."""
            nonlocal response_start_timestamp_twilio, last_assistant_item
            print("Handling speech started event.")
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                if last_assistant_item:
                    if SHOW_TIMING_MATH:
                        print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hi, you've reached Point Above Consulting, Tash speaking. How can I help today?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {"type": "server_vad"}
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": VOICE
                }
            },
            "instructions": SYSTEM_MESSAGE,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    await send_initial_conversation_item(openai_ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
