"""
Jarvis - Single-file, multi-feature personal assistant (Python)

Features:
- Voice input (speech_recognition) and TTS (pyttsx3)
- Optional OpenAI GPT (if you provide OPENAI_API_KEY)
- Face recognition login (opencv + face_recognition)
- Weather (OpenWeatherMap optional via OPENWEATHER_API_KEY)
- News scraping (requests + beautifulsoup4)
- Wikipedia search (wikipedia module)
- SQLite DB for notes & reminders
- Scheduler (schedule)
- GUI (tkinter)
- Music playback (pygame)
- Image capture & save
- Simple data visualization (matplotlib)
- File automation utilities
- Helpful logging & threading for concurrency

Make sure to provide API keys in environment variables or a .env if you use python-dotenv:
- OPENAI_API_KEY (optional)
- OPENWEATHER_API_KEY (optional)

"""

import os
import sys
import threading
import queue
import time
import json
import random
import sqlite3
import shutil
import glob
import re
from datetime import datetime, timedelta

# --- Audio / Voice ---
import speech_recognition as sr
import pyttsx3

# --- Web / Data ---
import requests
from bs4 import BeautifulSoup
import wikipedia

# --- Face recognition & images ---
import cv2
import face_recognition
from PIL import Image, ImageTk

# --- GUI ---
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# --- Scheduler & timing ---
import schedule

# --- Media ---
import pygame

# --- Visualization ---
import matplotlib.pyplot as plt

# --- Optional OpenAI integration ---
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# --- Misc ---
import pyperclip  # clipboard helper
from pathlib import Path

# --- Configuration & Globals ---
APP_NAME = "Jarvis"
BASE_DIR = Path(__file__).parent.resolve()
DB_PATH = BASE_DIR / "jarvis_data.db"
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
CAPTURE_DIR = BASE_DIR / "captures"
MUSIC_DIR = BASE_DIR / "music"

# Ensure directories exist
for d in (KNOWN_FACES_DIR, CAPTURE_DIR, MUSIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Load API keys from environment (optionally use dotenv)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

if OPENAI_AVAILABLE and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 170)  # words per minute

# Initialize speech recognizer
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Task queue for main loop
task_q = queue.Queue()

# Initialize pygame mixer for music
pygame.mixer.init()

# --- Utility Functions ----------------------------------------------------

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")

def speak(text, block=False):
    """Speak text using pyttsx3 and optionally block until done."""
    if not text:
        return
    log("Speaking: " + text)
    def _speak():
        tts_engine.say(text)
        tts_engine.runAndWait()
    if block:
        _speak()
    else:
        t = threading.Thread(target=_speak, daemon=True)
        t.start()

def listen(timeout=6, phrase_time_limit=8):
    """Record from microphone and return recognized text (or None)."""
    with microphone as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.6)
        try:
            audio = recognizer.listen(mic, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            return None
    try:
        text = recognizer.recognize_google(audio)
        log("Heard: " + text)
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        log("Speech recognition error: " + str(e))
        return None

# --- Database Setup ------------------------------------------------------

def init_db():
    con = sqlite3.connect(DB_PATH)
    c = con.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT,
            remind_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            done INTEGER DEFAULT 0
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    con.commit()
    con.close()
    log("Database initialized.")

def db_add_note(title, content):
    con = sqlite3.connect(DB_PATH)
    c = con.cursor()
    c.execute('INSERT INTO notes (title, content) VALUES (?, ?)', (title, content))
    con.commit()
    con.close()
    log("Note saved: " + title)

def db_get_notes():
    con = sqlite3.connect(DB_PATH)
    c = con.cursor()
    c.execute('SELECT id, title, content, created_at FROM notes ORDER BY created_at DESC')
    rows = c.fetchall()
    con.close()
    return rows

def db_add_reminder(message, remind_at_dt):
    con = sqlite3.connect(DB_PATH)
    c = con.cursor()
    c.execute('INSERT INTO reminders (message, remind_at) VALUES (?, ?)', (message, remind_at_dt))
    con.commit()
    con.close()
    log("Reminder added: " + message + " at " + str(remind_at_dt))

def db_get_pending_reminders():
    con = sqlite3.connect(DB_PATH)
    c = con.cursor()
    c.execute('SELECT id, message, remind_at FROM reminders WHERE done=0 ORDER BY remind_at ASC')
    rows = c.fetchall()
    con.close()
    return rows

def db_mark_reminder_done(reminder_id):
    con = sqlite3.connect(DB_PATH)
    c = con.cursor()
    c.execute('UPDATE reminders SET done=1 WHERE id=?', (reminder_id,))
    con.commit()
    con.close()
    log("Reminder marked done: " + str(reminder_id))

def db_log(action, details=""):
    con = sqlite3.connect(DB_PATH)
    c = con.cursor()
    c.execute('INSERT INTO usage_log (action, details) VALUES (?, ?)', (action, details))
    con.commit()
    con.close()

# --- Face Recognition ----------------------------------------------------

def load_known_faces():
    """
    Loads images from KNOWN_FACES_DIR and creates encodings.
    Each file name (without extension) is used as the person's name.
    """
    known_encodings = []
    known_names = []
    for img_path in KNOWN_FACES_DIR.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        try:
            img = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(img_path.stem)
                log(f"Loaded face for {img_path.stem}")
            else:
                log(f"No faces found in {img_path.name}")
        except Exception as e:
            log("Error loading face " + str(img_path) + " : " + str(e))
    return known_encodings, known_names

KNOWN_ENCODINGS, KNOWN_NAMES = load_known_faces()

def face_login(timeout=15):
    """
    Activate webcam and try to recognize face from KNOWN_FACES_DIR.
    Returns recognized name or None.
    """
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    recognized_name = None
    speak("Activating face login. Please look at the camera.")
    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if not ret:
            continue
        # resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(KNOWN_ENCODINGS, face_encoding, tolerance=0.45)
            face_distances = face_recognition.face_distance(KNOWN_ENCODINGS, face_encoding) if KNOWN_ENCODINGS else []
            if True in matches:
                best_idx = int(np.argmin(face_distances)) if face_distances != [] else matches.index(True)
                recognized_name = KNOWN_NAMES[best_idx]
                log("Recognized: " + recognized_name)
                cap.release()
                cv2.destroyAllWindows()
                speak(f"Welcome back, {recognized_name}")
                db_log("face_login", recognized_name)
                return recognized_name
        # show frame with a little rectangle
        cv2.imshow("Face Login - Press Q to cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    speak("Face login timed out or failed.")
    return None

# --- Commands: Web / Data ------------------------------------------------

def get_weather(city):
    """Get weather using OpenWeatherMap. API key required."""
    if not OPENWEATHER_API_KEY:
        return "OpenWeatherMap API key not configured."
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        r = requests.get(url, timeout=8)
        data = r.json()
        if data.get("cod") != 200:
            return f"Weather API: {data.get('message', 'error')}"
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        humidity = data['main']['humidity']
        wind = data['wind']['speed']
        res = f"Weather in {city}: {temp}°C, {desc}. Humidity {humidity}%. Wind {wind} m/s."
        log("Weather fetched for " + city)
        return res
    except Exception as e:
        log("Weather error: " + str(e))
        return "Failed to fetch weather."

def fetch_news_headlines(source_url="https://news.google.com/topstories?hl=en-IN&gl=IN&ceid=IN:en", count=6):
    """Scrape headlines from Google News (simple approach)."""
    try:
        r = requests.get(source_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        # Google News is dynamic; fallback to general article titles
        headlines = []
        for h in soup.find_all(["h3", "h4"], limit=count):
            text = h.get_text().strip()
            if text:
                headlines.append(text)
        if not headlines:
            # fallback generic
            for a in soup.find_all("a", limit=count):
                t = a.get_text().strip()
                if t:
                    headlines.append(t)
        log("Fetched news headlines: " + str(len(headlines)))
        return headlines[:count]
    except Exception as e:
        log("News fetch error: " + str(e))
        return []

def wiki_search(query, sentences=2):
    try:
        summary = wikipedia.summary(query, sentences=sentences, auto_suggest=True, redirect=True)
        log("Wiki summary fetched for: " + query)
        return summary
    except Exception as e:
        log("Wiki error: " + str(e))
        return "No summary found."

# --- OpenAI Chat (optional) ----------------------------------------------

def openai_reply(prompt, max_tokens=150):
    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        return "OpenAI integration not available. Provide OPENAI_API_KEY and install openai package."
    try:
        resp = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            n=1,
            stop=None,
        )
        text = resp.choices[0].text.strip()
        db_log("openai_query", prompt[:200])
        log("OpenAI replied.")
        return text
    except Exception as e:
        log("OpenAI error: " + str(e))
        return "OpenAI request failed."

# --- Music control -------------------------------------------------------

def list_music():
    files = []
    for ext in ("*.mp3", "*.wav", "*.ogg"):
        files.extend(MUSIC_DIR.glob(ext))
    return [str(p) for p in files]

def play_music(filepath=None):
    try:
        if not filepath:
            files = list_music()
            if not files:
                speak("No music files found in the music folder.")
                return
            filepath = random.choice(files)
        log("Playing music: " + filepath)
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        db_log("play_music", filepath)
        speak(f"Playing {os.path.basename(filepath)}")
    except Exception as e:
        log("Music error: " + str(e))
        speak("Failed to play music.")

def stop_music():
    pygame.mixer.music.stop()
    db_log("stop_music", "")
    speak("Music stopped.")

# --- File & Automation Helpers -------------------------------------------

def open_file_explorer(path="."):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform.startswith("darwin"):
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
    db_log("open_file_explorer", path)

def capture_image(save_path=None):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    if save_path is None:
        save_path = CAPTURE_DIR / f"capture_{int(time.time())}.jpg"
    cv2.imwrite(str(save_path), frame)
    db_log("capture_image", str(save_path))
    return str(save_path)

# --- Reminder Scheduler --------------------------------------------------

def refresh_reminders():
    rows = db_get_pending_reminders()
    now = datetime.now()
    for row in rows:
        _id, message, remind_at = row
        # remind_at may be string - parse
        remind_dt = datetime.fromisoformat(remind_at) if isinstance(remind_at, str) else remind_at
        if remind_dt <= now:
            speak(f"Reminder: {message}")
            db_mark_reminder_done(_id)
            db_log("reminder_trigger", message)

def start_scheduler_loop():
    schedule.every(30).seconds.do(refresh_reminders)
    while True:
        schedule.run_pending()
        time.sleep(1)

# --- Command parsing and handlers ---------------------------------------

def handle_command(text):
    """Main command parser. Returns a textual response for GUI log."""
    if not text:
        return "I didn't hear anything."
    t = text.lower()
    db_log("voice_cmd", t[:200])
    # Greetings
    if any(phr in t for phr in ["hello", "hi jarvis", "hey jarvis", "hey"]):
        resp = random.choice(["Hello.", "Hi there.", "At your service."])
        speak(resp)
        return resp

    if "your name" in t:
        speak("I am Jarvis, your personal assistant.")
        return "I am Jarvis."

    if "time" in t:
        now = datetime.now().strftime("%I:%M %p")
        speak("The time is " + now)
        return "Time: " + now

    if "date" in t:
        now = datetime.now().strftime("%A, %d %B %Y")
        speak("Today is " + now)
        return "Date: " + now

    if "weather in" in t or t.startswith("weather "):
        # parse city name
        m = re.search(r"weather (in )?(?P<city>[\w\s,]+)", t)
        if m:
            city = m.group("city").strip()
        else:
            city = "New Delhi"  # default if not found
        res = get_weather(city)
        speak(res)
        return res

    if t.startswith("search ") or t.startswith("google "):
        query = t.split(" ", 1)[1]
        speak(f"Searching the web for {query}")
        # simple google via web scraping (limited) - we fallback to wiki
        try:
            wiki_res = wiki_search(query, sentences=2)
            speak("I found this on Wikipedia.")
            return wiki_res
        except Exception:
            return "Search failed."

    if "wikipedia" in t:
        query = t.replace("wikipedia", "").strip()
        if not query:
            return "Ask Wikipedia about what?"
        res = wiki_search(query, sentences=2)
        speak(res)
        return res

    if "news" in t or "headlines" in t:
        headlines = fetch_news_headlines()
        if headlines:
            speak("Here are the top headlines.")
            for h in headlines[:5]:
                speak(h, block=False)
            return "\n".join(headlines)
        else:
            speak("Could not fetch news.")
            return "News fetch failed."

    if "play music" in t or "play song" in t:
        # optionally parse filename
        m = re.search(r"play (?:song )?(?P<name>.+)", t)
        name = m.group("name").strip() if m else ""
        files = list_music()
        if not files:
            speak("No music found in the music folder.")
            return "No music files."
        matching = [f for f in files if name.lower() in os.path.basename(f).lower()] if name else files
        if not matching:
            speak("No matching songs found, playing random.")
            play_music()
            return "Playing random music."
        play_music(matching[0])
        return "Playing: " + os.path.basename(matching[0])

    if "stop music" in t or "pause music" in t:
        stop_music()
        return "Music stopped."

    if "take picture" in t or "capture" in t:
        path = capture_image()
        if path:
            speak("Picture saved.")
            return "Captured: " + path
        else:
            return "Capture failed."

    if "note" in t or "remember" in t:
        # parse "remember to ..." or "note that ..."
        m = re.search(r"(remember|note) (to )?(?P<msg>.+)", t)
        if m:
            msg = m.group("msg").strip()
            # save simple note
            db_add_note("Voice Note", msg)
            speak("Saved note.")
            return "Saved note: " + msg
        else:
            return "What should I remember?"

    if "set reminder" in t or "remind me" in t:
        # try to parse "remind me to X at HH:MM" or "in N minutes"
        m1 = re.search(r"remind me to (?P<task>.+) at (?P<time>[\d:apm\s]+)", t)
        m2 = re.search(r"remind me to (?P<task>.+) in (?P<num>\d+) (minutes|minute|hours|hour)", t)
        if m1:
            task = m1.group("task").strip()
            timetext = m1.group("time").strip()
            remind_dt = parse_time_text(timetext)
            if remind_dt:
                db_add_reminder(task, remind_dt.isoformat())
                speak(f"Reminder set for {remind_dt.strftime('%I:%M %p')}.")
                return f"Reminder: {task} at {remind_dt}"
        elif m2:
            task = m2.group("task").strip()
            num = int(m2.group("num"))
            unit = "minutes" if "minute" in t else "hours"
            if "minute" in t:
                remind_dt = datetime.now() + timedelta(minutes=num)
            else:
                remind_dt = datetime.now() + timedelta(hours=num)
            db_add_reminder(task, remind_dt.isoformat())
            speak(f"Okay. I'll remind you in {num} {unit}.")
            return f"Reminder set: {task} at {remind_dt}"
        else:
            # fallback: ask for time - but we can't ask via voice here in this handler
            return "Try saying: remind me to <task> in 10 minutes or remind me to <task> at 06:30 PM."

    if "open" in t and ("folder" in t or "explorer" in t or "show files" in t):
        open_file_explorer(str(BASE_DIR))
        return "Opened file explorer."

    if "who am i" in t or "who is me" in t:
        # simple privacy: if face recognized earlier, can return name - but we don't track session here
        return "You are my user."

    if "chatgpt" in t or "openai" in t or "ask gpt" in t:
        # extract question
        q = t.split(" ", 1)[1] if " " in t else t
        resp = openai_reply(q)
        speak(resp)
        return resp

    # Fallback: small talk
    if "joke" in t:
        joke = random.choice([
            "I would tell you a UDP joke but you might not get it.",
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "I told my computer I needed a break, and it said 'No problem — I'll go to sleep.'"
        ])
        speak(joke)
        return joke

    # Very dumb fallback: echo
    speak("I didn't understand that. I can search the web or take a note for you.")
    return "Unknown command: " + text

# --- Helpers for parsing times -------------------------------------------

def parse_time_text(timetext):
    """
    Parse some human-like times: "6:30 pm", "18:45", "tomorrow 7am"
    Returns datetime or None.
    """
    try:
        timetext = timetext.strip().lower()
        now = datetime.now()
        # simple patterns
        m = re.match(r"(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ampm>am|pm)?", timetext)
        if m:
            h = int(m.group("h"))
            minute = int(m.group("m"))
            ampm = m.group("ampm")
            if ampm:
                if ampm == "pm" and h < 12:
                    h += 12
                if ampm == "am" and h == 12:
                    h = 0
            dt = now.replace(hour=h, minute=minute, second=0, microsecond=0)
            if dt < now:
                dt += timedelta(days=1)
            return dt
        m2 = re.match(r"(?P<h>\d{1,2})\s*(am|pm)", timetext)
        if m2:
            h = int(m2.group("h"))
            if "pm" in timetext and h < 12:
                h += 12
            dt = now.replace(hour=h, minute=0, second=0, microsecond=0)
            if dt < now:
                dt += timedelta(days=1)
            return dt
        if "tomorrow" in timetext:
            # fallback to tomorrow morning at 9
            dt = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
            return dt
    except Exception as e:
        log("parse_time_text error: " + str(e))
    return None

# --- GUI -----------------------------------------------------------------

class JarvisGUI:
    def __init__(self, root):
        self.root = root
        root.title(APP_NAME + " - Python Assistant")
        root.geometry("900x600")
        self.create_widgets()
        self.poll_queue()
        # start listening thread
        self.listening = False
        self.voice_thread = None

    def create_widgets(self):
        # Left: controls
        frame_left = tk.Frame(self.root, width=300)
        frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        btn_face = tk.Button(frame_left, text="Face Login", command=self.action_face_login)
        btn_face.pack(fill=tk.X, pady=4)

        btn_listen = tk.Button(frame_left, text="Listen (Voice Command)", command=self.toggle_listen)
        btn_listen.pack(fill=tk.X, pady=4)

        btn_notes = tk.Button(frame_left, text="My Notes", command=self.show_notes)
        btn_notes.pack(fill=tk.X, pady=4)

        btn_reminders = tk.Button(frame_left, text="Reminders", command=self.show_reminders)
        btn_reminders.pack(fill=tk.X, pady=4)

        btn_news = tk.Button(frame_left, text="Fetch News", command=self.gui_fetch_news)
        btn_news.pack(fill=tk.X, pady=4)

        btn_weather = tk.Button(frame_left, text="Get Weather", command=self.gui_get_weather)
        btn_weather.pack(fill=tk.X, pady=4)

        btn_play = tk.Button(frame_left, text="Play Random Music", command=lambda: play_music(None))
        btn_play.pack(fill=tk.X, pady=4)

        btn_stop = tk.Button(frame_left, text="Stop Music", command=stop_music)
        btn_stop.pack(fill=tk.X, pady=4)

        btn_capture = tk.Button(frame_left, text="Capture Photo", command=self.gui_capture)
        btn_capture.pack(fill=tk.X, pady=4)

        btn_quit = tk.Button(frame_left, text="Exit", command=self.root.quit)
        btn_quit.pack(fill=tk.X, pady=4)

        # Right: main log & input
        frame_right = tk.Frame(self.root)
        frame_right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=6, pady=6)

        self.log_area = scrolledtext.ScrolledText(frame_right, state='disabled', wrap=tk.WORD)
        self.log_area.pack(expand=True, fill=tk.BOTH)

        bottom_frame = tk.Frame(frame_right)
        bottom_frame.pack(fill=tk.X)

        self.entry = tk.Entry(bottom_frame)
        self.entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4, pady=4)
        self.entry.bind("<Return>", lambda e: self.gui_send_text())

        btn_send = tk.Button(bottom_frame, text="Send", command=self.gui_send_text)
        btn_send.pack(side=tk.RIGHT)

    def log(self, text):
        self.log_area['state'] = 'normal'
        t = datetime.now().strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{t}] {text}\n")
        self.log_area.see(tk.END)
        self.log_area['state'] = 'disabled'

    def poll_queue(self):
        try:
            while True:
                item = task_q.get_nowait()
                # item is a tuple (type, payload)
                typ, payload = item
                if typ == "log":
                    self.log(payload)
                elif typ == "speak":
                    speak(payload)
                    self.log("Spoke: " + payload)
                elif typ == "text":
                    # text is a result from command handler
                    self.log("Cmd: " + payload)
                else:
                    self.log("Unknown queue item: " + str(item))
        except queue.Empty:
            pass
        self.root.after(200, self.poll_queue)

    def action_face_login(self):
        # run face_login non-blocking
        t = threading.Thread(target=self._face_login_thread, daemon=True)
        t.start()

    def _face_login_thread(self):
        name = face_login()
        if name:
            self.log(f"Face login: {name}")
        else:
            self.log("Face login failed.")

    def toggle_listen(self):
        if not self.listening:
            self.listening = True
            self.voice_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.voice_thread.start()
            self.log("Started listening...")
            speak("I am listening.")
        else:
            self.listening = False
            self.log("Stopped listening.")
            speak("Stopped listening.")

    def _listen_loop(self):
        while self.listening:
            txt = listen(timeout=6, phrase_time_limit=8)
            if txt:
                self.log("You said: " + txt)
                res = handle_command(txt)
                self.log("Jarvis: " + (res if res else ""))
            time.sleep(0.4)

    def show_notes(self):
        notes = db_get_notes()
        win = tk.Toplevel(self.root)
        win.title("Notes")
        txt = scrolledtext.ScrolledText(win, wrap=tk.WORD)
        for n in notes:
            txt.insert(tk.END, f"--- {n[1]} ({n[3]}) ---\n{n[2]}\n\n")
        txt.pack(expand=True, fill=tk.BOTH)

    def show_reminders(self):
        rows = db_get_pending_reminders()
        win = tk.Toplevel(self.root)
        win.title("Pending Reminders")
        for r in rows:
            label = tk.Label(win, text=f"{r[0]} - {r[1]} at {r[2]}")
            label.pack(anchor='w')

    def gui_fetch_news(self):
        headlines = fetch_news_headlines()
        if headlines:
            for h in headlines:
                self.log("Headline: " + h)
                speak(h, block=False)
        else:
            self.log("News fetch failed.")
            speak("News fetch failed.")

    def gui_get_weather(self):
        # ask via simple dialog
        city = simple_input_dialog(self.root, "City", "Enter city for weather (e.g., New Delhi):")
        if city:
            res = get_weather(city)
            self.log(res)
            speak(res)

    def gui_capture(self):
        path = capture_image()
        if path:
            self.log("Captured: " + path)
            # show small preview
            win = tk.Toplevel(self.root)
            win.title("Photo")
            img = Image.open(path)
            img.thumbnail((400, 400))
            tkimg = ImageTk.PhotoImage(img)
            l = tk.Label(win, image=tkimg)
            l.image = tkimg
            l.pack()

    def gui_send_text(self):
        txt = self.entry.get().strip()
        self.entry.delete(0, tk.END)
        if not txt:
            return
        self.log("You: " + txt)
        # treat as a command
        res = handle_command(txt)
        self.log("Jarvis: " + (res if res else ""))

# --- Tiny GUI helpers ----------------------------------------------------

def simple_input_dialog(parent, title, prompt):
    win = tk.Toplevel(parent)
    win.title(title)
    tk.Label(win, text=prompt).pack(padx=6, pady=6)
    e = tk.Entry(win)
    e.pack(padx=6, pady=6)
    result = {"value": None}
    def on_ok():
        result["value"] = e.get().strip()
        win.destroy()
    btn = tk.Button(win, text="OK", command=on_ok)
    btn.pack(pady=4)
    parent.wait_window(win)
    return result["value"]

# --- Small visualization example ----------------------------------------

def plot_sample_usage():
    con = sqlite3.connect(DB_PATH)
    c = con.cursor()
    c.execute('SELECT action, COUNT(*) FROM usage_log GROUP BY action ORDER BY COUNT(*) DESC LIMIT 10')
    rows = c.fetchall()
    con.close()
    if not rows:
        log("No usage data to plot.")
        return
    actions = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    plt.figure(figsize=(8,4))
    plt.bar(actions, counts)
    plt.title("Top Jarvis Actions")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# --- Startup & Main -----------------------------------------------------

def initial_greeting():
    speak(f"Hello! I am {APP_NAME}. Initializing.", block=False)

def background_scheduler():
    t = threading.Thread(target=start_scheduler_loop, daemon=True)
    t.start()

def cli_listen_loop():
    """Simple CLI loop for those not using GUI."""
    print("Jarvis CLI. Type 'exit' to quit.")
    while True:
        try:
            text = input("You: ")
        except EOFError:
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            break
        res = handle_command(text)
        print("Jarvis:", res)

# --- Entrypoint ----------------------------------------------------------

def main(use_gui=True):
    init_db()
    initial_greeting()
    background_scheduler()
    # start GUI if asked
    if use_gui:
        root = tk.Tk()
        gui = JarvisGUI(root)
        # start a small thread to show usage plot occasionally (disabled by default)
        root.protocol("WM_DELETE_WINDOW", root.quit)
        root.mainloop()
    else:
        cli_listen_loop()

if __name__ == "__main__":
    # Launch GUI by default
    main(use_gui=True)
