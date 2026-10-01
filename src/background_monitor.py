# ============================================================
# EGATE AI - UNIFIED BACKGROUND MONITOR
# ============================================================
#
# This SINGLE program handles:
#
# 1. NSFW screen detection
# 2. YouTube intervention video
# 3. Addiction-risk prediction using your trained XGBoost model
# 4. Telegram family alerts
# 5. Telegram registration code
# 6. 5-minute intervention/cooldown
#
# IMPORTANT:
# - m.csv is used to TRAIN the model.
# - addiction_pipeline.pkl is used by this program.
# - The addiction model does NOT use screenshots.
# - The NSFW model DOES use screenshots.
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
# ============================================================
# ENVIRONMENT VARIABLES
# MUST BE SET BEFORE IMPORTING TRANSFORMERS
# ============================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import json
import secrets
import threading
import subprocess

import joblib
import pandas as pd
import requests





# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================

from PIL import ImageGrab
from transformers.pipelines import pipeline


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/Users/mo/Egate_Ai_camp"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "addiction_pipeline.pkl"
)

PROFILE_PATH = os.path.join(
    BASE_DIR,
    "user_profile.json"
)

USAGE_PATH = os.path.join(
    BASE_DIR,
    "current_usage.json"
)

TELEGRAM_AUTH_PATH = os.path.join(
    BASE_DIR,
    "telegram_authorized.json"
)

# Written by the Streamlit dashboard when the user submits a code there.
TELEGRAM_CODE_SUBMIT_PATH = os.path.join(
    BASE_DIR,
    "telegram_code_submit.json"
)

# Written by this monitor so the dashboard can show the pairing result.
TELEGRAM_PAIRING_STATUS_PATH = os.path.join(
    BASE_DIR,
    "telegram_pairing_status.json"
)

LOG_FILE = os.path.join(
    BASE_DIR,
    "monitor_debug.log"
)


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

REGISTRATION_CODE_LIFETIME = 600  # 10 minutes


# ============================================================
# NSFW SETTINGS
# ============================================================

NSFW_THRESHOLD = 0.10

CHECK_INTERVAL_SECONDS = 3

# Prevent repeatedly triggering the same event.
NSFW_COOLDOWN_SECONDS = 30


# ============================================================
# ADDICTION SETTINGS
# ============================================================

# Check the addiction model every 5 minutes.
ADDICTION_CHECK_INTERVAL = 300

# Telegram alert threshold.
ADDICTION_HIGH_RISK_THRESHOLD = 0.10


# ============================================================
# INTERVENTION
# ============================================================

LOCKOUT_DURATION = 300  # 5 minutes

YOUTUBE_URL = (
    "https://www.youtube.com/watch?v=1ZYbU82GVz4"
)


# ============================================================
# GLOBAL STATE
# ============================================================

authorized_chat_id = None

pending_codes = {}

telegram_lock = threading.Lock()

last_nsfw_alert_time = 0

last_addiction_alert_time = 0

monitor_running = True


# ============================================================
# LOGGING
# ============================================================

def log(message):

    try:

        with open(
            LOG_FILE,
            "a",
            encoding="utf-8"
        ) as log_f:

            timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            log_f.write(
                f"[{timestamp}] {message}\n"
            )

            log_f.flush()

    except Exception:

        pass


# ============================================================
# TELEGRAM AUTH STORAGE
# ============================================================

def load_authorized_chat():

    global authorized_chat_id

    try:

        with open(
            TELEGRAM_AUTH_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        authorized_chat_id = data.get(
            "chat_id"
        )

        if authorized_chat_id:

            log(
                f"Authorized Telegram account loaded: "
                f"{authorized_chat_id}"
            )

    except Exception:

        authorized_chat_id = None


def save_authorized_chat(chat_id):

    global authorized_chat_id

    try:

        with open(
            TELEGRAM_AUTH_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                {
                    "chat_id": chat_id
                },
                f,
                indent=2
            )

        authorized_chat_id = chat_id

        log(
            f"Telegram account authorized: {chat_id}"
        )

    except Exception as e:

        log(
            f"Could not save Telegram authorization: {e}"
        )


load_authorized_chat()


# ============================================================
# TELEGRAM API
# ============================================================

def telegram_api(
    method,
    data=None
):

    if not TELEGRAM_TOKEN:

        return None

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/{method}"
    )

    try:

        response = requests.post(
            url,
            data=data or {},
            timeout=30
        )

        return response.json()

    except Exception as e:

        log(
            f"Telegram API error: {e}"
        )

        return None


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def send_telegram_message(
    chat_id,
    message
):

    if not chat_id:

        return False

    result = telegram_api(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": message
        }
    )

    if result and result.get("ok"):

        log(
            f"Telegram message sent to {chat_id}"
        )

        return True

    log(
        f"Telegram send failed: {result}"
    )

    return False


# ============================================================
# TELEGRAM REGISTRATION CODE
# ============================================================

def generate_registration_code():

    return f"{secrets.randbelow(1000000):06d}"


# ============================================================
# TELEGRAM LISTENER
# ============================================================

def telegram_listener():

    log(
        "Telegram listener started."
    )

    # Make sure polling works.
    result = telegram_api(
        "deleteWebhook",
        {
            "drop_pending_updates": False
        }
    )

    if result and result.get("ok"):

        log(
            "Telegram webhook disabled."
        )

    # Start from newest update.
    initial = telegram_api(
        "getUpdates"
    )

    offset = 0

    if initial and initial.get("ok"):

        updates = initial.get(
            "result",
            []
        )

        if updates:

            offset = (
                updates[-1]["update_id"] + 1
            )


    while monitor_running:

        try:

            result = telegram_api(
                "getUpdates",
                {
                    "offset": offset,
                    "timeout": 20
                }
            )

            if not result or not result.get("ok"):

                time.sleep(3)

                continue


            updates = result.get(
                "result",
                []
            )


            for update in updates:

                offset = (
                    update["update_id"] + 1
                )

                message = update.get(
                    "message"
                )

                if not message:

                    continue

                text = (
                    message
                    .get("text", "")
                    .strip()
                    .lower()
                )

                chat = message.get(
                    "chat",
                    {}
                )

                chat_id = chat.get(
                    "id"
                )

                if not chat_id:

                    continue


                # ==================================================
                # /start
                # ==================================================

                if text == "/start":

                    code = (
                        generate_registration_code()
                    )

                    expires = (
                        time.time()
                        + REGISTRATION_CODE_LIFETIME
                    )

                    with telegram_lock:

                        pending_codes[code] = {
                            "chat_id": chat_id,
                            "expires": expires
                        }


                    send_telegram_message(
                        chat_id,
                        (
                            "👋 Welcome to Egate AI.\n\n"
                            "Your computer registration code is:\n\n"
                            f"🔐 {code}\n\n"
                            "Enter this 6-digit code "
                            "on the computer running "
                            "Egate AI.\n\n"
                            "⏱ The code expires in "
                            "10 minutes."
                        )
                    )

                    log(
                        f"Registration code generated "
                        f"for chat {chat_id}: {code}"
                    )


                # ==================================================
                # /newcode
                # ==================================================

                elif text == "/newcode":

                    code = (
                        generate_registration_code()
                    )

                    expires = (
                        time.time()
                        + REGISTRATION_CODE_LIFETIME
                    )

                    with telegram_lock:

                        pending_codes[code] = {
                            "chat_id": chat_id,
                            "expires": expires
                        }


                    send_telegram_message(
                        chat_id,
                        (
                            "🔐 New registration code:\n\n"
                            f"{code}\n\n"
                            "This code expires in "
                            "10 minutes."
                        )
                    )


                # ==================================================
                # /status
                # ==================================================

                elif text == "/status":

                    if authorized_chat_id == chat_id:

                        message_text = (
                            "✅ This Telegram account "
                            "is paired with Egate AI."
                        )

                    else:

                        message_text = (
                            "⚠️ This Telegram account "
                            "is not paired."
                        )

                    send_telegram_message(
                        chat_id,
                        message_text
                    )


                # ==================================================
                # /test
                # ==================================================

                elif text == "/test":

                    if authorized_chat_id == chat_id:

                        send_telegram_message(
                            chat_id,
                            (
                                "🟢 Egate AI test successful.\n\n"
                                "Telegram alerts are working."
                            )
                        )

                    else:

                        send_telegram_message(
                            chat_id,
                            (
                                "⚠️ This account is not "
                                "paired with the computer."
                            )
                        )

        except Exception as e:

            log(
                f"Telegram listener error: {e}"
            )

            time.sleep(5)


# ============================================================
# REGISTER COMPUTER WITH FAMILY TELEGRAM
# ============================================================

def _write_pairing_status(status, message):

    try:

        with open(
            TELEGRAM_PAIRING_STATUS_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                {
                    "status": status,
                    "message": message,
                    "updated_at": time.time()
                },
                f,
                indent=2
            )

    except Exception as e:

        log(
            f"Could not write pairing status: {e}"
        )


def check_code_submission():
    """
    Non-blocking replacement for the old terminal-based
    register_computer(). Looks for a code the Streamlit
    dashboard wrote to TELEGRAM_CODE_SUBMIT_PATH, validates
    it against pending_codes, and reports the result back
    through TELEGRAM_PAIRING_STATUS_PATH. Always deletes the
    submit file after handling it so it's never reprocessed.
    """

    if not os.path.exists(TELEGRAM_CODE_SUBMIT_PATH):

        return

    try:

        with open(
            TELEGRAM_CODE_SUBMIT_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            submitted = json.load(f)

        code = str(
            submitted.get("code", "")
        ).strip()

    except Exception as e:

        log(
            f"Could not read submitted Telegram code: {e}"
        )

        try:
            os.remove(TELEGRAM_CODE_SUBMIT_PATH)
        except Exception:
            pass

        return

    # Always remove the request file so a stale/duplicate
    # copy is never picked up again on the next poll.
    try:
        os.remove(TELEGRAM_CODE_SUBMIT_PATH)
    except Exception:
        pass

    if len(code) != 6 or not code.isdigit():

        _write_pairing_status(
            "error",
            "Enter exactly 6 digits."
        )

        return

    with telegram_lock:

        registration = pending_codes.get(code)

    if not registration:

        _write_pairing_status(
            "error",
            "Invalid registration code. Send /start to the "
            "bot again for a new one."
        )

        return

    if time.time() > registration["expires"]:

        with telegram_lock:

            pending_codes.pop(code, None)

        _write_pairing_status(
            "error",
            "That code expired. Send /start to the bot again "
            "for a new one."
        )

        return

    chat_id = registration["chat_id"]

    save_authorized_chat(chat_id)

    with telegram_lock:

        pending_codes.pop(code, None)

    send_telegram_message(
        chat_id,
        (
            "✅ COMPUTER PAIRED\n\n"
            "This Telegram account is now "
            "connected to Egate AI.\n\n"
            "You will receive safety alerts "
            "from the computer."
        )
    )

    _write_pairing_status(
        "success",
        "Computer paired successfully."
    )

    log(
        f"Telegram pairing completed via dashboard for chat_id {chat_id}"
    )


def code_submission_watcher():
    """
    Runs forever in its own daemon thread, polling for a code
    the dashboard has submitted. This is what lets pairing
    happen from the Streamlit app instead of this script's
    terminal — and it keeps working at any point while the
    monitor is running, not just at startup.
    """

    log("Telegram code-submission watcher started.")

    while True:

        try:

            check_code_submission()

        except Exception as e:

            log(
                f"Code-submission watcher error: {e}"
            )

        time.sleep(1)


        print()
        print(
            "✅ Computer successfully paired!"
        )

        return True


# ============================================================
# LOAD USER PROFILE
# ============================================================

def load_user_profile():

    if not os.path.exists(
        PROFILE_PATH
    ):

        raise FileNotFoundError(
            f"Missing:\n{PROFILE_PATH}"
        )


    with open(
        PROFILE_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        profile = json.load(f)


    return profile


# ============================================================
# LOAD CURRENT MAC USAGE
# ============================================================

def load_usage_data():

    if not os.path.exists(
        USAGE_PATH
    ):

        raise FileNotFoundError(
            f"Missing:\n{USAGE_PATH}\n\n"
            "The macOS usage collector has not "
            "created the file yet."
        )


    with open(
        USAGE_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        usage = json.load(f)


    return usage


# ============================================================
# LOAD ADDICTION MODEL
# ============================================================

def load_addiction_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Missing model:\n{MODEL_PATH}"
        )


    artifact = joblib.load(
        MODEL_PATH
    )


    preprocessor = artifact[
        "preprocessor"
    ]

    model = artifact[
        "model"
    ]

    feature_columns = artifact[
        "feature_columns"
    ]


    log(
        "Addiction model loaded successfully."
    )


    return (
        preprocessor,
        model,
        feature_columns
    )


# ============================================================
# BUILD ADDICTION MODEL FEATURES
# ============================================================

def build_addiction_features(
    profile,
    usage,
    feature_columns
):

    features = {

        # ----------------------------------------------
        # From macOS usage collector
        # ----------------------------------------------

        "daily_screen_time_hours":
            usage.get(
                "daily_screen_time_hours",
                0.0
            ),

        "social_media_hours":
            usage.get(
                "social_media_hours",
                0.0
            ),

        "gaming_hours":
            usage.get(
                "gaming_hours",
                0.0
            ),

        "notifications_per_day":
            usage.get(
                "notifications_per_day",
                0
            ),

        "app_opens_per_day":
            usage.get(
                "app_opens_per_day",
                0
            ),

        "weekend_screen_time":
            usage.get(
                "weekend_screen_time",
                profile.get(
                    "weekend_screen_time",
                    0.0
                )
            ),


        # ----------------------------------------------
        # From user profile
        # ----------------------------------------------

        "age":
            profile.get(
                "age"
            ),

        "gender":
            profile.get(
                "gender"
            ),

        "work_study_hours":
            profile.get(
                "work_study_hours",
                0.0
            ),

        "sleep_hours":
            profile.get(
                "sleep_hours",
                7.0
            ),

        "stress_level":
            profile.get(
                "stress_level",
                "Medium"
            ),

        "academic_work_impact":
            profile.get(
                "academic_work_impact",
                "No"
            ),
    }


    # ========================================================
    # CHECK MODEL FEATURES
    # ========================================================

    missing = []

    for column in feature_columns:

        value = features.get(
            column
        )

        if value is None:

            missing.append(
                column
            )


    if missing:

        raise ValueError(
            "Missing model features: "
            + ", ".join(missing)
        )


    return pd.DataFrame(
        [features],
        columns=feature_columns
    )


# ============================================================
# RUN ADDICTION MODEL
# ============================================================

def predict_addiction(
    preprocessor,
    model,
    feature_columns
):

    profile = load_user_profile()

    usage = load_usage_data()


    X = build_addiction_features(
        profile,
        usage,
        feature_columns
    )


    X_encoded = (
        preprocessor.transform(
            X
        )
    )


    probability = float(
        model.predict_proba(
            X_encoded
        )[0][1]
    )


    prediction = int(
        probability >= 0.50
    )


    return (
        prediction,
        probability,
        X.iloc[0].to_dict()
    )


# ============================================================
# ADDICTION TELEGRAM ALERT
# ============================================================

def send_addiction_alert(
    probability,
    features
):

    global last_addiction_alert_time


    if not authorized_chat_id:

        log(
            "Addiction alert skipped: "
            "no Telegram account paired."
        )

        return


    now = time.time()


    # Prevent alert spam.
    if (
        now - last_addiction_alert_time
        < ADDICTION_CHECK_INTERVAL
    ):

        return


    message = (
        "⚠️ EGATE AI USAGE ALERT\n\n"

        f"Estimated model risk: "
        f"{probability:.1%}\n\n"

        f"Daily screen time: "
        f"{features['daily_screen_time_hours']:.1f} h\n"

        f"Social media: "
        f"{features['social_media_hours']:.1f} h\n"

        f"Gaming: "
        f"{features['gaming_hours']:.1f} h\n"

        f"Work/study: "
        f"{features['work_study_hours']:.1f} h\n"

        f"Sleep: "
        f"{features['sleep_hours']:.1f} h\n"

        f"Notifications: "
        f"{features['notifications_per_day']}\n"

        f"App opens: "
        f"{features['app_opens_per_day']}\n\n"

        "Please review the usage pattern."
    )


    if send_telegram_message(
        authorized_chat_id,
        message
    ):

        last_addiction_alert_time = now


# ============================================================
# NSFW DETECTION MODEL
# ============================================================

def load_nsfw_model():

    log(
        "Loading NSFW detection model..."
    )


    classifier = pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=-1
    )


    log(
        "NSFW detection model loaded."
    )


    return classifier


# ============================================================
# DETECT NSFW
# ============================================================

def get_nsfw_score(
    classifier,
    screen
):

    results = classifier(
        screen
    )


    nsfw_score = 0.0


    for item in results:

        label = str(
            item.get(
                "label",
                ""
            )
        ).lower()


        if label == "nsfw":

            nsfw_score = float(
                item.get(
                    "score",
                    0.0
                )
            )

            break


    return nsfw_score


# ============================================================
# NSFW TELEGRAM ALERT
# ============================================================

def send_nsfw_alert(
    score
):

    global last_nsfw_alert_time


    now = time.time()


    # Prevent repeated Telegram alerts.
    if (
        now - last_nsfw_alert_time
        < NSFW_COOLDOWN_SECONDS
    ):

        return


    if not authorized_chat_id:

        log(
            "NSFW detected but no Telegram "
            "account is paired."
        )

        return


    message = (
        "⚠️ SAFETY ALERT\n\n"
        "Potentially inappropriate content "
        "was detected on the child's screen.\n\n"
        f"Detection score: {score:.2f}\n\n"
        "A safety intervention has been started."
    )


    if send_telegram_message(
        authorized_chat_id,
        message
    ):

        last_nsfw_alert_time = now


# ============================================================
# OPEN YOUTUBE ONCE
# ============================================================

def open_youtube():

    log(
        "Opening YouTube intervention video."
    )


    try:

        subprocess.Popen(
            [
                "open",
                YOUTUBE_URL
            ]
        )


        log(
            "YouTube opened successfully."
        )


    except Exception as e:

        log(
            f"Could not open YouTube: {e}"
        )


# ============================================================
# INTERVENTION
# ============================================================

def enforce_nsfw_intervention(
    score
):

    log(
        f"NSFW intervention started. "
        f"Score={score:.2f}"
    )


    # --------------------------------------------------------
    # 1. Telegram
    # --------------------------------------------------------

    send_nsfw_alert(
        score
    )


    # --------------------------------------------------------
    # 2. Open YouTube ONLY ONCE
    # --------------------------------------------------------

    open_youtube()


    # --------------------------------------------------------
    # 3. No pmset.
    #
    # We DO NOT repeatedly put the display to sleep.
    # --------------------------------------------------------

    log(
        "5-minute NSFW intervention cooldown started."
    )


    end_time = (
        time.time()
        + LOCKOUT_DURATION
    )


    while (
        time.time() < end_time
        and monitor_running
    ):

        time.sleep(1)


    log(
        "NSFW intervention cooldown finished."
    )


# ============================================================
# ADDICTION MODEL BACKGROUND WORKER
# ============================================================

def addiction_monitor_worker(
    preprocessor,
    model,
    feature_columns
):

    global monitor_running


    while monitor_running:

        try:

            (
                prediction,
                probability,
                features
            ) = predict_addiction(
                preprocessor,
                model,
                feature_columns
            )


            log(
                "Addiction prediction: "
                f"{probability:.4f}"
            )


            print(
                f"🧠 Addiction risk: "
                f"{probability:.1%}"
            )


            if (
                probability
                >= ADDICTION_HIGH_RISK_THRESHOLD
            ):

                send_addiction_alert(
                    probability,
                    features
                )


        except Exception as e:

            log(
                f"Addiction monitor error: {e}"
            )

            print(
                f"⚠️ Addiction monitor: {e}"
            )


        # ----------------------------------------------------
        # Check every 5 minutes.
        # ----------------------------------------------------

        for _ in range(
            ADDICTION_CHECK_INTERVAL
        ):

            if not monitor_running:

                break

            time.sleep(1)


# ============================================================
# STARTUP
# ============================================================

def main():

    global monitor_running


    print()
    print("=" * 60)
    print("             EGATE AI MONITOR")
    print("=" * 60)
    print()


    log("")
    log("=" * 60)
    log("EGATE AI MONITOR STARTING")
    log("=" * 60)


    # ========================================================
    # TELEGRAM TOKEN
    # ========================================================

    if not TELEGRAM_TOKEN:

        print(
            "❌ TELEGRAM_BOT_TOKEN is not set."
        )

        print()
        print(
            'Run:'
        )

        print(
            'export TELEGRAM_BOT_TOKEN="YOUR_TOKEN"'
        )

        return


    # ========================================================
    # TELEGRAM CONNECTION
    # ========================================================

    info = telegram_api(
        "getMe"
    )


    if not info or not info.get("ok"):

        print(
            "❌ Telegram connection failed."
        )

        log(
            f"Telegram connection failed: {info}"
        )

        return


    bot_username = (
        info["result"]
        .get(
            "username",
            "unknown"
        )
    )


    print(
        f"✅ Telegram bot: @{bot_username}"
    )


    # ========================================================
    # START TELEGRAM LISTENER
    # ========================================================

    telegram_thread = threading.Thread(
        target=telegram_listener,
        daemon=True
    )

    telegram_thread.start()


    # Give listener time to start.
    time.sleep(2)


    # ========================================================
    # REGISTRATION
    # ========================================================
    # Pairing now happens from the Streamlit dashboard instead
    # of this terminal. This thread just watches for a code the
    # dashboard writes to disk, at any point while the monitor
    # is running — not only at startup.

    code_watcher_thread = threading.Thread(
        target=code_submission_watcher,
        daemon=True
    )

    code_watcher_thread.start()

    if authorized_chat_id:

        print(
            "✅ Existing Telegram pairing found."
        )

    else:

        print(
            "⚠️ Telegram account is not paired yet."
        )

        print(
            "Open the Egate AI dashboard's 'Pair Telegram' "
            "section to finish pairing — the monitor will keep "
            "running and pick it up automatically."
        )


    # ========================================================
    # LOAD ADDICTION MODEL
    # ========================================================

    print()
    print(
        "Loading addiction model..."
    )


    try:

        (
            preprocessor,
            addiction_model,
            feature_columns
        ) = load_addiction_model()


        print(
            "✅ Addiction model loaded."
        )


        print()
        print(
            "Model features:"
        )


        for column in feature_columns:

            print(
                " -",
                column
            )


    except Exception as e:

        print(
            f"❌ Could not load addiction model: {e}"
        )

        log(
            f"Addiction model loading error: {e}"
        )

        return


    # ========================================================
    # START ADDICTION WORKER
    # ========================================================

    addiction_thread = threading.Thread(
        target=addiction_monitor_worker,
        args=(
            preprocessor,
            addiction_model,
            feature_columns
        ),
        daemon=True
    )

    addiction_thread.start()


    # ========================================================
    # LOAD NSFW MODEL
    # ========================================================

    print()
    print(
        "Loading NSFW detection model..."
    )


    try:

        nsfw_classifier = (
            load_nsfw_model()
        )


        print(
            "✅ NSFW model loaded."
        )


    except Exception as e:

        print(
            f"❌ NSFW model failed: {e}"
        )

        log(
            f"NSFW model loading error: {e}"
        )

        return


    # ========================================================
    # READY
    # ========================================================

    print()
    print("=" * 60)
    print("🟢 EGATE AI IS RUNNING")
    print("=" * 60)
    print()
    print(
        "NSFW monitoring: ON"
    )

    print(
        "Addiction monitoring: ON"
    )

    print(
        "Telegram alerts: "
        + (
            "ON"
            if authorized_chat_id
            else "OFF"
        )
    )

    print()
    print(
        "Press Control+C to stop."
    )
    print()


    log(
        "Egate AI monitoring started."
    )


    # ========================================================
    # MAIN NSFW LOOP
    # ========================================================

    try:

        while monitor_running:

            # -----------------------------------------------
            # Capture screenshot
            # -----------------------------------------------

            screen = (
                ImageGrab
                .grab()
                .convert("RGB")
            )


            # -----------------------------------------------
            # NSFW model
            # -----------------------------------------------

            nsfw_score = get_nsfw_score(
                nsfw_classifier,
                screen
            )


            log(
                f"Screen scanned. "
                f"NSFW score={nsfw_score:.4f}"
            )


            # -----------------------------------------------
            # NSFW detected
            # -----------------------------------------------

            if nsfw_score >= NSFW_THRESHOLD:

                print(
                    f"⚠️ NSFW detected: "
                    f"{nsfw_score:.2f}"
                )


                enforce_nsfw_intervention(
                    nsfw_score
                )


            time.sleep(
                CHECK_INTERVAL_SECONDS
            )


    except KeyboardInterrupt:

        print()
        print(
            "🛑 Stopping Egate AI..."
        )


        log(
            "--- Egate AI stopped by user ---"
        )


        monitor_running = False


    except Exception as e:

        print(
            f"❌ Main monitor error: {e}"
        )


        log(
            f"Main monitor error: {e}"
        )


        monitor_running = False


    print(
        "✅ Egate AI stopped."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()