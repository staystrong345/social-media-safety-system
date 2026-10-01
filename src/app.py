
import os 
import json
import subprocess
from datetime import datetime

import joblib
import pandas as pd
import requests
import streamlit as st


# ============================================================
# EGATE AI — MAIN WEB DASHBOARD
# ============================================================
#
# This dashboard controls and displays the Egate AI system.
#
# ENGINE:
#   background_monitor.py
#       - NSFW screenshot monitoring
#       - addiction model
#       - Telegram alerts
#       - intervention
#
# MODEL:
#   addiction_pipeline.pkl
#       - trained from m.csv
#
# USERS:
#   user_profiles.json
#   user_usage.json
#
# The dashboard does NOT retrain the model.
# It loads the trained .pkl and supplies the selected user's
# profile + usage features.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/Users/mo/Egate_Ai_camp"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "addiction_pipeline.pkl",
)

PROFILE_PATH = os.path.join(
    BASE_DIR,
    "user_profile.json",
)

PROFILES_PATH = os.path.join(
    BASE_DIR,
    "user_profiles.json",
)

USAGE_PATH = os.path.join(
    BASE_DIR,
    "current_usage.json",
)

USAGES_PATH = os.path.join(
    BASE_DIR,
    "user_usage.json",
)

ACTIVE_PROFILE_PATH = os.path.join(
    BASE_DIR,
    "active_profile.json",
)

TELEGRAM_AUTH_PATH = os.path.join(
    BASE_DIR,
    "telegram_authorized.json",
)

LOG_FILE = os.path.join(
    BASE_DIR,
    "monitor_debug.log",
)

MONITOR_PATH = os.path.join(
    BASE_DIR,
    "background_monitor.py",
)

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Nip",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            rgba(52, 152, 219, 0.16),
            rgba(155, 89, 182, 0.12)
        );
        border: 1px solid rgba(127, 127, 127, 0.20);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        opacity: 0.75;
        font-size: 1rem;
    }

    .section-card {
        padding: 1rem 1.15rem;
        border-radius: 14px;
        border: 1px solid rgba(127, 127, 127, 0.18);
        margin-bottom: 1rem;
    }

    .small-muted {
        opacity: 0.65;
        font-size: 0.86rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(127, 127, 127, 0.16);
        border-radius: 12px;
        padding: 0.6rem;
    }

    .risk-good {
        padding: 0.8rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(46, 204, 113, 0.35);
    }

    .risk-mid {
        padding: 0.8rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(241, 196, 15, 0.35);
    }

    .risk-high {
        padding: 0.8rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(231, 76, 60, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    if not os.path.exists(path):
        return None

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path, data):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=4,
        )


def normalize_profile(profile):
    return {
        "age": int(profile.get("age", 18)),
        "gender": profile.get("gender", "Male"),
        "work_study_hours": float(
            profile.get(
                "work_study_hours",
                4.0,
            )
        ),
        "sleep_hours": float(
            profile.get(
                "sleep_hours",
                7.0,
            )
        ),
        "stress_level": profile.get(
            "stress_level",
            "Medium",
        ),
        "academic_work_impact": profile.get(
            "academic_work_impact",
            "No",
        ),
        "weekend_screen_time": float(
            profile.get(
                "weekend_screen_time",
                0.0,
            )
        ),
    }


def load_profiles():
    data = load_json(PROFILES_PATH)

    if isinstance(data, dict):
        return data

    # Import the old single-profile file once.
    legacy = load_json(PROFILE_PATH)

    if isinstance(legacy, dict):
        profiles = {
            "Default User":
                normalize_profile(legacy)
        }
        save_json(
            PROFILES_PATH,
            profiles,
        )
        return profiles

    return {}


def save_profiles(profiles):
    save_json(
        PROFILES_PATH,
        profiles,
    )


def load_user_usages():
    data = load_json(USAGES_PATH)

    if isinstance(data, dict):
        return data

    return {}


def save_user_usages(usages):
    save_json(
        USAGES_PATH,
        usages,
    )


def get_active_profile_name():
    data = load_json(
        ACTIVE_PROFILE_PATH
    )

    if isinstance(data, dict):
        return data.get("name")

    return None


def set_active_profile_name(name):
    save_json(
        ACTIVE_PROFILE_PATH,
        {"name": name},
    )


def activate_profile(name, profile):
    # Legacy compatibility for background_monitor.py.
    save_json(
        PROFILE_PATH,
        normalize_profile(profile),
    )


def get_selected_usage(name):
    usages = load_user_usages()
    usage = usages.get(name)

    if isinstance(usage, dict):
        return usage

    # Legacy compatibility.
    legacy = load_json(USAGE_PATH)

    if isinstance(legacy, dict):
        return legacy

    return None


def save_selected_usage(name, usage):
    usages = load_user_usages()
    usages[name] = usage
    save_user_usages(usages)

    # Keep legacy file synchronized.
    save_json(
        USAGE_PATH,
        usage,
    )


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None


# ============================================================
# TELEGRAM
# ============================================================

def get_telegram_chat_id():
    data = load_json(
        TELEGRAM_AUTH_PATH
    )

    if isinstance(data, dict):
        return data.get("chat_id")

    return None


def telegram_request(
    method,
    data=None,
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
            timeout=10,
        )
        return response.json()
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }


def send_telegram_test():
    chat_id = get_telegram_chat_id()

    if not chat_id:
        return (
            False,
            "No family Telegram account is paired.",
        )

    result = telegram_request(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": (
                "🟢  TEST\n\n"
                "Your Telegram connection is working."
            ),
        },
    )

    if result and result.get("ok"):
        return (
            True,
            "Telegram test sent successfully.",
        )

    return (
        False,
        f"Telegram error: {result}",
    )


# ============================================================
# MONITOR CONTROL
# ============================================================

def monitor_is_running():
    try:
        result = subprocess.run(
            [
                "pgrep",
                "-f",
                "background_monitor.py",
            ],
            capture_output=True,
            text=True,
        )

        return bool(
            result.stdout.strip()
        )

    except Exception:
        return False


def start_background_monitor():
    if not os.path.exists(MONITOR_PATH):
        return (
            False,
            f"Monitor not found:\n{MONITOR_PATH}",
        )

    if monitor_is_running():
        return (
            True,
            "Background monitor is already running.",
        )

    try:
        subprocess.Popen(
            [
                "python3",
                MONITOR_PATH,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        return (
            True,
            "Background monitor started.",
        )

    except Exception as e:
        return (
            False,
            f"Could not start monitor: {e}",
        )


def stop_background_monitor():
    try:
        subprocess.run(
            [
                "pkill",
                "-f",
                "background_monitor.py",
            ],
            capture_output=True,
        )

        return (
            True,
            "Background monitor stopped.",
        )

    except Exception as e:
        return (
            False,
            f"Could not stop monitor: {e}",
        )


# ============================================================
# PREDICTION
# ============================================================

def run_prediction(
    profile,
    usage,
):
    artifact = load_model()

    if artifact is None:
        return {
            "error":
                "addiction_pipeline.pkl was not found.",
        }

    if profile is None:
        return {
            "error":
                "No user profile is selected.",
        }

    if usage is None:
        return {
            "error":
                "No usage data is available for this user.",
        }

    preprocessor = artifact[
        "preprocessor"
    ]

    model = artifact[
        "model"
    ]

    feature_columns = artifact[
        "feature_columns"
    ]

    features = {
        "age":
            profile.get("age"),

        "gender":
            profile.get("gender"),

        "daily_screen_time_hours":
            usage.get(
                "daily_screen_time_hours",
                0.0,
            ),

        "social_media_hours":
            usage.get(
                "social_media_hours",
                0.0,
            ),

        "gaming_hours":
            usage.get(
                "gaming_hours",
                0.0,
            ),

        "work_study_hours":
            profile.get(
                "work_study_hours",
                0.0,
            ),

        "sleep_hours":
            profile.get(
                "sleep_hours",
                7.0,
            ),

        "notifications_per_day":
            usage.get(
                "notifications_per_day",
                0,
            ),

        "app_opens_per_day":
            usage.get(
                "app_opens_per_day",
                0,
            ),

        "weekend_screen_time":
            usage.get(
                "weekend_screen_time",
                profile.get(
                    "weekend_screen_time",
                    0.0,
                ),
            ),

        "stress_level":
            profile.get(
                "stress_level",
                "Medium",
            ),

        "academic_work_impact":
            profile.get(
                "academic_work_impact",
                "No",
            ),
    }

    missing = [
        column
        for column in feature_columns
        if features.get(column) is None
    ]

    if missing:
        return {
            "error":
                "Missing model features: "
                + ", ".join(missing),
        }

    X = pd.DataFrame(
        [features],
        columns=feature_columns,
    )

    try:
        X_processed = (
            preprocessor.transform(X)
        )

        probability = float(
            model.predict_proba(
                X_processed
            )[0][1]
        )

        prediction = int(
            probability >= 0.50
        )

        return {
            "prediction": prediction,
            "probability": probability,
            "features": features,
        }

    except Exception as e:
        return {
            "error":
                f"Prediction failed: {e}",
        }


# ============================================================
# LOAD STATE
# ============================================================

profiles = load_profiles()

active_profile_name = (
    get_active_profile_name()
)

if (
    active_profile_name not in profiles
    and profiles
):
    active_profile_name = (
        next(iter(profiles))
    )

    set_active_profile_name(
        active_profile_name
    )

profile = (
    profiles.get(active_profile_name)
    if active_profile_name
    else None
)

usage = (
    get_selected_usage(
        active_profile_name
    )
    if active_profile_name
    else None
)

chat_id = get_telegram_chat_id()
model_artifact = load_model()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
             NIP
        </div>
        <div class="hero-subtitle">
            AI-assisted digital wellbeing & safety monitor
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Control Center")

    if active_profile_name:
        st.success(
            f"👤 Active user: {active_profile_name}"
        )
    else:
        st.warning(
            "No active user"
        )

    st.divider()

    if st.button(
        "🔄 Refresh",
        use_container_width=True,
    ):
        st.rerun()

    if st.button(
        "▶️ Start Monitor",
        use_container_width=True,
    ):
        ok, message = (
            start_background_monitor()
        )

        if ok:
            st.success(message)
        else:
            st.error(message)

    if st.button(
        "⏹ Stop Monitor",
        use_container_width=True,
    ):
        ok, message = (
            stop_background_monitor()
        )

        if ok:
            st.success(message)
        else:
            st.error(message)

    st.divider()

    st.caption(
        f"Updated: {datetime.now().strftime('%H:%M:%S')}"
    )


# ============================================================
# STATUS
# ============================================================

st.subheader("System Status")

col1, col2, col3, col4 = st.columns(4)

with col1,col2:
    st.metric(
        "Monitor",
        "🟢 ONLINE"
        if monitor_is_running()
        else "🔴 OFFLINE",
    )

#with col2:
 #   st.metric(
  #      "XGBoost",
   #     "READY"
    #    if model_artifact
     #   else "MISSING",
    

with col3:
    st.metric(
        "Telegram",
        "CONNECTED"
        if chat_id
        else "NOT PAIRED",
    )

with col4:
    st.metric(
        "Usage Data",
        "AVAILABLE"
        if usage
        else "WAITING",
    )


# ============================================================
# USER MANAGEMENT
# ============================================================

st.divider()
st.subheader("👤 User Management")

left, right = st.columns(
    [1.2, 1]
)

with left:

    if profiles:

        profile_names = list(
            profiles.keys()
        )

        current_index = (
            profile_names.index(
                active_profile_name
            )
            if active_profile_name
            in profile_names
            else 0
        )

        selected_name = st.selectbox(
            "Active user",
            profile_names,
            index=current_index,
        )

        if selected_name != active_profile_name:

            active_profile_name = (
                selected_name
            )

            profile = profiles[
                selected_name
            ]

            set_active_profile_name(
                selected_name
            )

            activate_profile(
                selected_name,
                profile,
            )

            st.rerun()

    else:

        st.info(
            "No users yet. Create the first profile."
        )


with right:

    st.caption(
        "Each person gets their own profile and usage record."
    )

    st.caption(
        "Changing the active user changes which data "
        "the XGBoost model analyzes."
    )


# ============================================================
# PROFILE EDITOR
# ============================================================

with st.expander(
    "➕ Create / Edit User Profile",
    expanded=(profile is None),
):

    defaults = normalize_profile(
        profile or {}
    )

    name_value = (
        active_profile_name
        if active_profile_name
        else ""
    )

    profile_name = st.text_input(
        "Profile name",
        value=name_value,
        placeholder="Example: Alex",
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=5,
            max_value=100,
            value=defaults["age"],
            step=1,
        )

        gender_options = [
            "Male",
            "Female",
            "Other",
        ]

        gender = st.selectbox(
            "Gender",
            gender_options,
            index=(
                gender_options.index(
                    defaults["gender"]
                )
                if defaults["gender"]
                in gender_options
                else 0
            ),
        )

        sleep = st.number_input(
            "Average sleep (hours)",
            min_value=0.0,
            max_value=24.0,
            value=defaults["sleep_hours"],
            step=0.5,
        )

    with col2:

        work = st.number_input(
            "Work / study (hours/day)",
            min_value=0.0,
            max_value=24.0,
            value=defaults["work_study_hours"],
            step=0.5,
        )

        stress_options = [
            "Low",
            "Medium",
            "High",
        ]

        stress = st.selectbox(
            "Stress level",
            stress_options,
            index=(
                stress_options.index(
                    defaults["stress_level"]
                )
                if defaults["stress_level"]
                in stress_options
                else 1
            ),
        )

        impact = st.selectbox(
            "Academic/work impact",
            ["No", "Yes"],
            index=(
                1
                if defaults[
                    "academic_work_impact"
                ] == "Yes"
                else 0
            ),
        )

    weekend_profile = st.number_input(
        "Typical weekend screen time (hours)",
        min_value=0.0,
        max_value=24.0,
        value=defaults[
            "weekend_screen_time"
        ],
        step=0.5,
    )

    if st.button(
        "💾 Save Profile",
        use_container_width=True,
    ):

        if not profile_name.strip():

            st.error(
                "Enter a profile name."
            )

        else:

            clean_name = (
                profile_name.strip()
            )

            new_profile = {
                "age": int(age),
                "gender": gender,
                "work_study_hours": float(
                    work
                ),
                "sleep_hours": float(
                    sleep
                ),
                "stress_level": stress,
                "academic_work_impact": impact,
                "weekend_screen_time": float(
                    weekend_profile
                ),
            }

            profiles[
                clean_name
            ] = new_profile

            save_profiles(
                profiles
            )

            set_active_profile_name(
                clean_name
            )

            activate_profile(
                clean_name,
                new_profile,
            )

            st.success(
                f"✅ {clean_name} is now the active user."
            )

            st.rerun()


# ============================================================
# USER SUMMARY
# ============================================================

if profile:

    st.subheader(
        f"👤 {active_profile_name}"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Age",
            profile["age"],
        )

    with c2:
        st.metric(
            "Gender",
            profile["gender"],
        )

    with c3:
        st.metric(
            "Sleep",
            f"{profile['sleep_hours']:.1f} h",
        )

    with c4:
        st.metric(
            "Stress",
            profile["stress_level"],
        )


# ============================================================
# USAGE
# ============================================================

st.divider()
st.subheader("📊 Usage Data")

st.caption(
    "For now, usage can be entered per user. "
    "Later, the macOS collector will update these values automatically."
)

with st.expander(
    "✏️ Enter / update this user's usage",
    expanded=(usage is None),
):

    current_usage = usage or {}

    c1, c2, c3 = st.columns(3)

    with c1:

        screen_time = st.number_input(
            "Daily screen time (hours)",
            min_value=0.0,
            max_value=24.0,
            value=float(
                current_usage.get(
                    "daily_screen_time_hours",
                    0.0,
                )
            ),
            step=0.1,
        )

        social_media = st.number_input(
            "Social media (hours)",
            min_value=0.0,
            max_value=24.0,
            value=float(
                current_usage.get(
                    "social_media_hours",
                    0.0,
                )
            ),
            step=0.1,
        )

    with c2:

        gaming = st.number_input(
            "Gaming (hours)",
            min_value=0.0,
            max_value=24.0,
            value=float(
                current_usage.get(
                    "gaming_hours",
                    0.0,
                )
            ),
            step=0.1,
        )

        notifications = st.number_input(
            "Notifications/day",
            min_value=0,
            max_value=10000,
            value=int(
                current_usage.get(
                    "notifications_per_day",
                    0,
                )
            ),
            step=1,
        )

    with c3:

        app_opens = st.number_input(
            "App opens/day",
            min_value=0,
            max_value=10000,
            value=int(
                current_usage.get(
                    "app_opens_per_day",
                    0,
                )
            ),
            step=1,
        )

        weekend_usage = st.number_input(
            "Weekend screen time (hours)",
            min_value=0.0,
            max_value=24.0,
            value=float(
                current_usage.get(
                    "weekend_screen_time",
                    profile.get(
                        "weekend_screen_time",
                        0.0,
                    )
                    if profile
                    else 0.0,
                )
            ),
            step=0.1,
        )

    if active_profile_name:

        if st.button(
            f"💾 Save usage for {active_profile_name}",
            use_container_width=True,
        ):

            new_usage = {
                "daily_screen_time_hours":
                    float(screen_time),
                "social_media_hours":
                    float(social_media),
                "gaming_hours":
                    float(gaming),
                "notifications_per_day":
                    int(notifications),
                "app_opens_per_day":
                    int(app_opens),
                "weekend_screen_time":
                    float(weekend_usage),
            }

            save_selected_usage(
                active_profile_name,
                new_usage,
            )

            st.success(
                f"✅ Usage saved for {active_profile_name}."
            )

            st.rerun()


if usage:

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Screen time",
            f"{usage.get('daily_screen_time_hours', 0):.2f} h",
        )

    with c2:
        st.metric(
            "Notifications",
            usage.get(
                "notifications_per_day",
                0,
            ),
        )

    with c3:
        st.metric(
            "App opens",
            usage.get(
                "app_opens_per_day",
                0,
            ),
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Social media",
            f"{usage.get('social_media_hours', 0):.2f} h",
        )

    with c2:
        st.metric(
            "Gaming",
            f"{usage.get('gaming_hours', 0):.2f} h",
        )

    with c3:
        st.metric(
            "Weekend",
            f"{usage.get('weekend_screen_time', 0):.2f} h",
        )

else:

    st.warning(
        "No usage data is available for this user yet."
    )


# ============================================================
# AI ANALYSIS
# ============================================================

st.divider()
st.subheader("🧠 AI Risk Analysis")

analyze = st.button(
    "🧠 Analyze Current User",
    type="primary",
    use_container_width=True,
)

if analyze:

    if not active_profile_name:

        st.error(
            "Create/select a user first."
        )

    elif profile is None:

        st.error(
            "The active user's profile is missing."
        )

    elif usage is None:

        st.error(
            "No usage data exists for this user."
        )

    else:

        result = run_prediction(
            profile,
            usage,
        )

        if "error" in result:

            st.error(
                result["error"]
            )

        else:

            probability = result[
                "probability"
            ]

            prediction = result[
                "prediction"
            ]

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Estimated Risk",
                    f"{probability:.1%}",
                )

                st.progress(
                    min(
                        probability,
                        1.0,
                    )
                )

            with c2:

                if probability >= 0.70:

                    st.error(
                        "⚠️ HIGH RISK"
                    )

                elif probability >= 0.50:

                    st.warning(
                        "🟡 ELEVATED RISK"
                    )

                else:

                    st.success(
                        "🟢 LOWER RISK"
                    )

            if probability >= 0.70:

                st.error(
                    "The model estimates a high-risk "
                    "usage pattern. Review the underlying "
                    "usage values before taking action."
                )

            elif probability >= 0.50:

                st.warning(
                    "The model estimates an elevated "
                    "usage pattern."
                )

            else:

                st.success(
                    "The model estimates a lower-risk "
                    "usage pattern."
                )

            with st.expander(
                "🔎 See values sent to the model"
            ):

                feature_df = pd.DataFrame(
                    [
                        result["features"]
                    ]
                ).T

                feature_df.columns = [
                    "Value"
                ]

                st.dataframe(
                    feature_df,
                    use_container_width=True,
                )


# ============================================================
# TELEGRAM
# ============================================================

st.divider()
st.subheader("📱 connect Telegram")

if chat_id:

    st.success(
        "✅ A Telegram account is paired."
    )

    masked = (
        "••••••••"
        + str(chat_id)[-4:]
    )

    c1, c2 = st.columns(
        [3, 1]
    )

    with c1:
        st.text_input(
            "Telegram Chat ID",
            value=masked,
            disabled=True,
        )

    with c2:

        if st.button(
            "👁 Reveal",
            use_container_width=True,
        ):
            st.info(
                f"Chat ID: {chat_id}"
            )

    if st.button(
        "📨 Send Test Alert",
        use_container_width=True,
    ):

        ok, message = (
            send_telegram_test()
        )

        if ok:
            st.success(message)
        else:
            st.error(message)

else:

    st.warning(
        "No family Telegram account is paired."
    )

    st.info(
        "Open your bot on the family phone, send /start, "
        "then use the registration code in the background monitor."
    )




# ============================================================
# SAFETY MONITOR
# ============================================================

st.divider()
st.subheader("🛡️ Safety Monitor")

c1, c2 = st.columns(2)

#   if monitor_is_running():

 #       st.success(
  #          "🟢 background_monitor.py is running."
        

   # else:

    #    st.warning(
     #       "🟡 background_monitor.py is not running."
        

#with c2:

 #   st.info(
  #      "NSFW detection runs in the background monitor, "
   #     "separately from the XGBoost addiction model."
    


# ============================================================
# LOG
# ============================================================

with st.expander(
    "📋 Recent monitor log"
):

    if os.path.exists(LOG_FILE):

        try:

            with open(
                LOG_FILE,
                "r",
                encoding="utf-8",
            ) as f:

                logs = f.readlines()

            st.code(
                "".join(
                    logs[-40:]
                ),
                language="text",
            )

        except Exception as e:

            st.error(
                f"Could not read log: {e}"
            )

    else:

        st.info(
            "No monitor log exists yet."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()




st.caption(
    "Model output is a statistical prediction based on "
    "the training data; it is not a diagnosis."
)
