import streamlit as st
import sounddevice as sd
import soundfile as sf
from tempfile import NamedTemporaryFile
from backend import get_engine # This line imports from your backend.py
import os
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Nepali AI Studio: Speech to Text & Summarization",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎤 Nepali AI Studio: Speech to Text & Summarization")

# --- Session State Initialization ---
if "tmp_path" not in st.session_state:
    st.session_state.tmp_path = None
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "audio_recorded" not in st.session_state:
    st.session_state.audio_recorded = False
if "audio_uploaded" not in st.session_state:
    st.session_state.audio_uploaded = False

# --- Backend Engine Initialization ---
@st.cache_resource
def load_engine():
    with st.spinner("Loading AI models... This might take a moment."):
        return get_engine()

engine = load_engine()

# --- Sidebar for Settings and Info ---
with st.sidebar:
    st.header("⚙️ Settings")
    duration = st.number_input(
        "Recording Duration (seconds)",
        min_value=1,
        max_value=60,
        value=10,
        help="Set the maximum duration for audio recording."
    )

    st.markdown("---")
    st.header("ℹ️ About This App")
    st.info(
        """
        This application leverages advanced AI models to convert spoken Nepali audio into text
        (Speech-to-Text) and then generate a concise summary of the transcript.
        """
    )

    with st.expander("📚 Model Information"):
        st.markdown(
            """
            **Speech-to-Text (ASR) Model:**
            `iamTangsang/Wav2Vec2_XLS-R-300m_Nepali_ASR`
            """
        )
        st.markdown(
            """
            **Summarization Model:**
            `csebuetnlp/mT5_multilingual_XLSum`
            """
        )
        st.caption("Models are loaded locally for processing.")

# --- Main Content Area ---

# --- Step 1: Audio Input ---
st.subheader("1. Audio Input")
input_tab, record_tab = st.tabs(["📤 Upload Audio", "🎤 Record Audio"])

with input_tab:
    audio_file = st.file_uploader(
        "Upload an audio file (WAV, MP3, M4A)",
        type=["wav", "mp3", "m4a"],
        help="Select an audio file from your local machine."
    )

    if audio_file:
        # Clear previous state if a new file is uploaded
        if st.session_state.tmp_path and os.path.exists(st.session_state.tmp_path):
            os.remove(st.session_state.tmp_path)
        st.session_state.transcript = ""
        st.session_state.summary = ""

        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            st.session_state.tmp_path = tmp.name
        st.audio(st.session_state.tmp_path, format=audio_file.type)
        st.success("Audio uploaded successfully! Ready for processing.")
        st.session_state.audio_uploaded = True
        st.session_state.audio_recorded = False # Ensure only one input method is active

with record_tab:
    if st.button("Start Recording", help=f"Click to record for {duration} seconds."):
        # Clear previous state if a new recording starts
        if st.session_state.tmp_path and os.path.exists(st.session_state.tmp_path):
            os.remove(st.session_state.tmp_path)
        st.session_state.transcript = ""
        st.session_state.summary = ""

        st.info(f"🔴 Recording in progress... Please speak for {duration} seconds.")
        try:
            fs = 16000 # Sample rate
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait() # Wait until recording is finished

            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, recording, fs)
                st.session_state.tmp_path = tmp.name
            st.audio(st.session_state.tmp_path)
            st.success("✅ Recording complete! Ready for processing.")
            st.session_state.audio_recorded = True
            st.session_state.audio_uploaded = False # Ensure only one input method is active
        except Exception as e:
            st.error(f"Failed to record audio: {e}")
            st.warning("Please ensure your microphone is connected and allowed by your browser.")

# --- Step 2: Process Audio ---
st.subheader("2. Process Audio")
process_button_col, reset_button_col = st.columns([0.7, 0.3])

with process_button_col:
    if st.button("🚀 Transcribe & Summarize", disabled=(not st.session_state.tmp_path)):
        if st.session_state.tmp_path:
            # Clear previous results if re-processing
            st.session_state.transcript = ""
            st.session_state.summary = ""

            st.markdown("---") # Separator for processing feedback
            st.info("🔄 Initiating audio processing...")

            # Transcribe
            with st.spinner("🎧 Transcribing Nepali audio..."):
                try:
                    transcript = engine.transcribe(st.session_state.tmp_path)
                    st.session_state.transcript = transcript
                    st.success("Transcription complete!")
                except Exception as e:
                    st.error(f"Error during transcription: {e}")
                    st.session_state.transcript = "Error: Could not transcribe audio."

            # Summarize
            if st.session_state.transcript and "Error" not in st.session_state.transcript:
                with st.spinner("📝 Generating summary..."):
                    try:
                        summary = engine.summarize(st.session_state.transcript)
                        st.session_state.summary = summary
                        st.success("Summary generated!")
                    except Exception as e:
                        st.error(f"Error during summarization: {e}")
                        st.session_state.summary = "Error: Could not generate summary."
            else:
                st.warning("Skipping summarization due to transcription errors or empty transcript.")
            st.markdown("---") # Separator after processing feedback
        else:
            st.warning("Please upload or record audio first before processing.")

with reset_button_col:
    if st.button("🧹 Clear All", help="Clears all uploaded audio, recordings, and results."):
        if st.session_state.tmp_path and os.path.exists(st.session_state.tmp_path):
            os.remove(st.session_state.tmp_path)
        st.session_state.tmp_path = None
        st.session_state.transcript = ""
        st.session_state.summary = ""
        st.session_state.audio_recorded = False
        st.session_state.audio_uploaded = False
        st.success("Application reset! You can start fresh.")
        st.rerun() # Rerun to clear the UI immediately

# --- Step 3: Results Display ---
st.subheader("3. Results")

if st.session_state.transcript or st.session_state.summary:
    # Transcript Section
    with st.expander("📝 Transcript", expanded=True):
        if st.session_state.transcript:
            st.markdown(f"**Generated Transcript:**")
            st.write(st.session_state.transcript)
            st.download_button(
                label="Download Transcript",
                data=st.session_state.transcript.encode("utf-8"),
                file_name="nepali_transcript.txt",
                mime="text/plain",
                help="Download the full transcribed text."
            )
        else:
            st.info("No transcript available yet. Process an audio file to see it here.")

    # Summary Section
    with st.expander("📚 Summary", expanded=True):
        if st.session_state.summary:
            st.markdown(f"**Generated Summary:**")
            st.write(st.session_state.summary)
            st.download_button(
                label="Download Summary",
                data=st.session_state.summary.encode("utf-8"),
                file_name="nepali_summary.txt",
                mime="text/plain",
                help="Download the concise summary."
            )
        else:
            st.info("No summary available yet. Process an audio file to see it here.")
else:
    st.info("Once you process audio, the transcript and summary will appear here.")