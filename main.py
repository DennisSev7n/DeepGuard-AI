import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF
import docx  # python-docx
from odf import text, teletype
from odf.opendocument import load
from utils import analyze_content, setup_session_state, analyze_text

# ---------------------------
# Extract text from files
# ---------------------------
def extract_text_from_file(file_bytes, filename):
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            return "\n".join([page.get_text() for page in doc])

        elif ext == ".txt":
            return file_bytes.decode("utf-8", errors="ignore")

        elif ext == ".rtf":
            return file_bytes.decode("utf-8", errors="ignore")

        elif ext == ".docx":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                doc = docx.Document(tmp_path)
                return "\n".join([p.text for p in doc.paragraphs])
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        elif ext == ".odt":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".odt") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                odt_doc = load(tmp_path)
                paragraphs = odt_doc.getElementsByType(text.P)
                return "\n".join([teletype.extractText(p) for p in paragraphs])
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        elif ext == ".doc":
            return "⚠️ .doc format not fully supported. Please upload as .docx."

        else:
            return f"Unsupported file format: {ext}"

    except Exception as e:
        return f"Error extracting text: {str(e)}"


# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(
    page_title="DeepGuard AI — Pro (with ML)",
    layout="wide",
    page_icon="🔍"
)

setup_session_state()

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("DeepGuard AI")
    modality = st.radio("Select modality:", ["Image", "Video", "Audio", "Text"])
    threshold = st.slider("Detection Threshold (%)", 0, 100, 60)
    st.markdown("---")
    st.info(
        """
        **How it works:**
        - Upload your file or paste text
        - Click 'Analyze' to start detection
        - Results show AI-generated likelihood
        - Higher percentage = more likely to be AI-generated
        """
    )

# ---------------------------
# Main UI
# ---------------------------
st.markdown("<h1 style='text-align:center;color:#1E88E5;'>DeepGuard AI — ML-enhanced Detection</h1>", unsafe_allow_html=True)

if modality == "Text":
    st.subheader("Analyze Text Content")

    text_input = st.text_area("Or paste text directly to analyze:", height=150)

    upload = st.file_uploader(
        "Upload text document",
        type=["txt", "pdf", "doc", "docx", "rtf", "odt"],
    )

    if st.button("Analyze", type="primary", use_container_width=True) and (text_input or upload):
        with st.spinner("Analyzing text content..."):
            progress_bar = st.progress(0)

            if text_input:
                text_to_analyze = text_input
                source_type = "direct_input"
                st.text_area("Preview of Input Text", value=text_to_analyze[:2000], height=200)

            else:
                file_bytes = upload.read()
                text_to_analyze = extract_text_from_file(file_bytes, upload.name)
                source_type = "file_upload"
                st.text_area("Preview of Extracted Text", value=text_to_analyze[:2000], height=200)

            score, details = analyze_text(text_to_analyze, progress_bar)
            details["source_type"] = source_type
            if source_type == "file_upload":
                details["file_name"] = upload.name

            st.session_state.analysis_done = True
            st.session_state.analysis_result = score
            st.session_state.analysis_details = details

            progress_bar.empty()

else:
    st.subheader(f"Analyze {modality} Content")

    file_types = {
        "Image": ["jpg", "jpeg", "png", "bmp", "gif", "tiff"],
        "Video": ["mp4", "mov", "avi", "mkv", "webm", "wmv"],
        "Audio": ["wav", "mp3", "flac", "m4a", "ogg", "aac"],
    }

    upload = st.file_uploader(f"Upload {modality} file", type=file_types[modality])

    if upload and st.button("Analyze", type="primary", use_container_width=True):
        with st.spinner("Analyzing content..."):
            progress_bar = st.progress(0)
            file_bytes = upload.read()

            # Display uploaded content preview
            if modality == "Image":
                st.image(file_bytes, caption=upload.name, use_container_width=True)

            elif modality == "Audio":
                st.audio(file_bytes, format="audio/wav")

            elif modality == "Video":
                st.video(file_bytes)

            # Run analysis
            score, details = analyze_content(modality, file_bytes, progress_bar, upload.name)
            details = {}
            details["file_name"] = upload.name

            st.session_state.analysis_done = True
            st.session_state.analysis_result = score
            st.session_state.analysis_details = details

            progress_bar.empty()

# ---------------------------
# Results
# ---------------------------
if st.session_state.get("analysis_done", False):
    st.markdown("---")
    st.markdown("### Analysis Results")

    result_color = "#FF4B4B" if st.session_state.analysis_result > threshold else "#00CC96"

    st.markdown(
        f"""
        <div style="text-align: center;">
            <div style="font-size:1.2rem;color:#666;">AI-Generated Content Likelihood</div>
            <div style="font-size:2.5rem;font-weight:bold;color:{result_color};">
                {st.session_state.analysis_result:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.analysis_result > threshold:
        st.error(f"**Verdict:** Likely AI-generated (above {threshold}% threshold)")
    else:
        st.success(f"**Verdict:** Likely authentic (below {threshold}% threshold)")

    with st.expander("View Detailed Analysis"):
        st.json(st.session_state.analysis_details)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>DeepGuard AI — Professional Content Authenticity Tool</div>", unsafe_allow_html=True)
