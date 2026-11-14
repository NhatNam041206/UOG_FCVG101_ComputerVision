"""[3] Streamlit UI. Layer 3: it only uploads, picks a mode, calls assistant.run_mode() and shows the result.

Run:  streamlit run app.py
Flow: user upload + mode  ->  io_utils.decode_image_bytes  ->  assistant.run_mode [3.1]  ->  display ModeResult
"""
import streamlit as st

from src.assistant import MODES, run_mode
from src.config import load_config
from src.io_utils import decode_image_bytes, to_rgb


@st.cache_resource
def get_config():
    """Load config once per app session."""
    return load_config()


def main():
    st.title("FCVG101 Vision Inspection Assistant")
    cfg = get_config()
    mode = st.selectbox("What do you want to do?", list(MODES))
    upload = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
    upload_2 = st.file_uploader("Second image (features only)", type=["jpg", "jpeg", "png"]) if mode == "features" else None
    if upload is None:
        st.info("Upload an image to start.")
        return
    try:
        image = decode_image_bytes(upload.getvalue())
        image_2 = decode_image_bytes(upload_2.getvalue()) if upload_2 is not None else None
        result = run_mode(mode, image, cfg, image_2=image_2)
    except NotImplementedError as err:
        st.warning(f"Step {err} is not built yet.")
        return
    except (ValueError, FileNotFoundError) as err:
        st.error(str(err))
        return
    for caption, img in result.images.items():
        st.image(to_rgb(img), caption=caption)
    if result.numbers:
        st.table(result.numbers)
    if result.message:
        st.write(result.message)


if __name__ == "__main__":
    main()
