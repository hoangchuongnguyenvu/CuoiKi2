import streamlit as st
from streamlit_option_menu import option_menu
import sys
from pathlib import Path

# Add the pages directory to system path
sys.path.append(str(Path(__file__).parent))

# Import all pages
from pages import (
    Face_Detection,
    Face_Verification,
    Features_Matching,
    GrabCut,
    MOT_SORT,
    SOT_KCF,
    Senmatic_Keypoints,
    Watershed_Segmentation
)

st.set_page_config(
    page_title="Computer Vision Applications",
    page_icon="🎯",
    layout="wide"
)

# Create the sidebar menu
with st.sidebar:
    st.title("Computer Vision Apps")
    selected = option_menu(
        menu_title=None,
        options=[
            "Face Detection",
            "Face Verification",
            "Features Matching",
            "GrabCut Segmentation",
            "Multiple Object Tracking",
            "Single Object Tracking",
            "Semantic Keypoints",
            "Watershed Segmentation"
        ],
        icons=[
            "person-bounding-box",
            "person-check",
            "intersect",
            "scissors",
            "people",
            "crosshair",
            "key",
            "water"
        ],
        menu_icon="cast",
        default_index=0,
    )

# Main content area
st.title("Computer Vision Applications")

# Route to different pages based on selection
if selected == "Face Detection":
    Face_Detection.main()
elif selected == "Face Verification":
    Face_Verification.main()
elif selected == "Features Matching":
    Features_Matching.main()
elif selected == "GrabCut Segmentation":
    GrabCut.main()
elif selected == "Multiple Object Tracking":
    MOT_SORT.main()
elif selected == "Single Object Tracking":
    SOT_KCF.main()
elif selected == "Semantic Keypoints":
    Senmatic_Keypoints.main()
elif selected == "Watershed Segmentation":
    Watershed_Segmentation.main()
