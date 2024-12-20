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
    Watershed_Segmentation,
    Instance_Search
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
            "GrabCut",
            "Watershed Segmentation",
            "Face Detection",
            "Senmatic Keypoints",
            "Features Matching",
            "SOT KCF",
            "MOT SORT"
        ],
        icons=[
            "scissors",
            "water",
            "person-bounding-box",
            "key",
            "intersect",
            "crosshair",
            "people"
        ],
        menu_icon="cast",
        default_index=0,
    )

# Main content area
st.title("Computer Vision Applications")

# Route to different pages based on selection
if selected == "GrabCut":
    GrabCut.main()
elif selected == "Watershed Segmentation":
    Watershed_Segmentation.main()
elif selected == "Face Detection":
    Face_Detection.main()
elif selected == "Senmatic Keypoints":
    Senmatic_Keypoints.main()
elif selected == "Features Matching":
    Features_Matching.main()
elif selectrd == "Instance Search" :
    Instance_Search.main()
elif selected == "SOT KCF":
    SOT_KCF.main()
elif selected == "MOT SORT":
    MOT_SORT.main()
