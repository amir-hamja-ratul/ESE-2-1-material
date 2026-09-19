import os
import base64
import streamlit as st

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="ESE Workspace",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. DIRECTORY & ASSET SETUP
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# 'assets' ফোল্ডার না থাকলে স্বয়ংক্রিয়ভাবে তৈরি করবে
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR, exist_ok=True)

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS & STYLES
# ---------------------------------------------------------
def get_base64_image(image_path):
    """ছবিকে Base64 স্ট্রিঙে রূপান্তর করার ফাংশন"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

# Custom CSS for polished look
st.markdown("""
    <style>
        .main-header {
            font-size: 2rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #475569;
            margin-bottom: 1.5rem;
        }
        .card {
            background-color: #F8FAFC;
            padding: 1.25rem;
            border-radius: 10px;
            border-left: 4px solid #10B981;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ---------------------------------------------------------
with st.sidebar:
    # CIRCULAR LOGO WITH GREEN BORDER
    IMAGE_NAME = "ese10_logo.jpg"
    sidebar_logo_path = os.path.join(ASSETS_DIR, IMAGE_NAME)
    
    if os.path.exists(sidebar_logo_path):
        img_base64 = get_base64_image(sidebar_logo_path)
        if img_base64:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; margin-top: 10px; margin-bottom: 20px;">
                    <img src="data:image/jpeg;base64,{img_base64}" style="
                        width: 110px;
                        height: 110px;
                        border-radius: 50%;
                        border: 3px solid #10B981;
                        object-fit: cover;
                        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.08);
                    ">
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning(f"⚠️ ছবি পাওয়া যায়নি: `assets/{IMAGE_NAME}`")

    st.markdown("<h3 style='font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-bottom: 12px;'>Workspace Navigation</h3>", unsafe_allow_html=True)
    
    # MENU NAVIGATION
    selected_page = st.radio(
        label="Select Page",
        options=[
            "🏠 Dashboard", 
            "📊 Data Analysis", 
            "📂 Projects & Field Reports", 
            "📢 Announcements",
            "⚙️ Settings"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("🌿 ESE Workspace System")

# ---------------------------------------------------------
# 5. MAIN CONTENT ROUTING
# ---------------------------------------------------------
if selected_page == "🏠 Dashboard":
    st.markdown("<h1 class='main-header'>🏠 Dashboard Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Welcome to the Environmental Science & Engineering Workspace!</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Active Projects", value="8", delta="+2")
    with col2:
        st.metric(label="Data Reports", value="24", delta="+4")
    with col3:
        st.metric(label="System Status", value="Online", delta="Stable")
        
    st.divider()
    
    st.markdown("""
        <div class="card">
            <h4 style="margin:0 0 8px 0; color:#0F172A;">📌 Overview & Quick Status</h4>
            <p style="margin:0; color:#475569;">নেভিগেশন মেনু ব্যবহার করে আপনার কাঙ্ক্ষিত মডিউল বা পেজে যান। ডাটা অ্যানালাইসিস বা প্রজেক্ট রিপোর্ট সেকশনে গিয়ে প্রয়োজন অনুযায়ী ফাইল প্রসেস করা যাবে।</p>
        </div>
    """, unsafe_allow_html=True)

elif selected_page == "📊 Data Analysis":
    st.markdown("<h1 class='main-header'>📊 Data Analysis</h1>", unsafe_allow_html=True)
    st.write("এখানে ডাটা ফাইল (CSV / Excel) আপলোড করে অ্যানালাইসিস সম্পন্ন করুন।")
    
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        st.success("File uploaded successfully!")

elif selected_page == "📂 Projects & Field Reports":
    st.markdown("<h1 class='main-header'>📂 Projects & Field Reports</h1>", unsafe_allow_html=True)
    st.write("ফিল্ড ট্রিপ রিপোর্ট, GIS ম্যাপ এবং অন্যান্য একাডেমিক ডকুমেন্টেশন ম্যানেজ করুন।")

elif selected_page == "📢 Announcements":
    st.markdown("<h1 class='main-header'>📢 Notice & Announcements</h1>", unsafe_allow_html=True)
    st.info("এখনো পর্যন্ত নতুন কোনো ইমার্জেন্সি নোটিশ নেই।")

elif selected_page == "⚙️ Settings":
    st.markdown("<h1 class='main-header'>⚙️ Settings</h1>", unsafe_allow_html=True)
    st.write("অ্যাপ্লিকেশন কনফিগারেশন এবং প্রেফারেন্স সেট করুন।")
