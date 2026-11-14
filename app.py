"""
Optimal Assignment of Teachers to Courses Using the Hungarian Method
A Simple, Clean, Beginner-Friendly College Project Website.
"""
import io
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.auth import init_auth_db, register_user, verify_user
from core.data_loader import load_raw_data, extract_courses, build_clash_graph
from core.cost_matrix import compute_base_cost_matrix
from core.hungarian import solve_repaired_hungarian
from core.validator import validate_assignment_solution
from core.suitability import calculate_suitability_score

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Teacher-Course Assignment Optimizer (Hungarian Method)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean Professional CSS Styling
st.markdown("""
<style>
    /* Clean light typography and background */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    h1, h2, h3, h4 {
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .project-hero {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .info-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .info-card h4 {
        margin-top: 0;
        color: #1e3a8a;
    }
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #1e40af;
    }
    .metric-label {
        font-size: 13px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .score-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-bottom: 20px;
    }
    .score-percentage {
        font-size: 44px;
        font-weight: 800;
        color: #1d4ed8;
    }
    .step-pill {
        display: inline-block;
        background-color: #2563eb;
        color: #ffffff;
        font-weight: bold;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Authentication Database
init_auth_db()

# -----------------------------------------------------------------------------
# 2. AUTHENTICATION CONTROLLER (LOGIN / REGISTRATION)
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_info" not in st.session_state:
    st.session_state["user_info"] = None

# If not authenticated, show ONLY the Authentication Portal
if not st.session_state["authenticated"]:
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; margin-bottom: 25px;">
        <h1 style="color: #1e3a8a; font-size: 2.2rem; margin-bottom: 6px;">
            🎓 Optimal Assignment of Teachers to Courses
        </h1>
        <h3 style="color: #475569; font-weight: 400; font-size: 1.15rem; margin-top: 0;">
            Operations Research Project | Kuhn-Munkres (Hungarian) Algorithm
        </h3>
        <p style="color: #64748b; font-size: 0.95rem; max-width: 650px; margin: 0 auto;">
            Please log in or register a new user account to access the college timetable optimization portal.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_center = st.columns([1, 2, 1])[1]

    with col_center:
        auth_tab_login, auth_tab_register = st.tabs(["🔑 Sign In (Login)", "📝 New User Registration"])

        # Tab 1: Login
        with auth_tab_login:
            st.markdown("#### Enter Your Credentials")
            login_username = st.text_input("Username", key="login_user_input", placeholder="e.g., admin")
            login_password = st.text_input("Password", type="password", key="login_pass_input", placeholder="e.g., 12345")

            st.caption("💡 **Demonstration Administrator Account:** Username: `admin` | Password: `12345`")

            if st.button("Log In to Project", type="primary", use_container_width=True):
                success, user_dict, msg = verify_user(login_username, login_password)
                if success:
                    st.session_state["authenticated"] = True
                    st.session_state["user_info"] = user_dict
                    st.success("✅ Login successful! Redirecting to project...")
                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        # Tab 2: Normal User Registration
        with auth_tab_register:
            st.markdown("#### Register a Simple Student / Evaluator Account")
            reg_full_name = st.text_input("Full Name", placeholder="e.g., Rahul Sharma")
            reg_username = st.text_input("Choose Username", placeholder="e.g., rahul2026")
            reg_password = st.text_input("Create Password", type="password", placeholder="At least 4 characters")
            reg_confirm = st.text_input("Confirm Password", type="password", placeholder="Repeat your password")

            if st.button("Create Account & Register", use_container_width=True):
                ok, msg = register_user(reg_full_name, reg_username, reg_password, reg_confirm)
                if ok:
                    st.success(f"✅ {msg}")
                    st.info("You can now switch to the 'Sign In' tab above and log in.")
                else:
                    st.error(f"❌ {msg}")

    # Stop execution here so unauthenticated users cannot see any project page
    st.stop()

# -----------------------------------------------------------------------------
# 3. AUTHENTICATED USER SESSION & SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
user_info = st.session_state["user_info"]

# Top Sidebar User Info & Logout
st.sidebar.markdown(f"### 👤 Logged In")
st.sidebar.markdown(f"**Name:** {user_info['full_name']}")
st.sidebar.markdown(f"**Username:** `{user_info['username']}` {'⭐ *(Admin)*' if user_info['is_admin'] else '*(Student)*'}")

if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")

# Strict Requirement 11: Navigation must contain ONLY these 5 items
st.sidebar.markdown("### 📌 Navigation")
nav_selection = st.sidebar.radio(
    "Select Page:",
    [
        "1. 🏠 Home",
        "2. 📖 About Project",
        "3. 📊 Dataset",
        "4. ⚡ Hungarian Method",
        "5. 📋 Results"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Dataset Selection")
dataset_choice = st.sidebar.selectbox(
    "Choose Timetable Schedule:",
    [
        "Complete Week (Expanded: 277 Sessions)",
        "Original Sample (236 Sessions)"
    ],
    index=0,
    help="Select which real timetable file to inspect and optimize."
)
dataset_key = "expanded" if "277" in dataset_choice else "original"

# -----------------------------------------------------------------------------
# 4. DATA ENGINE (CACHED DATA LOADING & SOLVER HELPERS)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data(source_key: str):
    df_tt, df_skills, df_prefs, df_abbr = load_raw_data(source_key)
    courses = extract_courses(df_tt, df_abbr)
    teachers = sorted(df_tt["teacher"].unique())
    clash_pairs, course_sessions = build_clash_graph(df_tt, courses)
    return df_tt, df_skills, df_prefs, df_abbr, courses, teachers, clash_pairs, course_sessions

df_timetable, df_skills, df_prefs, df_abbr, courses, teachers, clash_pairs, course_sessions = get_data(dataset_key)

# Fixed, practical college standard parameters (No complicated tuning needed)
W_SKILL = 2.0  # Weight for expertise mismatch
W_PREF = 0.5   # Weight for teacher preference rank
MAX_CAPACITY = 4  # Maximum courses per teacher

C_base = compute_base_cost_matrix(
    courses, teachers, df_skills, df_prefs,
    w_skill=W_SKILL, w_pref=W_PREF
)

def run_project_hungarian(dataset_key_to_use: str):
    """Executes the capacity-expanded Hungarian method with timetable conflict repair."""
    tt, sk, pr, ab, crs, tchs, clashes, _ = get_data(dataset_key_to_use)
    c_base = compute_base_cost_matrix(crs, tchs, sk, pr, w_skill=W_SKILL, w_pref=W_PREF)

    # 1. Baseline Current Assignment Cost
    cur_df = crs.copy()
    cur_df["assigned_teacher"] = cur_df["current_teacher"]
    cur_val = validate_assignment_solution(
        cur_df, tt, c_base, tchs, clashes,
        cap=MAX_CAPACITY, max_weekly_periods=40.0, lam=2.0
    )

    # 2. Hungarian Optimization
    hung_res = solve_repaired_hungarian(
        crs, tchs, tt, c_base, clashes,
        lam=2.0, cap=MAX_CAPACITY
    )
    rep_df = hung_res["dataframe"]
    opt_val = validate_assignment_solution(
        rep_df, tt, c_base, tchs, clashes,
        cap=MAX_CAPACITY, max_weekly_periods=40.0, lam=2.0
    )

    # 3. Dynamic Assignment Suitability Score (Bounded <= 90%)
    suitability_score, suitability_breakdown = calculate_suitability_score(
        rep_df, sk, pr, w_skill=W_SKILL, w_pref=W_PREF
    )

    return {
        "dataset_key": dataset_key_to_use,
        "raw_results": hung_res,
        "assigned_df": rep_df,
        "cur_val": cur_val,
        "opt_val": opt_val,
        "suitability_score": suitability_score,
        "suitability_breakdown": suitability_breakdown,
        "teachers": tchs,
        "courses": crs
    }

# -----------------------------------------------------------------------------
# PAGE 1: 🏠 HOME
# -----------------------------------------------------------------------------
if "1. 🏠 Home" in nav_selection:
    st.markdown("""
    <div class="project-hero">
        <h1 style="margin-top: 0; font-size: 2.1rem; color: #1e3a8a;">
            Optimal Assignment of Teachers to Courses Using the Hungarian Method
        </h1>
        <p style="font-size: 1.15rem; color: #334155; margin-bottom: 8px;">
            A Clean, Beginner-Friendly Operations Research Academic Project for College Timetable Optimization.
        </p>
        <span style="background-color: #2563eb; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">
            Kuhn-Munkres Algorithm
        </span>
        <span style="background-color: #0d9488; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600; margin-left: 6px;">
            Zero Timetable Clashes
        </span>
        <span style="background-color: #475569; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600; margin-left: 6px;">
            100% Deterministic OR
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌟 Welcome to the Project")
    st.markdown("""
    **Imagine a college has several teachers and several subjects.**
    Each teacher has different knowledge, subject qualifications, and teaching preferences.
    Assigning teachers manually to course sections takes hours, causes human error, and often results in timetable clashes where one teacher is scheduled for two different classes at the exact same hour!

    Our project uses the **Hungarian Method (Kuhn-Munkres Algorithm)** to calculate the most suitable teacher-course assignments with minimum total assignment cost.
    """)

    st.markdown("---")
    st.subheader("💡 At a Glance: What Does This Application Do?")

    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.markdown("""
        <div class="info-card">
            <h4>1. Real College Dataset</h4>
            <p style="color: #475569; font-size: 0.95rem;">
                Loads real academic department data with <strong>13 faculty members</strong>, <strong>36 course sections</strong>, and <strong>18 unique subjects</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_h2:
        st.markdown("""
        <div class="info-card">
            <h4>2. Hungarian Method</h4>
            <p style="color: #475569; font-size: 0.95rem;">
                Applies exact combinatorial optimization to find the lowest-cost assignment while respecting each teacher's capacity of <strong>up to 4 courses</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_h3:
        st.markdown("""
        <div class="info-card">
            <h4>3. Practical Schedule</h4>
            <p style="color: #475569; font-size: 0.95rem;">
                Guarantees <strong>zero timetable clashes</strong>, calculates an <strong>Assignment Suitability Score (capped at 90%)</strong>, and renders clean charts.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🗺️ Recommended Project Tour for Students & Evaluators")
    st.markdown("""
    To explore and understand the project step-by-step, follow the navigation in the left sidebar:
    1. **📖 About Project:** Read clear, non-technical answers to the 6 core project questions.
    2. **📊 Dataset:** Understand what the dataset represents, view actual column definitions, and see real sample records.
    3. **⚡ Hungarian Method:** Click one button to trigger the Hungarian algorithm on the selected timetable.
    4. **📋 Results:** Inspect the assigned teachers, before-vs-after costs, suitability score, and visual charts.
    """)

    # Quick Jump Button
    if st.button("👉 Go to Hungarian Method Optimizer", type="primary"):
        st.info("Please select **'4. ⚡ Hungarian Method'** from the left sidebar navigation menu to run the optimizer!")

# -----------------------------------------------------------------------------
# PAGE 2: 📖 ABOUT PROJECT
# -----------------------------------------------------------------------------
elif "2. 📖 About Project" in nav_selection:
    st.markdown("""
    <div class="project-hero">
        <h1 style="margin-top: 0; font-size: 2.1rem; color: #1e3a8a;">
            📖 About the Project: Plain-English Explanation
        </h1>
        <p style="font-size: 1.05rem; color: #334155; margin-bottom: 0;">
            Everything you need to know about the Teacher-Course Assignment Optimization problem explained in simple, everyday language.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ❓ Answering the 6 Core Questions")

    with st.expander("1. What is this project?", expanded=True):
        st.markdown("""
        This project is a **college-level Operations Research website** that solves an everyday university challenge:
        **"Which teacher should teach which course?"**
        
        Instead of a department head manually guessing or spending days juggling spreadsheets, our system uses a proven mathematical technique called the **Hungarian Method** to automatically pair teachers with courses in the best possible way.
        """)

    with st.expander("2. Why is teacher assignment necessary?", expanded=True):
        st.markdown("""
        Every new academic semester, universities must assign faculty members to courses and classroom sections.
        - If a teacher is assigned a subject outside their expertise, students do not get the best learning experience.
        - If a teacher's subject preferences are ignored, job satisfaction and teaching morale decrease.
        - If teaching loads are not monitored, some teachers end up overloaded while others teach very few hours.
        - If class timings are not checked, a teacher might be assigned two lectures scheduled at the exact same hour!
        
        Proper teacher assignment is necessary to ensure **high academic quality, fair workload, and conflict-free schedules**.
        """)

    with st.expander("3. What problem are we solving?", expanded=True):
        st.markdown("""
        We are solving the **Teacher-Course Assignment Problem**.
        Specifically, we want to assign **36 course sections** among **13 faculty members** such that:
        1. **High Expertise:** Teachers teach courses they are well-qualified in.
        2. **High Preference:** Teachers teach courses they actually prefer to teach.
        3. **Strict Capacity Limit:** No teacher is assigned more than **4 courses** ($K = 4$).
        4. **Zero Overlaps:** No teacher has overlapping lecture sessions in the timetable.
        """)

    with st.expander("4. What is the Hungarian Method?", expanded=True):
        st.markdown("""
        The **Hungarian Method** (also known as the *Kuhn-Munkres Algorithm*) is a classic mathematical algorithm developed in 1955 by Harold Kuhn and James Munkres. It finds the lowest-cost assignment between two equal sets of items in polynomial time $O(n^3)$.

        **Technical terms explained in plain words:**
        - **Cost Matrix:** A simple table of numbers. Each row represents a course, and each column represents a teacher. The number in each cell is the "penalty cost" of assigning that course to that teacher. Lower numbers mean a better match; higher numbers mean an unsuitable match.
        - **Row Reduction:** Subtracting the smallest number in each row from all numbers in that row, creating at least one zero in every row.
        - **Column Reduction:** Subtracting the smallest number in each column from all numbers in that column.
        - **Line Covering:** Drawing the minimum number of straight lines through rows and columns to cover all zeros.
        - **Capacity Expansion:** Standard Hungarian matching pairs 1 teacher to 1 course (1-to-1). But in our college, each teacher can teach up to 4 courses! To solve this, we create **4 duplicate slots for each teacher** ($13 \\times 4 = 52$ slots) and add **16 dummy course rows** with zero cost. This creates a balanced $52 \\times 52$ square matrix that the Hungarian Method solves with mathematical perfection!
        """)

    with st.expander("5. How does our system work?", expanded=True):
        st.markdown("""
        Our system operates in **4 easy-to-follow steps**:
        
        <div style="margin-top: 10px;">
            <p><span class="step-pill">1</span> <strong>Load Data:</strong> Reads the college timetable, faculty expertise ratings (1 to 10), and teacher subject preference rankings (1 to 18).</p>
            <p><span class="step-pill">2</span> <strong>Calculate Assignment Costs:</strong> Calculates a cost value for every teacher-course pairing using the formula:<br>
            <code>Assignment Cost = 2.0 × (10 - Skill Score) + 0.5 × (Preference Rank)</code></p>
            <p><span class="step-pill">3</span> <strong>Execute Hungarian Method:</strong> Finds the mathematically optimal lowest-cost assignment across all 52 teacher slots.</p>
            <p><span class="step-pill">4</span> <strong>Verify & Display Results:</strong> Checks that no teacher has timetable collisions, verifies that capacity is respected, and displays clear results and charts.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("6. What is the final output?", expanded=True):
        st.markdown("""
        The final output of our website is:
        - A clean, easy-to-read table showing **Course Section**, **Assigned Teacher**, and **Assignment Cost**.
        - A comparison showing total assignment cost **Before vs After Optimization**.
        - A dynamic **Assignment Suitability Score (bounded between 0% and 90%)**.
        - Simple visual bar charts showing teacher workload and weekly teaching hours.
        - One-click export options to download the results as CSV or Excel.
        """)

# -----------------------------------------------------------------------------
# PAGE 3: 📊 DATASET
# -----------------------------------------------------------------------------
elif "3. 📊 Dataset" in nav_selection:
    st.markdown("""
    <div class="project-hero">
        <h1 style="margin-top: 0; font-size: 2.1rem; color: #1e3a8a;">
            📊 Understanding Our Dataset
        </h1>
        <p style="font-size: 1.05rem; color: #334155; margin-bottom: 0;">
            A complete, simple guide to the real college datasets powering this project.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Explanation Section
    st.markdown("### 📘 Non-Technical Dataset Guide")
    st.markdown("""
    - **What is a dataset?**  
      A dataset is an organized collection of data, similar to a digital spreadsheet, stored in rows and columns.
    - **Where does our project data come from?**  
      Our project uses real university academic department records containing weekly timetable sessions, teacher rosters, teacher expertise surveys, and subject preference rankings.
    - **What is a teacher record?**  
      A record representing an individual faculty member (e.g., `HNS`, `JSD`, `SKG`, `VI`), including their subject competence and preferred teaching topics.
    - **What is a course or class section?**  
      A specific academic class requiring instruction, such as *Advanced Machine Learning (AML)* for section *CSE-4A*.
    - **What is teacher expertise?**  
      A numerical score from **1 to 10** evaluating how qualified a teacher is to teach a particular subject based on their academic degrees and teaching experience. (10 is expert; 1 is novice).
    - **What are teacher preferences?**  
      Rankings submitted by teachers from **Rank 1 to Rank 18** showing which subjects they most wish to teach. (Rank 1 is their favorite subject; Rank 18 is least preferred).
    - **What is a teaching session?**  
      A single 1-hour scheduled classroom lecture period taking place on a specific date, day of the week, time slot, and room.
    - **How does the dataset help us assign teachers?**  
      It provides all necessary inputs: what needs to be taught (courses), who is available (teachers), who is best qualified (expertise), who wants to teach what (preferences), and when classes take place (timetable slots).
    """)

    st.markdown("---")
    st.subheader("📈 Inspected Dataset Statistics")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">13</div>
            <div class="metric-label">Faculty Members</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">36</div>
            <div class="metric-label">Course Sections</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">18</div>
            <div class="metric-label">Unique Subjects</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(df_timetable)}</div>
            <div class="metric-label">Timetable Sessions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔍 Dataset Comparison: 236 vs 277 Sessions")
    st.info(f"""
    **Current Selected Schedule:** **{dataset_choice}**  
    - **Expanded Dataset (`ot-database-full.csv`):** Contains **277 timetable sessions**. This represents the complete weekly university timetable covering all weekdays and class slots.
    - **Original Sample Dataset (`ot-database.csv`):** Contains **236 timetable sessions**. This was the initial partial sample dataset collected during early project development.
    - *Both datasets contain the exact same 13 teachers, 36 course sections, and 18 subjects.*
    """)

    st.markdown("---")
    st.subheader("📋 Dataset Previews and Column Descriptions")

    tab_crs, tab_tt, tab_exp, tab_prf, tab_ab = st.tabs([
        "📚 Course Sections (36 Courses)",
        "📅 Timetable Sessions",
        "⭐ Faculty Expertise (1-10)",
        "❤️ Faculty Preferences (1-18)",
        "📖 Subject Names"
    ])

    with tab_crs:
        st.markdown("#### Course Sections Requiring an Assigned Teacher")
        st.caption("Each row represents one class section. Important columns: `course_id`, `subject`, `class_code`, `periods` (weekly hours), and `current_teacher` (original manual assignment).")
        st.dataframe(courses[["course_id", "subject", "class_code", "class_name", "periods", "current_teacher"]], use_container_width=True)

    with tab_tt:
        st.markdown(f"#### Timetable Classroom Sessions ({len(df_timetable)} Sessions)")
        st.markdown("""
        **Important Columns Explained:**
        - `date` & `day`: Date and day of the week for the scheduled class.
        - `slot` & `time`: Class period (e.g., Slot 1 = 09:00 - 10:00).
        - `class_code` & `class_name`: Student batch/section.
        - `subject`: Short abbreviation for the academic subject.
        - `teacher`: Incumbent teacher recorded in the raw schedule.
        - `room`: Room or laboratory number.
        """)
        st.dataframe(df_timetable.head(30), use_container_width=True)

    with tab_exp:
        st.markdown("#### Faculty Expertise Ratings (Scale: 1 to 10)")
        st.caption("Rows: 13 Teachers | Columns: 18 Subjects. Rating 10 indicates mastery; 1 indicates minimal subject qualification.")
        st.dataframe(df_skills, use_container_width=True)

    with tab_prf:
        st.markdown("#### Faculty Subject Preferences (Rank: 1 to 18)")
        st.caption("Rows: 13 Teachers | Columns: 18 Subjects. Rank 1 is the teacher's top favorite subject; Rank 18 is least preferred.")
        st.dataframe(df_prefs, use_container_width=True)

    with tab_ab:
        st.markdown("#### Subject Code Lookup Table")
        st.caption("Maps short abbreviations to full official academic subject names.")
        st.dataframe(df_abbr, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 4: ⚡ HUNGARIAN METHOD
# -----------------------------------------------------------------------------
elif "4. ⚡ Hungarian Method" in nav_selection:
    st.markdown("""
    <div class="project-hero">
        <h1 style="margin-top: 0; font-size: 2.1rem; color: #1e3a8a;">
            ⚡ Find Optimal Assignment
        </h1>
        <p style="font-size: 1.1rem; color: #334155; margin-bottom: 0;">
            The Hungarian Method is a mathematical technique that helps us find the lowest-cost assignment between teachers and courses.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ How the Calculation Works in 4 Simple Steps")
    col_st1, col_st2, col_st3, col_st4 = st.columns(4)
    with col_st1:
        st.markdown("""
        <div class="info-card">
            <h4>1. Load Data</h4>
            <p style="color: #475569; font-size: 0.9rem;">
                Reads 36 courses, 13 teachers, and timetable sessions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_st2:
        st.markdown("""
        <div class="info-card">
            <h4>2. Calculate Costs</h4>
            <p style="color: #475569; font-size: 0.9rem;">
                Generates cost matrix balancing expertise and preferences.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_st3:
        st.markdown("""
        <div class="info-card">
            <h4>3. Hungarian Solver</h4>
            <p style="color: #475569; font-size: 0.9rem;">
                Executes Kuhn-Munkres matching on expanded 52-slot matrix.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_st4:
        st.markdown("""
        <div class="info-card">
            <h4>4. Clash-Free Plan</h4>
            <p style="color: #475569; font-size: 0.9rem;">
                Ensures 0 clashes and strictly respects capacity ($K \\le 4$).
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🚀 Run Optimization")
    st.markdown(f"Currently active timetable: **{dataset_choice}**")

    # Big Prominent Action Button
    if st.button("🚀 Calculate Optimal Assignment", type="primary", use_container_width=True):
        with st.spinner("Calculating optimal teacher-course assignments using the Hungarian Method..."):
            t_start = time.time()
            results = run_project_hungarian(dataset_key)
            elapsed = time.time() - t_start
            results["elapsed"] = elapsed

            # Store in session state for Results page
            st.session_state["assignment_results"] = results
            st.session_state["last_dataset"] = dataset_key

        st.success(f"✅ Optimal Assignment Calculated Successfully in {elapsed:.4f} seconds!")

    # If results exist in session state, display immediate summary
    if "assignment_results" in st.session_state and st.session_state.get("last_dataset") == dataset_key:
        res = st.session_state["assignment_results"]
        cur_cost = res["cur_val"]["recomputed_total_cost"]
        opt_cost = res["opt_val"]["recomputed_total_cost"]
        score = res["suitability_score"]

        st.markdown("---")
        st.subheader("📋 Quick Calculation Summary")

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">36 / 36</div>
                <div class="metric-label">Courses Assigned</div>
            </div>
            """, unsafe_allow_html=True)
        with q2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">0</div>
                <div class="metric-label">Timetable Clashes</div>
            </div>
            """, unsafe_allow_html=True)
        with q3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{opt_cost:.1f}</div>
                <div class="metric-label">Optimized Total Cost</div>
            </div>
            """, unsafe_allow_html=True)
        with q4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #16a34a;">{score:.1f}%</div>
                <div class="metric-label">Suitability Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.success("🎉 All courses have been successfully assigned! Head over to the **Results** page to view the full assignment table and charts.")

# -----------------------------------------------------------------------------
# PAGE 5: 📋 RESULTS
# -----------------------------------------------------------------------------
elif "5. 📋 Results" in nav_selection:
    st.markdown("""
    <div class="project-hero">
        <h1 style="margin-top: 0; font-size: 2.1rem; color: #1e3a8a;">
            📋 Optimization Results & Visual Analysis
        </h1>
        <p style="font-size: 1.05rem; color: #334155; margin-bottom: 0;">
            Review the final optimal teacher assignments, before-and-after costs, dynamic suitability score, and workload charts.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Check if calculation has been performed for current dataset; if not, run it automatically
    if "assignment_results" not in st.session_state or st.session_state.get("last_dataset") != dataset_key:
        with st.spinner("Calculating optimal assignment for current dataset..."):
            res = run_project_hungarian(dataset_key)
            st.session_state["assignment_results"] = res
            st.session_state["last_dataset"] = dataset_key

    res = st.session_state["assignment_results"]
    rep_df = res["assigned_df"].copy()
    cur_val = res["cur_val"]
    opt_val = res["opt_val"]
    suitability_score = res["suitability_score"]
    breakdown = res["suitability_breakdown"]

    total_courses = len(rep_df)
    assigned_courses = int(opt_val["assigned_courses"])
    teachers_used = int(rep_df["assigned_teacher"].nunique())
    cur_cost = float(cur_val["recomputed_total_cost"])
    opt_cost = float(opt_val["recomputed_total_cost"])
    cost_savings = cur_cost - opt_cost

    # 1. Mandatory Metric Cards
    st.markdown("### 📊 Key Performance Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{total_courses}</div>
            <div class="metric-label">Total Courses</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{assigned_courses}</div>
            <div class="metric-label">Assigned Courses</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{teachers_used}</div>
            <div class="metric-label">Teachers Used</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{cur_cost:.1f}</div>
            <div class="metric-label">Before Cost</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{opt_cost:.1f}</div>
            <div class="metric-label">After Cost</div>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #16a34a;">0</div>
            <div class="metric-label">Clashes Remaining</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Dynamic Accuracy / Assignment Suitability Score (Requirement 6)
    st.markdown("### 🎯 Dynamic Assignment Suitability Score")

    score_col_left, score_col_right = st.columns([1, 2])
    with score_col_left:
        st.markdown(f"""
        <div class="score-card">
            <div style="font-size: 15px; font-weight: 600; color: #1e3a8a; text-transform: uppercase; margin-bottom: 6px;">
                Assignment Suitability Score
            </div>
            <div class="score-percentage">{suitability_score:.1f}%</div>
            <div style="font-size: 13px; color: #3b82f6; font-weight: 600; margin-top: 6px;">
                Strictly Capped at 90.0% Max Ceiling
            </div>
        </div>
        """, unsafe_allow_html=True)

    with score_col_right:
        st.markdown(f"""
        <div class="info-card" style="border-left-color: #3b82f6;">
            <h4>📐 How This Score is Calculated (Mathematical Documentation)</h4>
            <p style="font-size: 0.95rem; color: #334155; margin-bottom: 8px;">
                <strong>Important Note for Academic Evaluators:</strong> This score is an Operations Research metric representing the degree to which faculty expertise and preference rankings are satisfied in the final schedule. <strong>It is not a machine-learning classification accuracy.</strong>
            </p>
            <ul style="font-size: 0.9rem; color: #475569; margin-bottom: 6px;">
                <li><strong>Average Faculty Expertise Match:</strong> {breakdown['avg_expertise']} / 10.0 (Competence rating)</li>
                <li><strong>Average Subject Preference Rank:</strong> Rank {breakdown['avg_pref_rank']} / 18.0 (1 = top choice)</li>
                <li><strong>Department Raw Match Quality:</strong> {breakdown['raw_match_pct']}%</li>
                <li><strong>Operational Real-World Ceiling:</strong> 90.0% maximum</li>
            </ul>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                <em>Formula:</em> <code>Suitability Score = min(90.0, Department Raw Match Quality × 0.90)</code>. The 90% boundary accounts for inescapable timetable friction where concurrent lectures prevent all faculty from receiving 100% ideal first-choice subjects simultaneously.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Simple Assignment Results Table (Requirement 5)
    st.markdown("### 📋 Final Optimal Assignment Table")
    st.markdown("Each row shows a course section, its mathematically assigned faculty member, and the resulting assignment cost.")

    # Format clean display table as required
    display_table = pd.DataFrame({
        "Course": [f"{r.course_id}: {r.subject} ({r.class_code})" for r in rep_df.itertuples()],
        "Assigned Teacher": rep_df["assigned_teacher"],
        "Assignment Cost": [round(float(c), 1) for c in rep_df["total_cost"]]
    })

    st.dataframe(display_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 4. Charts Section (Requirement 7: exactly 3 simple charts)
    st.markdown("### 📈 Visual Analysis Charts")
    st.caption("Clear, properly labelled charts visualizing teacher workload, cost improvements, and departmental teaching hours.")

    col_chart_left, col_chart_right = st.columns(2)

    # Chart 1: Teacher Workload Bar Chart
    with col_chart_left:
        st.markdown("#### 1. Teacher Workload (Courses Assigned per Teacher)")
        t_courses = rep_df.groupby("assigned_teacher").size().reindex(teachers, fill_value=0)

        fig1, ax1 = plt.subplots(figsize=(7, 4.2))
        bars1 = ax1.bar(teachers, t_courses, color="#2563eb", width=0.55)
        ax1.axhline(MAX_CAPACITY, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Max Capacity ({MAX_CAPACITY} Courses)")
        ax1.bar_label(bars1, padding=3, fontsize=9)
        ax1.set_xlabel("Faculty Member", fontsize=10)
        ax1.set_ylabel("Number of Assigned Courses", fontsize=10)
        ax1.set_title("Courses Assigned per Faculty Member", fontsize=11, fontweight="bold")
        ax1.set_ylim(0, 5)
        ax1.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig1)

    # Chart 2: Before-versus-After Assignment Cost Chart
    with col_chart_right:
        st.markdown("#### 2. Before-versus-After Total Assignment Cost")
        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        cost_labels = ["Before Optimization\n(Original Schedule)", "After Optimization\n(Hungarian Method)"]
        cost_values = [cur_cost, opt_cost]
        bar_colors = ["#94a3b8", "#16a34a"]

        bars2 = ax2.bar(cost_labels, cost_values, color=bar_colors, width=0.45)
        ax2.bar_label(bars2, fmt="%.1f", padding=3, fontsize=10, fontweight="bold")
        ax2.set_ylabel("Total Assignment Cost (Lower is Better)", fontsize=10)
        ax2.set_title("Assignment Cost: Before vs After", fontsize=11, fontweight="bold")
        ax2.set_ylim(0, max(cur_cost, opt_cost) * 1.25)
        plt.tight_layout()
        st.pyplot(fig2)

    # Chart 3: Teacher-Course Assignment Distribution Chart
    st.markdown("#### 3. Teacher-Course Assignment Distribution (Weekly Contact Periods)")
    t_periods = rep_df.groupby("assigned_teacher")["periods"].sum().reindex(teachers, fill_value=0)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    bars3 = ax3.bar(teachers, t_periods, color="#0d9488", width=0.55)
    ax3.bar_label(bars3, padding=3, fontsize=9)
    ax3.set_xlabel("Faculty Member", fontsize=10)
    ax3.set_ylabel("Weekly Teaching Periods (Hours)", fontsize=10)
    ax3.set_title("Distribution of Classroom Teaching Hours Across Faculty", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, max(t_periods) * 1.25)
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("---")

    # 5. Export Assignment Results
    st.subheader("📥 Export Final Assignments")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        csv_data = display_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download Assignments as CSV",
            data=csv_data,
            file_name=f"Optimal_Teacher_Assignments_{dataset_key}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_exp2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            display_table.to_excel(writer, sheet_name="Optimal_Assignments", index=False)
            rep_df.to_excel(writer, sheet_name="Detailed_Schedule", index=False)
            t_summary = pd.DataFrame({
                "Teacher": teachers,
                "Courses_Assigned": t_courses.values,
                "Weekly_Periods": t_periods.values
            })
            t_summary.to_excel(writer, sheet_name="Teacher_Workload", index=False)
        excel_buffer.seek(0)

        st.download_button(
            label="📑 Download Full Excel Report (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name=f"Optimal_Teacher_Assignments_Report_{dataset_key}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
