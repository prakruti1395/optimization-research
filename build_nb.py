"""
Builds and updates the definitive academic Jupyter Notebook: OT_Teacher_Assignment_Hungarian.ipynb
Imports the validated core package modules to demonstrate Hungarian, MILP, and GA.
"""
import json

cells = []

def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': [l + '\n' for l in text.split('\n')]}

def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [l + '\n' for l in text.split('\n')]}

# Title
cells.append(md("""# Optimal Assignment of Teachers to Courses — Hungarian Method, MILP & GA Comparison
**Optimization Techniques (OT) — Multi-Model Assignment, Integer Programming & Metaheuristics**

- **Dataset:** Integrated MCA + MSc (LJKU) Complete Timetable (Full Semester Week: Monday–Friday including Wednesday, 277 total sessions).
- **Core Models:**
  1. **Hungarian Algorithm ($O(n^3)$ Kuhn-Munkres):** Exact bipartite matching on augmented cost matrix with teacher slot capacities and iterative conflict repair.
  2. **Mixed-Integer Linear Program (MILP):** Exact mathematical program with simultaneous hard anti-clash constraints and period capacity caps solved via HiGHS.
  3. **Genetic Algorithm (GA):** Discrete population-based metaheuristic with tournament selection, uniform crossover, targeted clash mutation, and elitism.
- **Objective Cost:**
  $$\\min \\sum_{i} \\sum_{j} C_{ij} x_{ij}$$
  where $C_{ij} = w_{\\text{skill}} \\times (10 - \\text{Skill}_{ij}) + w_{\\text{pref}} \\times \\text{PrefRank}_{ij} + \\lambda \\cdot k$."""))

# Cell 1: Imports & Setup
cells.append(code("""import os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
if "." not in sys.path:
    sys.path.insert(0, ".")

import core

os.makedirs("figs", exist_ok=True)
sns.set_theme(style="whitegrid")
print("Environment and core modules loaded successfully.")"""))

# Cell 2: Data Loading & Exploration
cells.append(md("""## 1. Data Ingestion & Course Section Extraction
We load the complete weekly schedule (277 contact sessions), faculty skill ratings (1–10), faculty subject preferences (1–18), and subject abbreviation lookups."""))

cells.append(code("""df_tt, df_skills, df_prefs, df_abbr = core.load_raw_data("expanded")
courses = core.extract_courses(df_tt, df_abbr)
teachers = sorted(df_tt["teacher"].unique())
clash_pairs, course_sessions = core.build_clash_graph(df_tt, courses)

print(f"Total sessions: {len(df_tt)} across {df_tt['day'].nunique()} days.")
print(f"Course sections: {len(courses)}")
print(f"Faculty members ({len(teachers)}): {teachers}")
print(f"Conflicting course pairs sharing time slots: {len(clash_pairs)}")
courses.head()"""))

# Cell 3: Faculty Expertise Heatmap
cells.append(md("""## 2. Faculty Expertise Ratings & Subject Preferences
Visualizing domain qualifications across faculty members and course subjects."""))

cells.append(code("""plt.figure(figsize=(12, 5))
sns.heatmap(df_skills, annot=True, cmap="YlGnBu", cbar_kws={"label": "Survey Skill Rating (1-10)"})
plt.title("Faculty Expertise Ratings by Subject (Survey Data)")
plt.tight_layout()
plt.savefig("figs/expertise_heatmap.png", dpi=160)
plt.show()"""))

# Cell 4: Cost Matrix Formulation & Viva 5x5 Example
cells.append(md("""## 3. Mathematical Cost Formulation & Step-by-Step Viva Hand-Worked Demo
Examiners often request hand-worked verification of the Kuhn-Munkres reduction steps:
1. **Row Reduction:** Subtract row minimum from each row.
2. **Column Reduction:** Subtract column minimum from each column.
3. **Independent Matching:** Optimal matching on zeros."""))

cells.append(code("""C_base = core.compute_base_cost_matrix(courses, teachers, df_skills, df_prefs, w_skill=2.0, w_pref=0.5)

# Step-by-step 5x5 slice
viva_dict = core.get_viva_step_by_step_matrices(
    C_base[:5, :5],
    row_labels=[f"{courses.iloc[i].course_id} ({courses.iloc[i].subject})" for i in range(5)],
    col_labels=[f"Prof. {t}" for t in teachers[:5]]
)
print("Step 0: Initial 5x5 Cost Matrix Slice:")
display(viva_dict["step0_initial"])

print("\\nStep 1: Row-Reduced Matrix:")
display(viva_dict["step1_row_reduced"])

print("\\nStep 2: Column-Reduced Matrix:")
display(viva_dict["step2_col_reduced"])

print("\\nOptimal Independent Matching on 5x5 Slice:")
display(viva_dict["optimal_matching"])"""))

# Cell 5: Model Execution & Benchmarking
cells.append(md("""## 4. Multi-Model Optimization Execution
Running Raw Hungarian, Repaired Hungarian, MILP, and GA on identical data."""))

cells.append(code("""comp_df, store = core.run_comprehensive_comparison(
    courses, teachers, df_tt, C_base, clash_pairs,
    lam=2.0, cap=4, max_weekly_periods=28.0,
    ga_pop_size=100, ga_generations=120, seed=42
)
display(comp_df)"""))

# Cell 6: Visual Comparison Plots
cells.append(md("""## 5. Visual Comparison: Cost, Clashes, and Runtime"""))

cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.barplot(data=comp_df, x="Method", y="Total Assignment Cost", ax=axes[0], palette="Blues_d")
axes[0].tick_params(axis="x", rotation=30)
axes[0].set_title("Total Assignment Cost")

sns.barplot(data=comp_df, x="Method", y="Timetable Clashes", ax=axes[1], palette="Reds_d")
axes[1].tick_params(axis="x", rotation=30)
axes[1].set_title("Timetable Clashes")

sns.barplot(data=comp_df, x="Method", y="Execution Time (s)", ax=axes[2], palette="Greens_d")
axes[2].tick_params(axis="x", rotation=30)
axes[2].set_title("Execution Runtime (s)")

plt.tight_layout()
plt.savefig("figs/cost_comparison.png", dpi=160)
plt.show()"""))

# Cell 7: Workload Distribution
cells.append(md("""## 6. Faculty Workload Contact Hours Analysis"""))

cells.append(code("""milp_df = store["milp"]["dataframe"]
cur_periods = df_tt.groupby("teacher").size().reindex(teachers, fill_value=0)
milp_periods = milp_df.groupby("assigned_teacher")["periods"].sum().reindex(teachers, fill_value=0)

plt.figure(figsize=(11, 4))
x = np.arange(len(teachers))
plt.bar(x - 0.2, cur_periods, width=0.4, label="Current Timetable Periods", color="#9aa5b1")
plt.bar(x + 0.2, milp_periods, width=0.4, label="MILP Balanced Periods", color="#2a9d8f")
plt.xticks(x, teachers)
plt.ylabel("Weekly Classroom Periods")
plt.title("Faculty Teaching Load: Current vs MILP")
plt.axhline(28, color="red", linestyle="--", label="Weekly Cap (28 Periods)")
plt.legend()
plt.tight_layout()
plt.savefig("figs/workload.png", dpi=160)
plt.show()"""))

# Cell 8: Sensitivity Analysis
cells.append(md("""## 7. Multi-Parameter Sensitivity Analysis"""))

cells.append(code("""sens_df = core.run_sensitivity_analysis(
    courses, teachers, df_tt, df_skills, df_prefs, clash_pairs,
    parameter="lambda", values=[0.0, 1.0, 2.0, 3.0, 5.0, 8.0],
    default_lam=2.0, default_cap=4, default_max_periods=28.0, solver="milp"
)
display(sens_df)

plt.figure(figsize=(8, 4))
plt.plot(sens_df["Value"], sens_df["Total Cost"], "o-", color="#2a9d8f", label="Total Cost", lw=2)
plt.plot(sens_df["Value"], sens_df["Base Cost"], "s--", color="#e76f51", label="Base Cost", lw=2)
plt.xlabel("Workload Penalty Weight (lambda)")
plt.ylabel("Cost Value")
plt.title("Cost Sensitivity to Workload Balancing Weight")
plt.legend()
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig("figs/sensitivity.png", dpi=160)
plt.show()"""))

# Cell 9: Teacher Absence Simulation
cells.append(md("""## 8. What-If Teacher Absence Simulation (Leave-One-Out)"""))

cells.append(code("""absence_records = []
for absent_t in ["NVS", "MOG", "JSD"]:
    abs_res = core.simulate_teacher_absence(
        courses, teachers, df_tt, df_skills, df_prefs, absent_t, clash_pairs,
        solver="milp", max_weekly_periods=32.0
    )
    realloc_summary = "; ".join([f"{r.course_id} ({r.subject}) -> {r.reallocated_teacher}" for r in abs_res["reallocated_courses"].itertuples()])
    absence_records.append({
        "Absent Teacher": absent_t,
        "Total Cost": round(abs_res["total_cost"], 2),
        "Avg Expertise": abs_res["avg_expertise"],
        "Clashes": abs_res["clashes"],
        "Reallocated Courses": realloc_summary
    })

display(pd.DataFrame(absence_records))"""))

# Build nb JSON
nb_data = {
    'cells': cells,
    'metadata': {
        'language_info': {'name': 'python'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open("OT_Teacher_Assignment_Hungarian.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_data, f, indent=1)

print("Updated notebook OT_Teacher_Assignment_Hungarian.ipynb built successfully!")
