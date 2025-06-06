import streamlit as st
import pandas as pd
import os
import openai
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# --- GPT SUGGESTION FUNCTION ---
def gpt_suggest(prompt, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in organizational roles, automation, and job design."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.6
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# --- PAGE SETUP ---
st.set_page_config(layout="wide")

# --- FILE PATHS & COLUMNS ---
CSV_PATH = "roles.csv"
COLUMNS = [
    "Role",
    "Responsibilities",
    "Overlaps With",
    "Automatable Tasks",
    "Automation %",
    "Potential Hours Saved"
]

# --- LOAD/INIT DATA ---
if "roles" not in st.session_state:
    if os.path.exists(CSV_PATH):
        st.session_state["roles"] = pd.read_csv(CSV_PATH)
    else:
        st.session_state["roles"] = pd.DataFrame(columns=COLUMNS)
roles = st.session_state["roles"]

# --- Org Completion Progress Bar ---
def calculate_completion(roles_df):
    if roles_df.empty:
        return 0
    if "Responsibilities" not in roles_df or "Automatable Tasks" not in roles_df:
        return 0
    filled_responsibilities = roles_df["Responsibilities"].fillna("").astype(str).str.strip() != ""
    filled_tasks = roles_df["Automatable Tasks"].fillna("").astype(str).str.strip() != ""
    complete = (filled_responsibilities & filled_tasks).sum()
    total = len(roles_df)
    percent = int((complete / total) * 100) if total > 0 else 0
    return max(0, min(percent, 100))

org_completion = calculate_completion(roles)



# --- SIDEBAR ---
with st.sidebar:
    st.image("matriqx-logo.png", width=200)  # Make sure the file is in the same folder as app.py

with st.sidebar:
    st.markdown("#### 🎯 Org Design Completion")
    st.progress(org_completion)
    st.markdown(f"<b>{org_completion}% complete</b>", unsafe_allow_html=True)
    
    missing_responsibilities = roles[roles["Responsibilities"].fillna("").astype(str).str.strip() == ""]
    missing_tasks = roles[roles["Automatable Tasks"].fillna("").astype(str).str.strip() == ""]
    
    if not roles.empty:
        st.markdown(
            f"""
            <div style='padding: 8px 10px; background: #f6fafd; border-radius: 6px; margin-bottom: 10px;'>
            <ul style='margin-bottom:8px; font-size: 1rem;'>
              <li><b>{len(missing_responsibilities)}</b> roles missing <b>responsibilities</b></li>
              <li><b>{len(missing_tasks)}</b> roles missing <b>automatable tasks</b></li>
            </ul>
            <span style='color: #2949a7;'>Tip: Each role should have both responsibilities and automatable tasks filled.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        if len(missing_responsibilities) > 0:
            st.markdown(
                "**Roles missing responsibilities:**<br>" +
                ", ".join([f"`{role}`" for role in missing_responsibilities["Role"]]),
                unsafe_allow_html=True
            )
        if len(missing_tasks) > 0:
            st.markdown(
                "**Roles missing automatable tasks:**<br>" +
                ", ".join([f"`{role}`" for role in missing_tasks["Role"]]),
                unsafe_allow_html=True
            )
    else:
        st.info("No roles yet. Add a role to get started!")

import streamlit as st
from openai import OpenAI

# Load your API key securely from Streamlit secrets
OPENAI_API_KEY = st.secrets["openai_api_key"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Helper function to query ChatGPT
def ask_chatgpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant for organizational design and automation."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.7,
    )
    return response.choices[0].message.content

# Make sure 'roles' DataFrame exists in session state
roles = st.session_state.get("roles", None)

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🤖 MatriQx AI")

    user_question = st.text_area("Ask MatriQx AI about your org, roles, or automation:", key="copilot_input")

    if st.button("Ask MatriQx Bot"):
        if user_question.strip():
            with st.spinner("MatriQx AI is thinking..."):
                # Prepare summary context for AI prompt
                if roles is not None:
                    missing_resp = roles[roles["Responsibilities"].fillna("").str.strip() == ""]
                    missing_tasks = roles[roles["Automatable Tasks"].fillna("").str.strip() == ""]

                    missing_resp_roles = [str(r) for r in missing_resp["Role"].dropna() if str(r).lower() != "nan"]
                    missing_task_roles = [str(r) for r in missing_tasks["Role"].dropna() if str(r).lower() != "nan"]

                    context_summary = (
                        f"There are {len(roles)} roles. "
                        f"Roles missing responsibilities: {', '.join(missing_resp_roles) if missing_resp_roles else 'None'}. "
                        f"Roles missing automatable tasks: {', '.join(missing_task_roles) if missing_task_roles else 'None'}."
                    )
                else:
                    context_summary = "No roles data available."

                prompt = (
                    f"You are MatriQx AI Copilot, expert in org design and automation.\n"
                    f"Context: {context_summary}\n\n"
                    f"User question: {user_question}\n"
                    f"Answer concisely and provide actionable insights."
                )

                answer = ask_chatgpt(prompt)
                st.markdown(f"**MatriQx AI:** {answer}")
        else:
            st.info("Type a question above and click 'Ask Copilot'.")

with st.sidebar:
    st.header("Add New Role")
    new_role = st.text_input("Role Name", key="new_role_name")
    if st.button("Add Role", key="add_role_button"):
        if new_role and new_role not in roles["Role"].values:
            new_row = pd.DataFrame([{
                "Role": new_role,
                "Responsibilities": "",
                "Overlaps With": "",
                "Automatable Tasks": "",
                "Automation %": 0,
                "Potential Hours Saved": 0
            }])
            st.session_state["roles"] = pd.concat([roles, new_row], ignore_index=True)
            st.session_state["roles"].to_csv(CSV_PATH, index=False)
            st.success(f"Role '{new_role}' added!")
            st.rerun()
        elif new_role in roles["Role"].values:
            st.warning("This role already exists. Try a different name.")
        else:
            st.warning("Please enter a role name.")
    st.markdown("---")
    st.header("CSV Import / Export")
    csv = roles.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Roles as CSV",
        data=csv,
        file_name='roles_export.csv',
        mime='text/csv'
    )
    uploaded_file = st.file_uploader("Upload CSV to Import Roles", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if all(col in df.columns for col in COLUMNS):
                st.session_state["roles"] = df
                st.session_state["roles"].to_csv(CSV_PATH, index=False)
                st.success("Roles imported successfully!")
                st.rerun()
            else:
                st.error(f"CSV missing required columns. Expected: {COLUMNS}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# --- MAIN APP ---
import streamlit as st




st.title("MatriQx RoleScope™ – Living Role Framework")

if roles.empty:
    st.info("No roles yet. Add a role in the sidebar to get started!")
else:
    # --- Role selection ---
    role_names = roles["Role"].tolist()
    selected_role = st.selectbox("Select a Role", ["-- Select --"] + role_names, key="role_select")
    if selected_role and selected_role != "-- Select --":
        idx = roles.index[roles["Role"] == selected_role][0]
        # --- Templates ---
        role_templates = {
            "CTO": "1. Set technology vision and roadmap...\n2. Lead architecture...",
            "QA Engineer": "1. Write and maintain test cases...\n2. Automate regression testing...",
            "Data Scientist": "1. Analyze datasets...\n2. Build predictive models..."
        }
        template_role = st.selectbox(
            "Insert a template for this role",
            ["-- Select --"] + list(role_templates.keys()),
            key="select_template"
        )
        if template_role and template_role in role_templates:
            if st.button(f"Use {template_role} Template for Responsibilities"):
                st.session_state["resp_input"] = role_templates[template_role]
                st.success(f"Template loaded for {template_role}!")
        # --- Responsibilities ---
        if "resp_input" not in st.session_state:
            st.session_state["resp_input"] = roles.at[idx, "Responsibilities"]
        st.markdown(f"### {selected_role}: Responsibilities")
        if st.button("Suggest Responsibilities with MatriQx AI"):
            prompt = f"List the 5-8 most important responsibilities for a {selected_role} in a tech company, as bullet points."
            suggestion = gpt_suggest(prompt, st.secrets["openai_api_key"])
            st.session_state["ai_suggestion"] = suggestion
        if "ai_suggestion" in st.session_state:
            st.info(st.session_state["ai_suggestion"])
            if st.button("Use Suggested Responsibilities"):
                st.session_state["resp_input"] = st.session_state["ai_suggestion"]
        # Reset responsibilities field after save
        if st.session_state.get("reset_resp_input"):
            st.session_state["resp_input"] = ""
            del st.session_state["reset_resp_input"]
        resp = st.text_area(
            "Edit Responsibilities Below",
            value=st.session_state.get("resp_input", ""),
            key="resp_input"
        )
        if st.button("Save Responsibilities"):
            roles.at[idx, "Responsibilities"] = resp
            st.session_state["roles"] = roles
            st.session_state["roles"].to_csv(CSV_PATH, index=False)
            st.success(f"Responsibilities for {selected_role} updated!")
            st.session_state["reset_resp_input"] = True  # Set reset flag
            st.rerun()
        # --- Automatable Tasks ---
        # Reset automatable tasks field after save
        if st.session_state.get("reset_tasks_input"):
            st.session_state["tasks_input"] = ""
            del st.session_state["reset_tasks_input"]

        st.markdown(f"### {selected_role}: Automatable Tasks")
        if st.button("Suggest Automatable Tasks with MatriQx AI"):
            prompt = f"List 3-7 routine or repetitive tasks for a {selected_role} in a tech company that could be automated using modern tools or AI."
            tasks_suggestion = gpt_suggest(prompt, st.secrets["openai_api_key"])
            st.session_state["ai_tasks_suggestion"] = tasks_suggestion
        if "ai_tasks_suggestion" in st.session_state:
            st.info(st.session_state["ai_tasks_suggestion"])
            if st.button("Use Suggested Automatable Tasks"):
                st.session_state["tasks_input"] = st.session_state["ai_tasks_suggestion"]
        tasks = st.text_area(
            "Edit Automatable Tasks Below",
            value=st.session_state.get("tasks_input", roles.at[idx, "Automatable Tasks"]),
            key="tasks_input"
        )
        if st.button("Save Automatable Tasks"):   # <--- This line is now correctly indented
            roles.at[idx, "Automatable Tasks"] = tasks
            st.session_state["roles"] = roles
            st.session_state["roles"].to_csv(CSV_PATH, index=False)
            st.success(f"Automatable tasks for {selected_role} updated!")
            st.session_state["reset_tasks_input"] = True  # Set reset flag
            st.rerun()

        # --- Overlaps With ---
        all_other_roles = [r for r in role_names if r != selected_role]
        default_overlaps = [x.strip() for x in str(roles.at[idx, "Overlaps With"]).split(";") if x.strip() and x.strip() in all_other_roles]
        overlaps_selected = st.multiselect(
            f"{selected_role}: Overlaps With (select any roles that share responsibilities or tasks)",
            all_other_roles,
            default=default_overlaps,
            key=f"overlap_{selected_role}"
        )
        if st.button("Save Overlaps With"):
            roles.at[idx, "Overlaps With"] = "; ".join(overlaps_selected)
            st.session_state["roles"] = roles
            st.session_state["roles"].to_csv(CSV_PATH, index=False)
            st.success(f"Overlaps for {selected_role} updated!")
        # --- All Roles Overview (Editable) ---
        st.markdown("#### 📝 All Roles Overview (Editable)")
        editable_roles = st.data_editor(
            roles,
            use_container_width=True,
            num_rows="dynamic",
            key="editable_roles_table"
        )
        if st.button("💾 Save Table Changes"):
            st.session_state["roles"] = editable_roles
            st.session_state["roles"].to_csv(CSV_PATH, index=False)
            st.success("Saved edits to table!")
            st.rerun()
	
    else:
        st.info("Select a role to begin editing responsibilities and tasks.")
  


  # --- Automation Opportunity Visuals ---
    st.markdown("### 📊 Automation Opportunity Visuals")
    if not roles.empty and "Potential Hours Saved" in roles.columns and "Automation %" in roles.columns:
        fig1 = px.bar(
            roles,
            x="Role",
            y="Potential Hours Saved",
            title="Potential Hours Saved per Month by Role",
            color="Potential Hours Saved",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.pie(
            roles,
            names="Role",
            values="Automation %",
            title="Automation % by Role",
            hole=0.4
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Add roles and automation data to see visualizations!")
   
 # --- Role Overlap Matrix (Interactive) ---
    st.markdown("#### Role Overlap Matrix (Interactive)")
    if "Overlaps With" in roles.columns:
        overlap_matrix = pd.DataFrame(0, index=role_names, columns=role_names)
        for i, row in roles.iterrows():
            if isinstance(row["Overlaps With"], str):
                overlaps = [r.strip() for r in str(row["Overlaps With"]).split(";") if r.strip() and r.strip() in role_names]
                for o in overlaps:
                    if o in overlap_matrix.columns:
                        overlap_matrix.at[row["Role"], o] = 1
        # Interactivity
        import plotly.graph_objects as go
        z = overlap_matrix.values.tolist()
        heatmap = go.Figure(
            data=go.Heatmap(
                z=z,
                x=role_names,
                y=role_names,
                colorscale=[ [0, "#f6fafd"], [1, "#163968"] ],
                showscale=False,
                hoverongaps=False
            )
        )
        heatmap.update_layout(
            autosize=True,
            height=420,
            margin=dict(l=140, r=30, t=10, b=30),
            xaxis=dict(tickangle=-30, tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12))
        )
        st.plotly_chart(heatmap, use_container_width=True)
import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import tempfile
import os

# --- Build Overlap Pairs ---
overlap_pairs = []
for r1 in role_names:
    for r2 in role_names:
        if r1 != r2 and overlap_matrix.at[r1, r2] == 1:
            overlap_pairs.append(tuple(sorted([r1, r2])))

# Remove duplicates
overlap_pairs = list(set(overlap_pairs))

# Clean role names to ensure all are valid strings and not null
roles = roles[roles["Role"].notnull()]
roles["Role"] = roles["Role"].astype(str)

role_names = roles["Role"].tolist()


# --- NETWORK GRAPH ---
st.markdown("## 🕸️ Role Overlap Network")
with st.spinner("Building network..."):
    G = nx.Graph()
    for role in role_names:
        G.add_node(role)
    for r1, r2 in overlap_pairs:
        G.add_edge(r1, r2)

    # Pyvis network setup
    net = Network(height='500px', width='100%', bgcolor='#ffffff', font_color='black')
    net.from_nx(G)
    net.repulsion(node_distance=160, spring_length=200)
    for node in net.nodes:
        node['title'] = node['label']  # Tooltip
        node['color'] = '#2e77d0'      # MatriQx blue

    for edge in net.edges:
        edge['color'] = '#7f8c8d'      # Muted grey

    # Save and show network HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        st.components.v1.html(open(tmp_file.name, 'r', encoding='utf-8').read(), height=540)
        

# --- ACCORDION: Overlaps by Role ---
st.markdown("## 📋 Overlaps by Role")
for role in role_names:
    overlaps = [r2 for (r1, r2) in overlap_pairs if r1 == role] + [r1 for (r1, r2) in overlap_pairs if r2 == role]
    overlaps = sorted(set(overlaps))
    with st.expander(f"{role} ({len(overlaps)} overlap{'s' if len(overlaps)!=1 else ''})"):
        if overlaps:
            st.markdown(
                "\n".join([f"- **{role}** shares a key responsibility/task with **{o}**" for o in overlaps]),
                unsafe_allow_html=True
            )
        else:
            st.markdown("*No overlaps for this role.*")

# --- Optionally: DataFrame as expandable raw data ---
with st.expander("Show Overlap Matrix (Raw Table)"):
    st.dataframe(overlap_matrix)


import streamlit as st

st.markdown("## 🚀 Next Steps")

next_steps = []

# Example 1: Identify roles missing key data
missing_responsibilities = roles[roles["Responsibilities"].astype(str).str.strip() == ""]
if not missing_responsibilities.empty:
    next_steps.append(f"📝 Add responsibilities for: {', '.join(missing_responsibilities['Role'].tolist())}")

missing_tasks = roles[roles["Automatable Tasks"].astype(str).str.strip() == ""]
if not missing_tasks.empty:
    next_steps.append(f"⚙️ Add automatable tasks for: {', '.join(missing_tasks['Role'].tolist())}")

# Example 2: Suggest increasing automation where it's low
low_automation = roles[roles["Automation %"].fillna(0).astype(float) < 20]
if not low_automation.empty:
    next_steps.append(f"🤖 Review automation opportunities for: {', '.join(low_automation['Role'].tolist())} (low automation %)")

# Example 3: High overlaps
for idx, row in roles.iterrows():
    overlaps = [r.strip() for r in str(row["Overlaps With"]).split(";") if r.strip()]
    if len(overlaps) > 3:
        next_steps.append(f"🔗 Consider reviewing overlapping responsibilities for {row['Role']} (overlaps with: {', '.join(overlaps)})")

# Example 4: General suggestions
if len(roles) < 5:
    next_steps.append("➕ Add more roles for a richer analysis!")
if not next_steps:
    next_steps.append("🎉 All set! Review your insights or export your data.")

for step in next_steps:
    st.info(step)
