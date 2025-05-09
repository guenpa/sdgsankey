# app.py

# Required libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import io  # For reading uploaded files in memory
import os  # For checking default file paths

# --- Configuration ---
DEFAULT_NODE_COLOR     = "#CCCCCC"    # Grey for nodes.
DEFAULT_FLOW_COLOR     = "#009EDB"    # UN Blue for flows.
DEFAULT_DATA_PATH      = "./data/sankey_data.xlsx"
DEFAULT_POSITIONS_PATH = "./data/node_positions.csv"

# --- Page Config & Theme Enforcement ---
st.set_page_config(page_title="Interactive Sankey Editor", layout="wide")
# Hide main menu (including theme switcher)
st.markdown("""
    <style>
    #MainMenu { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
def is_valid_color(color_val):
    """Checks if a color value is valid (not None, NaN, or blank)."""
    return bool(color_val and pd.notna(color_val) and str(color_val).strip() != "")

def add_alpha(hex_color, alpha=1.0):
    """Convert a hex color (e.g., "#009EDB") to an rgba string with the given alpha."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

# --- Core Sankey Function (NEW Plotly version) ---
def create_sankey_diagram(data, positions_df=None):
    if data is None or data.empty:
        return go.Figure()

    # 1) Clean & prepare data
    df = data.copy()
    df.columns = df.columns.str.strip().str.lower()
    if "flow value" in df.columns:
        df = df.rename(columns={"flow value": "value"})
    df['source'] = df['source'].astype(str).str.strip()
    df['target'] = df['target'].astype(str).str.strip()
    df['value']  = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['source','target','value'])
    df['is_zero'] = df['value'] <= 0
    df['value']   = df['value'].apply(lambda x: 0.001 if x <= 0 else x)

    # 2) Determine valid nodes
    valid_nodes = sorted(set(df['source']).union(df['target']))

    # 3) Parse positions_df (if provided) into mappings + ordering
    if positions_df is not None and not positions_df.empty:
        pos = positions_df.copy()
        pos.columns = pos.columns.str.strip().str.lower()
        if 'node' in pos.columns:
            pos['node'] = pos['node'].astype(str).str.strip()
            # keep only nodes that appear in the flows
            pos = pos[pos['node'].isin(valid_nodes)]

            # build node order: as in CSV, then any missing nodes
            nodes = pos['node'].tolist()
            for n in valid_nodes:
                if n not in nodes:
                    nodes.append(n)

            # ensure required columns exist
            for col in ['x','y','node_color','incoming_flow_color','outgoing_flow_color']:
                if col not in pos.columns:
                    pos[col] = np.nan if col in ['x','y'] else ""

            node_colors_custom      = dict(zip(pos['node'], pos['node_color']))
            incoming_flow_map       = dict(zip(pos['node'], pos['incoming_flow_color']))
            outgoing_flow_map       = dict(zip(pos['node'], pos['outgoing_flow_color']))
            # build predefined positions (flip y for Plotly)
            xs = pd.to_numeric(pos.set_index('node').reindex(nodes)['x'], errors='coerce')
            ys = pd.to_numeric(pos.set_index('node').reindex(nodes)['y'], errors='coerce')
            predefined_positions    = {
                'x': xs.where(xs.notna(), None).tolist(),
                'y': [None if pd.isna(y) else 1-y for y in ys.tolist()]
            }
            arrangement = 'perpendicular'
        else:
            # missing 'node' column: fallback
            nodes                = valid_nodes
            node_colors_custom   = {}
            incoming_flow_map    = {}
            outgoing_flow_map    = {}
            predefined_positions = None
            arrangement          = 'snap'
            st.warning("⚠️ Positions file missing 'node' column; using defaults.")
    else:
        # no positions provided
        nodes                = valid_nodes
        node_colors_custom   = {}
        incoming_flow_map    = {}
        outgoing_flow_map    = {}
        predefined_positions = None
        arrangement          = 'snap'

    # 4) Compute node values & labels
    node_values = {
        n: max(
            df.loc[df['target']==n, 'value'].sum(),
            df.loc[df['source']==n, 'value'].sum(),
            0.001
        )
        for n in nodes
    }
    labels = [f"{n} ({node_values[n]:,.0f})" for n in nodes]

    # 5) Final node colors (safe .strip())
    final_node_colors = []
    for n in nodes:
        raw = node_colors_custom.get(n, "")
        c   = str(raw).strip()
        final_node_colors.append(c if (c and c.lower() != "nan") else DEFAULT_NODE_COLOR)

    # 6) Build link arrays
    idx     = {n:i for i,n in enumerate(nodes)}
    sources = [idx[s] for s in df['source']]
    targets = [idx[t] for t in df['target']]
    values  = df['value'].tolist()

    # 7) Link colors (with zero-flow alpha)
    link_colors = []
    for i, s in enumerate(df['source']):
        if df['is_zero'].iat[i]:
            link_colors.append(add_alpha(DEFAULT_FLOW_COLOR, 0.25))
        else:
            # outgoing preference
            raw_oc = outgoing_flow_map.get(s, "")
            oc     = str(raw_oc).strip()
            if oc and oc.lower() != "nan":
                link_colors.append(oc)
            else:
                # incoming fallback
                raw_ic = incoming_flow_map.get(df['target'].iat[i], "")
                ic     = str(raw_ic).strip()
                link_colors.append(ic if (ic and ic.lower() != "nan") else DEFAULT_FLOW_COLOR)

    # 8) Build the Plotly Sankey
    fig = go.Figure(go.Sankey(
        arrangement=arrangement,
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=final_node_colors,
            x=predefined_positions['x'] if predefined_positions else None,
            y=predefined_positions['y'] if predefined_positions else None,
            hovertemplate='Node: %{label}<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=[[df['source'].iloc[i], values[i]] for i in range(len(values))],
            hovertemplate='%{customdata[0]}: %{customdata[1]:,.0f}<extra></extra>'
        )
    ))

    # 9) Theme‐aware styling
    theme = st.get_option("theme.base")  # "light" or "dark"
    if theme == 'dark':
        bg     = '#000000'
        font_c = '#FFFFFF'
    else:
        bg     = '#FFFFFF'
        font_c = '#000000'

    fig.update_layout(
        font=dict(color=font_c, size=12),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        height=800,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig

# ----- STREAMLIT APP -----
st.title("📊 Interactive Sankey Diagram with Editable Data")

# Session state initialization
for key in ['data_df','positions_df','editable_data']:
    if key not in st.session_state:
        st.session_state[key] = None

# Load default data if present
if (st.session_state.data_df is None) and os.path.exists(DEFAULT_DATA_PATH):
    try:
        df = pd.read_excel(DEFAULT_DATA_PATH, sheet_name=0)
        st.session_state.data_df      = df
        st.session_state.editable_data = df.copy()
        st.info(f"✅ Loaded default data from {DEFAULT_DATA_PATH}")
    except Exception as e:
        st.warning(f"⚠️ Default data load failed: {e}")

# Load default positions if present
if (st.session_state.positions_df is None) and os.path.exists(DEFAULT_POSITIONS_PATH):
    try:
        st.session_state.positions_df = pd.read_csv(DEFAULT_POSITIONS_PATH)
        st.info(f"✅ Loaded default positions from {DEFAULT_POSITIONS_PATH}")
    except Exception as e:
        st.warning(f"⚠️ Default positions load failed: {e}")

# Chart placeholder
chart_placeholder = st.empty()

# --- Data Editor Section ---
if st.session_state.editable_data is not None:
    st.markdown("---")
    st.subheader("📝 Edit Flow Data (Live Updates Chart)")
    edited = st.data_editor(
        st.session_state.editable_data,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )
    st.session_state.editable_data = edited
else:
    st.markdown("---")
    st.warning("No data available. Upload an Excel file to start.")

# --- Uploaders ---
st.markdown("---")
st.subheader("📂 Upload New Files")
c1, c2 = st.columns(2)
with c1:
    up = st.file_uploader("📥 Excel Data File", type=["xlsx"], key="up_data")
    if up:
        try:
            dfn = pd.read_excel(io.BytesIO(up.getvalue()), sheet_name=0)
            st.session_state.data_df      = dfn
            st.session_state.editable_data = dfn.copy()
            st.success("✅ Data file loaded.")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"❌ {e}")
with c2:
    up2 = st.file_uploader("🎨 Positions CSV", type=["csv"], key="up_pos")
    if up2:
        try:
            p2 = pd.read_csv(io.BytesIO(up2.getvalue()))
            st.session_state.positions_df = p2
            st.success("✅ Positions file loaded.")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"❌ {e}")

# --- Render Chart ---
if st.session_state.editable_data is not None and not st.session_state.editable_data.empty:
    try:
        fig = create_sankey_diagram(
            st.session_state.editable_data,
            st.session_state.positions_df
        )
        with chart_placeholder.container():
            st.subheader("📈 Live Sankey Preview")
            st.plotly_chart(fig, use_container_width=True, key="sankey_chart")
    except Exception as e:
        with chart_placeholder.container():
            st.error(f"❌ Error generating chart: {e}")
            st.exception(e)
elif st.session_state.data_df is None:
    with chart_placeholder.container():
        fn1 = os.path.basename(DEFAULT_DATA_PATH)
        fn2 = os.path.basename(DEFAULT_POSITIONS_PATH)
        st.info(f"Place '{fn1}' and/or '{fn2}' in ./data/ to begin.")

# --- Instructions Footer ---
st.markdown("---")
st.markdown("""
**Instructions:**
1. Default data may be loaded if you place files in `./data/`.
2. Upload new files using the buttons above.
3. Edit flow data live—the chart updates automatically.
""")
