# app.py

# Required libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import io  # Needed for reading uploaded files in memory
import os  # Needed for checking default file paths

# --- Configuration ---
DEFAULT_NODE_COLOR = "#CCCCCC"    # Grey for nodes.
DEFAULT_FLOW_COLOR = "#009EDB"    # UN Blue for flows.
DEFAULT_DATA_PATH = "./data/sankey_data.xlsx"
DEFAULT_POSITIONS_PATH = "./data/node_positions.csv"

# --- Page Config & Theme Enforcement ---
st.set_page_config(page_title="Interactive Sankey Editor", layout="wide")
# Disable theme switcher by hiding main menu (includes theme option)
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions ---
def is_valid_color(color_val):
    """Checks if a color value is valid (not None, NaN, or empty/whitespace string)."""
    return bool(color_val and pd.notna(color_val) and str(color_val).strip() != "")

def add_alpha(hex_color, alpha=1.0):
    """
    Convert a hex color (e.g., "#009EDB") to an rgba string with the given alpha value.
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

# --- Core Sankey Function ---
def create_sankey_diagram(data, positions_df=None):
    """
    Generate a Plotly Sankey diagram using the provided data
    and optional positions dataframe for nodes and flow colors,
    implementing the priority logic for flow colors.
    """
    if data is None or data.empty:
        return go.Figure()

    df = data.copy()
    df.columns = df.columns.str.strip().str.lower()
    if "flow value" in df.columns:
        df = df.rename(columns={"flow value": "value"})

    # Ensure required columns
    required_cols = ['source', 'target', 'value']
    if not all(col in df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
        return go.Figure()

    # Clean & flag zeros
    df['source'] = df['source'].astype(str).str.strip()
    df['target'] = df['target'].astype(str).str.strip()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['source','target','value'])
    df['is_zero'] = df['value'] <= 0
    df['value'] = df['value'].apply(lambda x: 0.001 if x <= 0 else x)

    # Nodes and optional positions
    valid_nodes = sorted(set(df['source']) | set(df['target']))
    node_colors_map = {}
    in_colors = {}
    out_colors = {}
    node_x = node_y = None
    arrangement = 'snap'

    if positions_df is not None and not positions_df.empty:
        pos = positions_df.copy()
        pos.columns = pos.columns.str.strip().str.lower()
        if 'node' in pos.columns:
            pos['node'] = pos['node'].astype(str).str.strip()
            pos = pos[pos['node'].isin(valid_nodes)]
            if not pos.empty:
                order = list(pos['node'])
                order += [n for n in valid_nodes if n not in order]
                valid_nodes = order
                for col in ['x','y','node_color','incoming_flow_color','outgoing_flow_color']:
                    if col not in pos.columns:
                        pos[col] = np.nan if col in ['x','y'] else ''
                idx = pos.set_index('node')
                node_colors_map = idx['node_color'].to_dict()
                in_colors = idx['incoming_flow_color'].to_dict()
                out_colors = idx['outgoing_flow_color'].to_dict()
                xs = pd.to_numeric(idx.reindex(valid_nodes)['x'], errors='coerce')
                ys = pd.to_numeric(idx.reindex(valid_nodes)['y'], errors='coerce')
                if xs.notna().any() and ys.notna().any():
                    node_x = xs.where(xs.notna(), None).tolist()
                    node_y = ys.where(ys.notna(), None).tolist()
                    arrangement = 'perpendicular'
        else:
            st.warning("⚠️ Positions file missing 'node' column; ignoring.")

    # Build mappings & labels
    label_idx = {n:i for i,n in enumerate(valid_nodes)}
    node_vals = {}
    final_colors = []
    for n in valid_nodes:
        inc = df.loc[df['target']==n, 'value'].sum()
        out = df.loc[df['source']==n, 'value'].sum()
        node_vals[n] = max(inc, out, 0.001)
        c = node_colors_map.get(n)
        final_colors.append(c if is_valid_color(c) else DEFAULT_NODE_COLOR)
    labels = [f"{n} ({node_vals[n]:,.0f})" for n in valid_nodes]

    links_df = df[df['source'].isin(label_idx) & df['target'].isin(label_idx)]
    if links_df.empty:
        st.warning("⚠️ No valid flows to display.")
        return go.Figure()
    srcs = [label_idx[s] for s in links_df['source']]
    tgts = [label_idx[t] for t in links_df['target']]
    vals = links_df['value'].tolist()

    link_colors = []
    for _,r in links_df.iterrows():
        if r['is_zero']:
            link_colors.append(add_alpha(DEFAULT_FLOW_COLOR, 0.25))
        else:
            oc = out_colors.get(r['source'])
            if is_valid_color(oc):
                link_colors.append(oc)
            else:
                ic = in_colors.get(r['target'])
                link_colors.append(ic if is_valid_color(ic) else DEFAULT_FLOW_COLOR)

    # Theme-based contrast
    theme = st.get_option("theme.base")  # "light" or "dark"
    if theme == 'dark':
        bg = '#000000'
        font_c = '#FFFFFF'
    else:
        bg = '#FFFFFF'
        font_c = '#000000'

    # Build the Sankey figure
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement=arrangement,
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=labels,
                    color=final_colors,
                    x=node_x,
                    y=node_y,
                    hovertemplate='Node: %{label}<extra></extra>'
                ),
                link=dict(
                    source=srcs,
                    target=tgts,
                    value=vals,
                    color=link_colors,
                    hovertemplate='Flow: %{value:,.0f}<extra></extra>'
                ),
                textfont=dict(color=font_c, size=12)
            )
        ]
    )
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

# Load defaults
if os.path.exists(DEFAULT_DATA_PATH):
    try:
        df = pd.read_excel(DEFAULT_DATA_PATH, sheet_name=0)
        st.session_state.data_df = df
        st.session_state.editable_data = df.copy()
    except Exception as e:
        st.warning(f"⚠️ Default data load failed: {e}")
if os.path.exists(DEFAULT_POSITIONS_PATH):
    try:
        st.session_state.positions_df = pd.read_csv(DEFAULT_POSITIONS_PATH)
    except Exception as e:
        st.warning(f"⚠️ Default positions load failed: {e}")

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
    st.warning("No data available. Upload a file below.")

# --- Uploaders ---
st.markdown("---")
st.subheader("📂 Upload New Files")
c1, c2 = st.columns(2)
with c1:
    up = st.file_uploader("📥 Excel Data File", type=["xlsx"], key="up_data")
    if up:
        try:
            dfn = pd.read_excel(io.BytesIO(up.getvalue()), sheet_name=0)
            st.session_state.data_df = dfn
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
        fig = create_sankey_diagram(st.session_state.editable_data, st.session_state.positions_df)
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
        st.info(f"Place '{fn1}' or '{fn2}' in ./data/ to begin.")
