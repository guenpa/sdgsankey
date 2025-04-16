# Required libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import io # Needed for reading uploaded files in memory
import os # Needed for checking default file paths

# --- Configuration ---
DEFAULT_NODE_COLOR = "#CCCCCC"  # Grey for nodes.
DEFAULT_FLOW_COLOR = "#009EDB"  # UN Blue for flows.
DEFAULT_DATA_PATH = "./data/sankey_data.xlsx"
DEFAULT_POSITIONS_PATH = "./data/node_positions.csv"

# --- Helper Function ---
def is_valid_color(color_val):
    """Checks if a color value is valid (not None, NaN, or empty/whitespace string)."""
    return color_val and pd.notna(color_val) and str(color_val).strip() != ""

# --- Core Sankey Function ---
# (Keep the create_sankey_diagram function exactly as in the previous version)
def create_sankey_diagram(data, positions_df=None):
    """
    Generate a Plotly Sankey diagram using the provided data
    and optional positions dataframe for nodes and flow colors,
    implementing the priority logic for flow colors.
    """
    if data is None or data.empty:
        return go.Figure()

    df = data.copy()

    # Data Cleaning & Preparation
    df.columns = df.columns.str.strip().str.lower()
    if "flow value" in df.columns:
        df = df.rename(columns={"flow value": "value"})

    required_cols = ['source', 'target', 'value']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        st.error(f"❌ Error: Missing required columns in data: {', '.join(missing)}")
        return go.Figure()

    df['source'] = df['source'].astype(str).str.strip()
    df['target'] = df['target'].astype(str).str.strip()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=["value", "source", "target"])
    df['value'] = df['value'].apply(lambda x: 0.001 if x <= 0 else x)

    # Node and Position Processing
    valid_nodes = {str(n) for n in set(df['source']).union(set(df['target']))}
    nodes_order = sorted(list(valid_nodes))
    node_x, node_y = None, None
    node_color_mapping = {}
    node_incoming_colors = {}
    node_outgoing_colors = {}
    arrangement = "snap"

    if positions_df is not None and not positions_df.empty:
        pos_df = positions_df.copy()
        pos_df.columns = pos_df.columns.str.strip().str.lower()

        if "node" in pos_df.columns:
            pos_df["node"] = pos_df["node"].astype(str).str.strip()
            pos_df = pos_df[pos_df["node"].isin(valid_nodes)]

            if not pos_df.empty:
                nodes_order = pos_df["node"].tolist()
                missing_nodes = sorted([n for n in valid_nodes if n not in nodes_order])
                nodes_order.extend(missing_nodes)

                for col in ['x', 'y', 'node_color', 'incoming_flow_color', 'outgoing_flow_color']:
                     if col not in pos_df.columns: pos_df[col] = np.nan if col in ['x','y'] else ""

                pos_indexed = pos_df.set_index("node")
                node_color_mapping = pos_indexed["node_color"].to_dict()
                node_incoming_colors = pos_indexed["incoming_flow_color"].to_dict()
                node_outgoing_colors = pos_indexed["outgoing_flow_color"].to_dict()

                x_list = [pos_indexed.loc[n, 'x'] if n in pos_indexed.index else np.nan for n in nodes_order]
                y_list = [pos_indexed.loc[n, 'y'] if n in pos_indexed.index else np.nan for n in nodes_order]
                x_numeric = pd.to_numeric(pd.Series(x_list), errors='coerce')
                y_numeric = pd.to_numeric(pd.Series(y_list), errors='coerce')

                if x_numeric.notna().any() and y_numeric.notna().any():
                     node_x = x_numeric.where(x_numeric.notna(), None).tolist()
                     node_y = y_numeric.where(y_numeric.notna(), None).tolist()
                     arrangement = "perpendicular"
                else:
                     if 'x' in pos_df.columns or 'y' in pos_df.columns:
                         st.warning("⚠️ Valid numeric 'x'/'y' not found. Using automatic layout.")
                     node_x, node_y = None, None
        else:
             st.warning("⚠️ Positions file lacks 'node' column. Ignoring positions.")


    # Final Node Properties
    label_to_index = {label: i for i, label in enumerate(nodes_order)}
    node_values = {}
    final_node_colors = []

    for node in nodes_order:
        incoming = df.loc[df['target'] == node, 'value'].sum()
        outgoing = df.loc[df['source'] == node, 'value'].sum()
        node_pad_value = max(incoming, outgoing, 0.001)
        node_values[node] = node_pad_value

        custom_color = node_color_mapping.get(node)
        final_node_colors.append(custom_color if is_valid_color(custom_color) else DEFAULT_NODE_COLOR)

    node_labels = [f"{node} ({node_values[node]:,.0f})" for node in nodes_order]

    # Link Processing
    df_filtered = df[df['source'].isin(label_to_index) & df['target'].isin(label_to_index)].copy()
    if df_filtered.empty:
         # Avoid erroring out, just show warning and empty chart
         st.warning("⚠️ No flows connect the nodes defined in the current data/positions.")
         return go.Figure()

    sources = [label_to_index[src] for src in df_filtered['source']]
    targets = [label_to_index[tgt] for tgt in df_filtered['target']]
    values = df_filtered['value'].tolist()

    link_colors = []
    for index, row in df_filtered.iterrows():
        src_node, tgt_node = row['source'], row['target']
        link_color = DEFAULT_FLOW_COLOR
        src_outgoing_color = node_outgoing_colors.get(src_node)
        if is_valid_color(src_outgoing_color):
            link_color = src_outgoing_color
        else:
            tgt_incoming_color = node_incoming_colors.get(tgt_node)
            if is_valid_color(tgt_incoming_color):
                link_color = tgt_incoming_color
        link_colors.append(link_color)

    # Create Plotly Figure
    sankey_node = dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                       label=node_labels, color=final_node_colors, x=node_x, y=node_y,
                       hovertemplate='Node: %{label}<extra></extra>')
    sankey_link = dict(source=sources, target=targets, value=values, color=link_colors,
                       hovertemplate='Flow: %{value:,.0f}<extra></extra>') # Simpler hover

    fig = go.Figure(data=[go.Sankey(arrangement=arrangement, node=sankey_node, link=sankey_link)])
    fig.update_layout(font_size=10, height=800, plot_bgcolor='white', paper_bgcolor='white')
    return fig


# ----- STREAMLIT APP -----

st.set_page_config(page_title="Interactive Sankey Editor", layout="wide")
st.title("📊 Interactive Sankey Diagram with Editable Data")

# --- Initialize Session State & Load Defaults ---
# data_df: Stores the original DataFrame from the last successful file upload (or default).
# positions_df: Stores the positions DataFrame.
# editable_data: Stores the DataFrame currently bound to st.data_editor, used for live chart updates.
if 'data_df' not in st.session_state: st.session_state.data_df = None
if 'positions_df' not in st.session_state: st.session_state.positions_df = None
if 'editable_data' not in st.session_state: st.session_state.editable_data = None

default_loaded_flag = False # Track if defaults were loaded this run

# Load default data if session state is empty AND file exists
if st.session_state.data_df is None:
    if os.path.exists(DEFAULT_DATA_PATH):
        try:
            st.session_state.data_df = pd.read_excel(DEFAULT_DATA_PATH, sheet_name=0)
            # Initialize editable_data when defaults load
            st.session_state.editable_data = st.session_state.data_df.copy()
            default_loaded_flag = True
        except Exception as e:
            st.warning(f"⚠️ Could not load default data file ({DEFAULT_DATA_PATH}): {e}")
    # else: # Don't show this message if defaults were just loaded
    #     if not default_loaded_flag:
    #          st.info(f"ⓘ Default data file not found at {DEFAULT_DATA_PATH}. Upload a file to start.")

# Load default positions if session state is empty AND file exists
if st.session_state.positions_df is None:
     if os.path.exists(DEFAULT_POSITIONS_PATH):
        try:
            st.session_state.positions_df = pd.read_csv(DEFAULT_POSITIONS_PATH)
            default_loaded_flag = True # Set flag if positions also loaded
        except Exception as e:
            st.warning(f"⚠️ Could not load default positions file ({DEFAULT_POSITIONS_PATH}): {e}")


# --- Display Chart Area (Top) ---
chart_placeholder = st.empty()


# --- Data Editor (Middle Section) ---
# Only show the editor if there's data available
if st.session_state.editable_data is not None:
    st.markdown("---") # Separator before editor
    st.subheader("📝 Edit Flow Data (Live Updates Chart)")
    # The data editor displays and directly updates 'editable_data' via its key mechanism
    edited_data_live = st.data_editor(
        st.session_state.editable_data, # Bind editor to this state variable
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor" # Stable key allows state persistence across reruns
    )
    # Update the state variable with the editor's current content on each interaction/rerun
    st.session_state.editable_data = edited_data_live
else:
    # If no editable_data, maybe data failed to load, show message instead of editor
    st.markdown("---")
    st.warning("No data available to edit. Please upload a data file.")


# --- Display Uploaders (Below Editor) ---
st.markdown("---") # Separator before uploaders
st.subheader("📂 Upload New Files")
col1, col2 = st.columns(2)

with col1:
    uploaded_data_file = st.file_uploader("📥 Upload/Replace Excel Data File", type=["xlsx"], key="data_uploader")
    if uploaded_data_file:
        try:
            bytes_data = uploaded_data_file.getvalue()
            new_data = pd.read_excel(io.BytesIO(bytes_data), sheet_name=0)
            # Update base data_df and reset editable_data if file is different
            if st.session_state.data_df is None or not new_data.equals(st.session_state.data_df):
                st.session_state.data_df = new_data
                st.session_state.editable_data = new_data.copy() # Reset editor to new file
                st.success("✅ New data file loaded. Editor reset.")
                st.rerun() # Rerun needed to ensure editor displays the NEW data
            # No need to explicitly clear uploader state, Streamlit handles it
        except Exception as e:
            st.error(f"❌ Error reading Excel file: {e}")

with col2:
    uploaded_positions_file = st.file_uploader("🎨 Upload/Replace Node Positions CSV", type=["csv"], key="positions_uploader")
    if uploaded_positions_file:
        try:
            bytes_data = uploaded_positions_file.getvalue()
            new_positions = pd.read_csv(io.BytesIO(bytes_data))
            # Update positions_df if file is different
            if st.session_state.positions_df is None or not new_positions.equals(st.session_state.positions_df):
                st.session_state.positions_df = new_positions
                st.success("✅ New positions file loaded.")
                st.rerun() # Rerun needed to redraw chart with new positions
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")


# --- Generate and Display Chart (using editable_data) ---
# This now uses the live 'editable_data' state
if st.session_state.editable_data is not None and not st.session_state.editable_data.empty:
     try:
        # Generate the figure using the potentially edited data
        fig = create_sankey_diagram(st.session_state.editable_data, st.session_state.positions_df)
        with chart_placeholder.container():
             st.subheader("📈 Live Sankey Preview")
             st.plotly_chart(fig, use_container_width=True, key="sankey_chart")
     except Exception as e:
          with chart_placeholder.container():
              st.error(f"❌ Error generating Sankey diagram: {e}")
              st.exception(e)
# Handle case where no data could be loaded initially
elif st.session_state.data_df is None and not default_loaded_flag:
     with chart_placeholder.container():
         st.info(f"📊 Upload an Excel data file or place default files ('{os.path.basename(DEFAULT_DATA_PATH)}', '{os.path.basename(DEFAULT_POSITIONS_PATH)}') in ./data/ to begin.")


# # --- Instructions Footer (Last Section) ---
# st.markdown("---")
# st.markdown(f"""
# **Instructions:**
# 1.  Default data may be loaded from `{DEFAULT_DATA_PATH}` and `{DEFAULT_POSITIONS_PATH}` on startup if present in a `./data/` subfolder.
# 2.  Upload new files using the buttons below the table to replace the current data/positions.
# 3.  Edit flow data directly in the table. **The chart updates automatically** after each edit.
# """)
