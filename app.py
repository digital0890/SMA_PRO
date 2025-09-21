# -------------------------------
# Styling & Theme (Professional Dark, Unified Layout)
# -------------------------------
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');"
DARK_BG = "#0d1321"
CARD_BG = "#151b2d"
ACCENT = "#2dd4bf"
ACCENT_SECOND = "#4f46e5"
TEXT = "#f3f4f6"
MUTED = "#a1a1aa"
ERROR = "#ef4444"

CUSTOM_CSS = f"""
{FONT_IMPORT}
:root {{
  --bg: {DARK_BG};
  --card: {CARD_BG};
  --accent: {ACCENT};
  --accent-2: {ACCENT_SECOND};
  --text: {TEXT};
  --muted: {MUTED};
  --error: {ERROR};
  --border: rgba(255,255,255,0.1);
  --shadow: rgba(0,0,0,0.35);
}}

[data-testid='stAppViewContainer'] {{
  background: linear-gradient(180deg, #0d1321 0%, #1f2937 100%);
  color: var(--text);
  font-family: 'Inter', 'Poppins', system-ui, sans-serif;
  padding: 2rem 2rem 4rem 2rem;
  min-height: 100vh;
  overflow-x: hidden;
}}

[data-testid='stSidebar'] {{
  background: linear-gradient(180deg, #151b2d 0%, #1e293b 100%);
  border-right: 1px solid var(--border);
  padding: 2rem 1.75rem;
  box-shadow: 0 4px 20px var(--shadow);
  width: 320px;
  border-radius: 12px;
}}

.streamlit-card {{
  background: var(--card);
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 8px 28px var(--shadow);
  border: 1px solid var(--border);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  margin-bottom: 2rem;
}}

.streamlit-card:hover {{
  transform: translateY(-5px);
  box-shadow: 0 12px 36px var(--shadow);
}}

.stSelectbox, .stSlider, .stDateInput, .stTimeInput {{
  background: rgba(255,255,255,0.08);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.8rem;
  transition: all 0.3s ease;
  margin-bottom: 1rem;
}}

.stSelectbox:hover, .stSlider:hover, .stDateInput:hover, .stTimeInput:hover {{
  background: rgba(255,255,255,0.12);
  border-color: var(--accent-2);
}}

.stSelectbox > div > div > select,
.stDateInput > div > div > input,
.stTimeInput > div > div > input {{
  color: var(--text);
  background: transparent;
  border: none;
  outline: none;
  font-size: 0.95rem;
}}

.stSlider > div > div > div > div {{
  background: var(--accent);
  border-radius: 12px;
}}

.plotly-graph-div {{
  background: transparent !important;
  border-radius: 16px;
  border: 1px solid var(--border);
  overflow: hidden;
  box-shadow: 0 6px 18px var(--shadow);
}}

h3 {{
  font-family: 'Poppins', sans-serif;
  font-weight: 600;
  font-size: 1.7rem;
  color: var(--text);
  margin-bottom: 1.5rem;
  letter-spacing: -0.02em;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.5rem;
}}

button, .stButton > button {{
  background: var(--accent-2);
  color: var(--text);
  border: none;
  border-radius: 12px;
  padding: 0.7rem 1.6rem;
  font-family: 'Inter', sans-serif;
  font-weight: 500;
  font-size: 0.95rem;
  transition: all 0.3s ease;
  margin-top: 1rem;
}}

button:hover, .stButton > button:hover {{
  background: #4338ca;
  transform: translateY(-2px);
  box-shadow: 0 4px 14px var(--shadow);
}}

.chart-container {{
  margin-top: 2rem;
  animation: slideIn 0.5s ease-in-out;
}}

@keyframes slideIn {{
  from {{ opacity: 0; transform: translateY(15px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

.stSpinner > div {{
  color: var(--accent);
  font-family: 'Inter', sans-serif;
}}

@media (max-width: 1024px) {{
  [data-testid='stAppViewContainer'] {{
    padding: 1.5rem;
  }}
  [data-testid='stSidebar'] {{
    width: 100%;
    padding: 1.5rem;
    border-radius: 8px;
  }}
  .streamlit-card {{
    padding: 1.75rem;
  }}
  h3 {{
    font-size: 1.5rem;
  }}
}}

@media (max-width: 768px) {{
  .streamlit-card {{
    padding: 1.5rem;
  }}
  button, .stButton > button {{
    padding: 0.6rem 1.2rem;
    font-size: 0.9rem;
  }}
}}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)
