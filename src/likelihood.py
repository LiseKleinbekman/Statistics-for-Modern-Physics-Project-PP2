import sys
from pathlib import Path

# Add parent directory to path so we can import from data folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_load_function import load_data

m_center, counts, uncertainty, bin_width, m_lo, m_hi = load_data()
print(m_center, counts, uncertainty, bin_width, m_lo, m_hi)
