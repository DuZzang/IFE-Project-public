"""The one evaluation window used for every strategy in the team report (decided 2026-09-24).

Holding months 2005-01 to 2019-12 (180 months; formation months 2004-12 to 2019-11).
- Start: the Frazzini-Pedersen beta needs five years of history, so the first usable
  formation month is 2004-12.
- End: the authors' CNN I20 images stop at the 2019-12 formation month. Ending every
  strategy in 2019-12 keeps the three- and four-signal combinations on the same months.
2020 is not used for evaluation; ``stress_test_2020.py`` reports it separately as a
COVID stress test. The CNN comparison with the paper keeps the paper's 2001-2019 window.
"""

from __future__ import annotations

import pandas as pd

START, END = "2005-01", "2019-12"
FIRST_FORMATION, LAST_FORMATION = "2004-12", "2019-11"
MONTHS = 180
PERIOD_TEXT = "formation 2004-12 to 2019-11; returns 2005-01 to 2019-12 (180 months)"


def clip(frame: pd.DataFrame | pd.Series, start: str = START, end: str = END) -> pd.DataFrame | pd.Series:
    """Keep rows whose holding-month index (YYYY-MM) lies inside the window."""
    index = frame.index.astype(str)
    return frame.loc[(index >= start) & (index <= end)]
