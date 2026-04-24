from __future__ import annotations

import pandas as pd
import pytest

from src.data.fetch_treasury import _parse_treasury_xml


XML_SAMPLE = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata">
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2026-01-02T00:00:00</d:NEW_DATE>
        <d:BC_10YEAR>4.56</d:BC_10YEAR>
        <d:BC_2YEAR>4.12</d:BC_2YEAR>
      </m:properties>
    </content>
  </entry>
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2026-01-05T00:00:00</d:NEW_DATE>
        <d:BC_10YEAR>4.60</d:BC_10YEAR>
        <d:BC_2YEAR>4.20</d:BC_2YEAR>
      </m:properties>
    </content>
  </entry>
</feed>
"""


def test_parse_treasury_xml_feed() -> None:
    df = _parse_treasury_xml(XML_SAMPLE, {"us10y": "DGS10", "us2y": "DGS2"})

    assert list(df.columns) == ["us10y", "us2y"]
    assert df.index.name == "date"
    assert df.loc[pd.Timestamp("2026-01-02"), "us10y"] == pytest.approx(4.56)
    assert df.loc[pd.Timestamp("2026-01-05"), "us2y"] == pytest.approx(4.20)
