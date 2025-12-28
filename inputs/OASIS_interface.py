"""OASIS Data Interface Module

This module provides helper functions to fetch and parse electricity market price data
from CAISO's OASIS (Open Access Same-Time Information System) API.

The module handles:
- CSV file parsing from OASIS ZIP responses
- Error detection and reporting from OASIS responses (in XML)
- Day-ahead nodal LMP (Locational Marginal Price) data retrieval
- Schema normalization for different CAISO API versions
- Pagination of long date ranges to respect API limits
- Rate limit management through configurable sleep intervals

Typical usage:
    >>> from OASIS_interface import get_historical_caiso_lmp
    >>> data = get_historical_caiso_lmp(
    ...     node="TH_N001",
    ...     start_date="20251206T00:00-0000",
    ...     end_date="20251216T00:00-0000"
    ... )
    >>> print(data.head())
"""

import pandas as pd
import numpy as np
from datetime import datetime
import requests
from io import BytesIO
from zipfile import ZipFile
import time
import os
import xml.etree.ElementTree as ET



def is_xml_file(raw_bytes):
    """Detect whether a file is XML.
    
    Parameters
    ----------
    raw_bytes : bytes
        Raw file content as bytes.
    
    Returns
    -------
    bool
        True if the file appears to be XML, False otherwise.
    """

    start = raw_bytes[:20].lstrip()
    return start.startswith(b'<?xml') or start.startswith(b'<')


def _strip_ns(tag):
    """Remove XML namespace from tag.
    
    Internal helper function to strip namespace prefixes from XML element tags.
    
    Parameters
    ----------
    tag : str
        XML tag name, potentially with namespace (e.g., "ns:ELEMENT").
    
    Returns
    -------
    str
        Tag name without namespace prefix.
    
    Examples
    --------
    >>> _strip_ns("{http://example.com}ELEMENT")
    'ELEMENT'
    >>> _strip_ns("ELEMENT")
    'ELEMENT'
    """
    return tag.split("}", 1)[-1] if "}" in tag else tag

def xml_contains_error(raw_bytes):
    """Check if XML response contains an error block.
    
    OASIS error responses may use different namespace prefixes (e.g., <m:ERROR>),
    so this function checks for the generic "ERROR>" pattern.
    
    Parameters
    ----------
    raw_bytes : bytes
        Raw XML content as bytes.
    
    Returns
    -------
    bool
        True if an error block is detected, False otherwise.
    """
    return b"ERROR>" in raw_bytes or b"ERR_CODE>" in raw_bytes

def parse_oasis_error_xml(raw_bytes):
    """Extract error code and description from OASIS error XML.
    
    Parameters
    ----------
    raw_bytes : bytes
        Raw XML content containing error information.
    
    Returns
    -------
    tuple[str or None, str or None]
        Tuple of (error_code, error_description). Returns (None, None) if
        error elements are not found.
    """
    root = ET.fromstring(raw_bytes)
    err_code = None
    err_desc = None

    for elem in root.iter():
        tag = _strip_ns(elem.tag).upper()
        if tag == "ERR_CODE":
            err_code = elem.text
        elif tag == "ERR_DESC":
            err_desc = elem.text

    return err_code, err_desc


def parse_oasis_xml(raw_bytes):
    """Parse OASIS XML that contains REPORT_DATA rows."""
    root = ET.fromstring(raw_bytes)

    # Find all REPORT_DATA elements regardless of namespace
    report_rows = [
        elem for elem in root.iter()
        if _strip_ns(elem.tag).upper() == "REPORT_DATA"
    ]

    if not report_rows:
        raise ValueError("XML contains no REPORT_DATA elements.")

    rows = []
    for rpt in report_rows:
        row = {}
        for child in rpt:
            key = _strip_ns(child.tag).upper()
            row[key] = (child.text or "").strip()
        rows.append(row)

    return pd.DataFrame(rows)

def read_oasis_zip(response_bytes):
    """Extract and parse all CSV and XML files from OASIS ZIP response.
    
    OASIS returns compressed ZIP files containing data in either CSV or XML format.
    This function automatically detects the file type and parses accordingly,
    concatenating all valid data into a single DataFrame.
    
    Parameters
    ----------
    response_bytes : bytes
        Raw bytes from OASIS API response (ZIP file content).
    
    Returns
    -------
    pandas.DataFrame
        Concatenated data from all valid CSV and XML files in the archive.
        Returns empty DataFrame if no usable data is found.
    
    Notes
    -----
    - Error XML files are skipped with a warning message
    - Parse errors are logged but do not halt processing
    - Files other than CSV or XML are ignored
    """
    dfs = []

    with ZipFile(BytesIO(response_bytes)) as z:
        for fname in z.namelist():
            if not fname.lower().endswith((".csv", ".xml")):
                continue

            with z.open(fname) as f:
                raw = f.read()

            # XML case
            if is_xml_file(raw):
                if xml_contains_error(raw):
                    # Log the error
                    code, desc = parse_oasis_error_xml(raw)
                    print(f"[OASIS ERROR] File: {fname} | Code: {code} | Desc: {desc}")
                    continue  # Skip this file

                # Valid XML with data
                try:
                    df = parse_oasis_xml(raw)
                except Exception as e:
                    print(f"[XML PARSE ERROR] {fname}: {e}")
                    continue

            # CSV case
            else:
                try:
                    df = pd.read_csv(BytesIO(raw))
                except Exception as e:
                    print(f"[CSV PARSE ERROR] {fname}: {e}")
                    continue

            dfs.append(df)

    if not dfs:
        print("[WARN] No usable data found in ZIP.")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def fetch_caiso_week(node, start_date, end_date):
    """
    Fetch CAISO day-ahead nodal LMPs from OASIS for a single week.
    
    Retrieves price data from the CAISO OASIS PRC_LMP query and normalizes
    the response schema to a consistent format regardless of API version.
    
    Parameters
    ----------
    node : str
        CAISO PNode name (e.g., "TH_N001", "119TH_BP_LNODEXF1").
    start_date : str
        Start datetime in ISO format: YYYYMMDDT00:00-0000
    end_date : str
        End datetime in ISO format: YYYYMMDDT00:00-0000

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe with columns:
        - INTERVAL_START : datetime - When the price interval begins
        - NODE : str - The requested node identifier
        - LMP : float - Locational Marginal Price
        - MCE : float - Marginal Cost of Energy
        - MCC : float - Marginal Cost of Congestion
        - MCL : float - Marginal Cost of Loss
        
    Raises
    ------
    RuntimeError
        If the HTTP request fails (non-200 status code).
    ValueError
        If required columns are missing or schema cannot be recognized.
        
    Notes
    -----
    - Only tested for weekly or shorter timeframes
    - May fail or return incomplete data for timeframes longer than 1 week
    - Use get_historical_caiso_lmp for multi-week date ranges
    - Handles two CAISO schema versions: LMP/MCE/MCC/MCL columns or MW value column
    
    Examples
    --------
    >>> data = fetch_caiso_week(
    ...     node="TH_N001",
    ...     start_date="20251206T00:00-0000",
    ...     end_date="20251207T00:00-0000"
    ... )
    >>> print(data.columns.tolist())
    ['INTERVAL_START', 'NODE', 'LMP', 'MCE', 'MCC', 'MCL']
    """

    base_url = "http://oasis.caiso.com/oasisapi/SingleZip"

    params = {
        "queryname": "PRC_LMP",
        "market_run_id": "DAM",
        "node": node,
        "startdatetime": start_date,
        "enddatetime": end_date,
        "resultformat": 6,  # Produces CSV instead of XML
        "version": 12  # Confirmed to work with v12; not sure about others
    }

    print(f"Requesting data for node {node}...")
    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        raise RuntimeError(f"Failed request: {response.status_code}")

    # Unzip the returned file
    df = read_oasis_zip(response.content)

    if len(df) == 0:
        return pd.DataFrame()

    # Normalize column names
    df.columns = df.columns.str.upper()

    # Detect schema
    schema_a = "LMP" in df.columns  # older schema
    schema_b = "MW" in df.columns   # newer schema

    if schema_a:
        # Schema A: LMP, MCE, MCC, MCL already present
        df["VALUE"] = df["LMP"]

    elif schema_b:
        # Schema B: MW column contains the numeric value
        df["VALUE"] = df["MW"]

    else:
        raise ValueError("Unrecognized CAISO PRC_LMP schema. No LMP or MW column found.")

    # Standardize timestamp column names
    time_cols = [c for c in df.columns if "INTERVALSTART" in c]
    if time_cols:
        df["INTERVAL_START"] = pd.to_datetime(df[time_cols[0]])
    else:
        raise ValueError("Could not find interval start time column.")

    # Standardize node column
    if "NODE" not in df.columns:
        raise ValueError("NODE column missing from PRC_LMP file.")

    # Pivot into wide format
    df_pivot = df.pivot_table(
        index=["INTERVAL_START", "NODE"],
        columns="LMP_TYPE",
        values="VALUE"
    ).reset_index()

    # Ensure all expected columns exist
    for col in ["LMP", "MCE", "MCC", "MCL"]:
        if col not in df_pivot.columns:
            df_pivot[col] = None

    return df_pivot

def get_historical_caiso_lmp(node, start_date, end_date, sleep_time=5, periodic_save=False, fpath=""):
    """
    Fetch CAISO historical day-ahead nodal LMPs for arbitrary date ranges.
    
    This is the primary function for retrieving CAISO price data. It handles
    long date ranges by automatically breaking them into weekly chunks to
    respect API limitations and rate limits.
    
    Parameters
    ----------
    node : str
        CAISO PNode name (e.g., "TH_N001", "119TH_BP_LNODEXF1").
    start_date : str
        Start datetime in ISO format: YYYYMMDDT00:00-0000
    end_date : str
        End datetime in ISO format: YYYYMMDDT00:00-0000
    sleep_time : int, optional
        Time to sleep (in seconds) between consecutive API calls.
        Used to respect OASIS rate limits. Default is 5 seconds.
        You will get error code 429 if you request too fast (i.e. this value is too low). 
    periodic_save : bool, optional
        Whether to save intermediate data during long requests. Default is False.
        Useful for very large date ranges to prevent data loss from interruptions.
    fpath : str, optional
        Directory path for saving intermediate data files.
        Used only if periodic_save=True. Default is empty string.

    Returns
    -------
    pandas.DataFrame
        Complete historical data with columns:
        - INTERVAL_START : datetime - When the price interval begins
        - NODE : str - The requested node identifier
        - LMP : float - Locational Marginal Price
        - MCE : float - Marginal Cost of Energy
        - MCC : float - Marginal Cost of Congestion
        - MCL : float - Marginal Cost of Loss
        
    Raises
    ------
    RuntimeError
        If any individual API request fails.
    ValueError
        If data cannot be parsed from responses.
        
    Notes
    -----
    - For date ranges longer than 1 week, this function automatically
      chunks the request into weekly segments
    - Intermediate results are concatenated with ignore_index=True
    - When periodic_save=True, intermediate files are saved as:
      intermediate_{node}_LMP_data.csv
    
    Examples
    --------
    >>> # Fetch a single week
    >>> week_data = get_historical_caiso_lmp(
    ...     node="TH_N001",
    ...     start_date="20251206T00:00-0000",
    ...     end_date="20251213T00:00-0000"
    ... )
    
    >>> # Fetch a month with periodic saves
    >>> month_data = get_historical_caiso_lmp(
    ...     node="TH_N001",
    ...     start_date="20251201T00:00-0000",
    ...     end_date="20251231T00:00-0000",
    ...     sleep_time=2,
    ...     periodic_save=True,
    ...     fpath="./data"
    ... )
    """

    # Break the date range into weekly chunks. 
    # This is necessary because the CAISO API has a limit on the number of days that can be requested at once.

    if (pd.Timestamp(end_date) - pd.Timestamp(start_date)) > pd.Timedelta(weeks=1):

        weekly_data = []

        # Construct range of dates
        date_range = pd.date_range(start=start_date, end=end_date, freq="W")

        if date_range[0] != pd.Timestamp(start_date):
            weekly_data.append(fetch_caiso_week(node, start_date, date_range[0].strftime("%Y%m%dT00:00-0000")))

        for i in range(len(date_range) - 1):
            time.sleep(sleep_time)  # Sleep for a second to avoid hitting the API rate limit

            start_week = date_range[i].strftime("%Y%m%dT00:00-0000")
            end_week = date_range[i + 1].strftime("%Y%m%dT00:00-0000")

            weekly_data.append(fetch_caiso_week(node, start_week, end_week))

            if periodic_save: 
                # Save the data periodically to avoid losing it in case of a crash
                intermediate_df = pd.concat(weekly_data, ignore_index=True)
                intermediate_df.to_csv(os.path.join(fpath, f"intermediate_{node}_LMP_data.csv"), index=False)
        
        if date_range[-1] != pd.Timestamp(end_date):
            time.sleep(sleep_time)  # Sleep for a second to avoid hitting the API rate limit
            weekly_data.append(fetch_caiso_week(node, date_range[-1].strftime("%Y%m%dT00:00-0000"), end_date))

        # Concatenate all the weekly data into a single dataframe
        df_historical = pd.concat(weekly_data, ignore_index=True)

    else: 
        df_historical = fetch_caiso_week(node, start_date, end_date)

    return df_historical


