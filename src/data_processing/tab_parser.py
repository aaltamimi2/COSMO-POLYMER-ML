"""
Parser for COSMO-therm tab files (both REF and SLE formats)
"""
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


class COSMOThermTabParser:
    """Parse COSMO-therm tab files and extract solubility data"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.jobs = []

    def parse(self) -> pd.DataFrame:
        """Parse the tab file and return a DataFrame with all data"""
        with open(self.filepath, 'r') as f:
            content = f.read()

        # Split into individual jobs
        job_sections = re.split(r'\n(?=\s*Property\s+job\s+\d+)', content)

        all_data = []
        for section in job_sections:
            if not section.strip():
                continue
            job_data = self._parse_job_section(section)
            if job_data:
                all_data.extend(job_data)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        return df

    def _parse_job_section(self, section: str) -> List[Dict]:
        """Parse a single job section"""
        lines = section.strip().split('\n')

        # Extract metadata from header lines
        metadata = {}

        # Extract temperature from Settings line
        for line in lines:
            if 'Settings' in line and 'T=' in line:
                temp_match = re.search(r'T=\s*([\d.]+)\s*K', line)
                if temp_match:
                    metadata['Temperature_K'] = float(temp_match.group(1))

        # Detect file type (REF or SLE) from General line
        file_type = 'UNKNOWN'
        for line in lines:
            if 'General' in line:
                if 'w_solub' in line:
                    file_type = 'REF'
                elif 'w_SLE' in line:
                    file_type = 'SLE'

                # Extract H_fus if available (REF only)
                h_fus_match = re.search(r'H_fus\(solute\)\s*=\s*([\d.]+)', line)
                if h_fus_match:
                    metadata['H_fus'] = float(h_fus_match.group(1))

                # Extract DG_fus
                dg_fus_match = re.search(r'DG_fus\(solute\)\s*=\s*([\d.]+)', line)
                if dg_fus_match:
                    metadata['DG_fus'] = float(dg_fus_match.group(1))

        metadata['Type'] = file_type

        # Find the data table
        data_start = -1
        header_line = -1

        for i, line in enumerate(lines):
            if 'Nr Solvent' in line:
                header_line = i
                data_start = i + 1
                break

        if data_start == -1:
            return []

        # Parse header
        header = lines[header_line].strip().split()
        # Remove 'Nr' from header
        if header[0] == 'Nr':
            header = header[1:]

        # Parse data rows
        job_data = []
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith('Property') or line.startswith('Settings'):
                break

            row_data = self._parse_data_row(line, header, metadata, file_type)
            if row_data:
                job_data.append(row_data)

        return job_data

    def _parse_data_row(self, line: str, header: List[str], metadata: Dict, file_type: str) -> Dict:
        """Parse a single data row"""
        # Split on whitespace, but handle 'NA' values and multi-word solvent names
        parts = line.split()

        if len(parts) < 2:
            return None

        # First element is row number
        row_num = parts[0]

        # Find where numeric data starts by looking for the first float
        # Solvent name can be multi-word, so we need to be careful
        numeric_start = -1
        solvent_parts = []

        for i in range(1, len(parts)):
            # Try to parse as float
            try:
                float(parts[i])
                numeric_start = i
                break
            except ValueError:
                solvent_parts.append(parts[i])

        if numeric_start == -1:
            return None

        solvent_name = ' '.join(solvent_parts)
        numeric_values = parts[numeric_start:]

        # Build data dictionary
        data = {
            'Solvent': solvent_name,
            **metadata
        }

        # Map numeric values to headers (excluding 'Solvent' from header)
        header_without_solvent = [h for h in header if h != 'Solvent']

        for i, value in enumerate(numeric_values):
            if i < len(header_without_solvent):
                col_name = header_without_solvent[i]
                try:
                    if value.upper() == 'NA':
                        data[col_name] = None
                    else:
                        data[col_name] = float(value)
                except ValueError:
                    data[col_name] = value

        return data


def parse_tab_file(filepath: str) -> pd.DataFrame:
    """Convenience function to parse a tab file"""
    parser = COSMOThermTabParser(filepath)
    return parser.parse()


def extract_common_columns(df: pd.DataFrame, method_type: str) -> pd.DataFrame:
    """
    Extract and rename columns to common format

    Parameters:
    -----------
    df : pd.DataFrame
        Parsed dataframe from tab file
    method_type : str
        'REF' or 'SLE'

    Returns:
    --------
    pd.DataFrame with standardized column names
    """

    common_cols = {
        'Solvent': 'Solvent',
        'Temperature_K': 'Temperature_K',
        'mu(self)': 'mu_self',
        'mu(solv)': 'mu_solv',
        'Solvent_density': 'Solvent_density',
        'Solvent_MolWeight': 'Solvent_MolWeight',
        'Type': 'Method'
    }

    if method_type == 'REF':
        specific_cols = {
            'log10(x_solub)': 'log10_x',
            'w_solub': 'w',
            'log10(S)': 'log10_S'
        }
    elif method_type == 'SLE':
        specific_cols = {
            'log10(x_SLE)': 'log10_x',
            'w_SLE': 'w',
            'log10(S_SLE)': 'log10_S',
            'x(SLE)': 'x'
        }
    else:
        raise ValueError(f"Unknown method type: {method_type}")

    # Combine column mappings
    col_mapping = {**common_cols, **specific_cols}

    # Select and rename columns that exist
    existing_cols = {old: new for old, new in col_mapping.items() if old in df.columns}

    df_out = df[list(existing_cols.keys())].copy()
    df_out = df_out.rename(columns=existing_cols)

    # Convert numeric columns to proper types
    numeric_cols = ['log10_x', 'log10_S', 'w', 'mu_self', 'mu_solv',
                    'Solvent_density', 'Solvent_MolWeight', 'Temperature_K']

    for col in numeric_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')

    # Drop rows with missing critical values
    critical_cols = ['log10_x', 'Solvent', 'Temperature_K']
    available_critical = [c for c in critical_cols if c in df_out.columns]
    df_out = df_out.dropna(subset=available_critical)

    # Add chemical potential difference
    if 'mu_self' in df_out.columns and 'mu_solv' in df_out.columns:
        df_out['delta_mu'] = df_out['mu_solv'] - df_out['mu_self']

    return df_out
