"""
EDA tools. All tools are closures over a pandas DataFrame.
Call make_tools(df) to get a list of bound LangChain tools.
"""

import numpy as np
import pandas as pd
from langchain_core.tools import tool


def make_tools(df: pd.DataFrame):

    @tool
    def get_schema() -> str:
        """Returns column names, dtypes, shape, and a sample of the dataframe."""
        info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape,
            "sample": df.head(5).to_dict(orient="records"),
        }
        return str(info)

    @tool
    def get_missing_values() -> str:
        """Returns missing value counts and percentages per column."""
        missing = df.isnull().sum()
        pct = (missing / len(df) * 100).round(2)
        result = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
        result = result[result["missing_count"] > 0]
        if result.empty:
            return "No missing values detected."
        return result.to_string()

    @tool
    def get_numerical_summary() -> str:
        """Returns descriptive statistics for all numerical columns."""
        num_cols = df.select_dtypes(include="number")
        if num_cols.empty:
            return "No numerical columns found."
        return num_cols.describe().round(3).to_string()

    @tool
    def get_categorical_summary() -> str:
        """Returns value counts and cardinality for categorical columns."""
        cat_cols = df.select_dtypes(include=["object", "category", "bool"])
        if cat_cols.empty:
            return "No categorical columns found."
        summary = {}
        for col in cat_cols.columns:
            summary[col] = {
                "cardinality": df[col].nunique(),
                "top_5_values": df[col].value_counts().head(5).to_dict(),
                "null_count": int(df[col].isnull().sum()),
            }
        return str(summary)

    @tool
    def get_correlation_matrix() -> str:
        """Returns the Pearson correlation matrix for numerical columns."""
        num_cols = df.select_dtypes(include="number")
        if num_cols.shape[1] < 2:
            return "Not enough numerical columns to compute correlations."
        return num_cols.corr().round(3).to_string()

    @tool
    def get_outlier_summary() -> str:
        """Detects outliers in numerical columns using the IQR method."""
        num_cols = df.select_dtypes(include="number")
        if num_cols.empty:
            return "No numerical columns found."
        summary = {}
        for col in num_cols.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[
                (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
            ]
            summary[col] = {
                "outlier_count": len(outliers),
                "outlier_pct": round(len(outliers) / len(df) * 100, 2),
                "lower_bound": round(q1 - 1.5 * iqr, 3),
                "upper_bound": round(q3 + 1.5 * iqr, 3),
            }
        return str(summary)

    @tool
    def get_temporal_columns() -> str:
        """Identifies datetime columns and returns their range and frequency hints."""
        df_copy = df.copy()
        dt_cols = list(df_copy.select_dtypes(include=["datetime", "datetimetz"]).columns)
        for col in df_copy.select_dtypes(include="object").columns:
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], infer_datetime_format=True)
                dt_cols.append(col)
            except Exception:
                pass
        if not dt_cols:
            return "No datetime columns detected."
        summary = {}
        for col in dt_cols:
            summary[col] = {
                "min": str(df_copy[col].min()),
                "max": str(df_copy[col].max()),
                "range_days": (df_copy[col].max() - df_copy[col].min()).days,
                "null_count": int(df_copy[col].isnull().sum()),
            }
        return str(summary)

    return [
        get_schema,
        get_missing_values,
        get_numerical_summary,
        get_categorical_summary,
        get_correlation_matrix,
        get_outlier_summary,
        get_temporal_columns,
    ]