import logging
import re
import math
from typing import Any, Callable, Optional, Union
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def read_jsonl_to_dataframe(*paths: Union[Path, str]) -> pd.DataFrame:
    full_df = pd.DataFrame()

    for path in paths:
        path = Path(path)
        if not path.exists():
            logging.error(f'"{path}" does not exist.')
            continue

        files = []
        if path.is_file():
            if path.suffix == ".jsonl":
                files = [path]
            else:
                logging.error(f'"{path}" is not a JSONL file.')
                continue
        elif path.is_dir():
            files = list(path.glob("*.jsonl"))
            if files == []:
                logging.error(f'"{path}" does not contain any JSONL files.')
                continue
        else:
            logging.error(f'"{path}" is neither a file nor a directory.')
            continue

        for file in files:
            df = pd.read_json(file, lines=True)
            full_df = pd.concat([full_df, df], ignore_index=True)

    return full_df


def clean_text(row: pd.Series) -> Optional[str]:
    # clean chinese text: spaces, '\n'
    # clean eng text: \n

    review = row["review"]
    pass


def calculate_review_length(
    row: pd.Series, chinese_websites: list[str]
) -> Optional[int]:
    if row["website"] in chinese_websites:
        return len(row["review"])
    else:
        return len(re.findall(r"\b\w+\b", row["review"]))


def calculate_rating_level(row: pd.Series) -> Optional[str]:
    rating_ratio = row["rating_ratio"]
    if (math.isnan(rating_ratio)) or (rating_ratio is None):
        return None
    elif rating_ratio >= 0.8:
        return "Good (>=8/10)"
    elif rating_ratio <= 0.4:
        return "Bad (<=4/10)"
    else:
        return "Ok (4~8/10)"


def calculate_like_level(row: pd.Series) -> Optional[str]:
    like_ratio = row["like_ratio"]
    if (math.isnan(like_ratio)) or (like_ratio == None):
        return None
    elif like_ratio >= 0.8:
        return "Mostly Agree (>80%)"
    elif 0.5 < like_ratio < 0.8:
        return "Somewhat Agree (50%~80%)"
    elif 0.2 < like_ratio <= 0.5:
        return "Somewhat Disgree (20%~50%)"
    else:
        return "Mostly Disagree (<20%)"


def update_column(
    df: pd.DataFrame, apply_function: Callable, column_name: str, *args, **kwargs
) -> pd.DataFrame:
    df[column_name] = df.apply(lambda row: apply_function(row, *args, **kwargs), axis=1)
    return df


def get_sub_dataframe(df: pd.DataFrame, condition: Callable) -> pd.DataFrame:
    """
    def condition(data):
        return (data['A'] > 2) & (data['B'] < 9) & (data['C'] == 'c')
    """
    return df.loc[condition]


