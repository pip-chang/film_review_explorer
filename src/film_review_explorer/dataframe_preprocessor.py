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


def get_column_types(df: pd.DataFrame) -> dict[str, str]:
    column_types = {}

    df_types = df.applymap(type)

    for column in df.columns:
        unique_types = df_types[column].unique()

        unique_types = [t.__name__ for t in unique_types]

        column_types[column] = ", ".join(set(unique_types))

    return column_types


def clean_cn_noise(text: str) -> str:
    text = text.replace("\n", "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(
        r"([。，！？#＄¥％＆＊＋－／：；（）＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟–—‘’‛“”„‟…‧﹏.])\1+",
        r"\1",
        text,
    )

    return text


def clean_en_noise(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"([!?#$%&'()*+,-./:;<=>@[\]^_`{|}~\'\"])\1+",
        r"\1",
        text,
    )

    return text


def clean_text_basic(
    row: pd.Series,
    cn_websites: list[str],
    clean_cn_noise=clean_cn_noise,
    clean_en_noise=clean_en_noise,
) -> str:
    review = row["review"]
    if row["website"] in cn_websites:
        return clean_cn_noise(review)
    else:
        return clean_en_noise(review)


def calculate_review_length(row: pd.Series, cn_websites: list[str]) -> int:
    if row["website"] in cn_websites:
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


def condition_filter(df: pd.DataFrame, condition: Callable) -> pd.DataFrame:
    """
    def condition(data):
        return (data['A'] > 2) & (data['B'] < 9) & (data['C'] == 'c')
    """
    return df.loc[condition]
