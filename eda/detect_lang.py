"""Detect the language of every training report. Writes data/lang.csv.

Half of the fold group key (issue #3); manufacturer and model need a DICOM pass
on Kaggle, which this does not do.

Run: .venv/bin/python eda/detect_lang.py
"""
import pandas as pd
from langdetect import DetectorFactory, detect

# langdetect samples randomly; without a fixed seed it returns different answers
# for the same short report between runs.
DetectorFactory.seed = 0


def lang(text):
    try:
        return detect(text)
    except Exception:
        return "unk"


# ponytail: langdetect over hand-written cue rules. Rules keyed on anatomy words
# merge Croatian into Turkish ("menisk" is in both) and Spanish into Portuguese.
def main():
    train = pd.read_csv("data/train.csv")
    train["lang"] = train.Report.map(lang)
    print(train.lang.value_counts().to_string())
    assert train.lang.eq("unk").sum() == 0, "some reports did not classify"
    train[["StudyInstanceUID", "lang"]].to_csv("data/lang.csv", index=False)
    print("\nwrote data/lang.csv")


if __name__ == "__main__":
    main()
