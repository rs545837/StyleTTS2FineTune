import argparse
from phonemizer import phonemize
import os
from tqdm import tqdm
import re

# argument parser
parser = argparse.ArgumentParser(description="Phonemize transcriptions.")
parser.add_argument(
    "--language",
    type=str,
    default="en-us",
    help="The language to use for phonemization.",
)

args = parser.parse_args()

with open("./trainingdata/output.txt", "r") as f:  # Path to output.txt
    lines = f.readlines()

# Phonemize the transcriptions
phonemized = []
filenames = []
transcriptions = []
speakers = []
phonemized_lines = []

for line in lines:  # Split filenames, text and speaker without phonemizing. Prevents mem error
    filename, transcription, speaker = line.strip().split("|")
    filenames.append(filename)
    transcriptions.append(transcription)
    speakers.append(speaker)

# Phonemize all text in one go to avoid triggering mem error
phonemized = phonemize(
    transcriptions,
    language=args.language,
    backend="espeak",
    preserve_punctuation=True,
    with_stress=True,
)

for i in tqdm(range(len(filenames))):
    phonemized_lines.append(
        (filenames[i], f"{filenames[i]}|{phonemized[i]}|{speakers[i]}\n")
    )

def extract_number(filename):
    # Try to extract a number from the filename
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        # If no number is found, return the filename as a string
        # This will sort non-numeric filenames alphabetically after numeric ones
        return filename

# Sort the lines using the new extract_number function
phonemized_lines.sort(key=lambda x: extract_number(x[0]))

# Split training/validation set
train_lines = phonemized_lines[: int(len(phonemized_lines) * 0.9)]
val_lines = phonemized_lines[int(len(phonemized_lines) * 0.9) :]

with open("./trainingdata/train_list.txt", "w+", encoding="utf-8") as f:  # Path for train_list.txt in the training data folder
    for _, line in train_lines:
        f.write(line)

with open("./trainingdata/val_list.txt", "w+", encoding="utf-8") as f:  # Path for val_list.txt in the training data folder
    for _, line in val_lines:
        f.write(line)
