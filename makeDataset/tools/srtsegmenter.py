import pysrt
from pydub import AudioSegment
import os
from phonemizer import phonemize
import glob
from tqdm import tqdm
from wvmos import get_wvmos
import torch
import numpy as np

# Initialize WVMOS model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wvmos_model = get_wvmos(cuda=torch.cuda.is_available())

output_dir = './segmentedAudio/'
bad_audio_dir = './badAudio/'
srt_dir = './srt/'
audio_dir = './audio/'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(bad_audio_dir, exist_ok=True)
os.makedirs(srt_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)
os.makedirs('./trainingdata', exist_ok=True)

srt_list = glob.glob("./srt/*.srt")
audio_list = glob.glob("./audio/*.wav")

if len(srt_list) == 0 or len(audio_list) == 0:
    raise Exception(f"You need to have at least 1 srt file and 1 audio file, you have {len(srt_list)} srt and {len(audio_list)} audio files!")

print(f"SRT Files: {len(srt_list)}")

buffer_time = 200
max_allowed_gap = 1.5 * buffer_time

def calculate_wvmos_score(audio_path):
    return wvmos_model.calculate_one(audio_path)

for sub_file in tqdm(srt_list):
    subs = pysrt.open(sub_file)
    audio_name = os.path.basename(sub_file).replace(".srt", ".wav")

    audio = AudioSegment.from_wav(f'./{audio_dir}/{audio_name}')
    
    with open('./trainingdata/output.txt', 'a+') as out_file:
        for i, sub in enumerate(subs):
            start_time = (sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
            end_time = (sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.start.milliseconds

            if i < len(subs) - 1:
                next_sub = subs[i + 1]
                next_start_time = (next_sub.start.minutes * 60 + next_sub.start.seconds) * 1000 + next_sub.start.milliseconds
                gap_to_next = next_start_time - end_time

                if gap_to_next > max_allowed_gap:
                    end_time += buffer_time
                    print(f"Added buffer time to segment {i}: New end_time = {end_time}")
                else:
                    adjustment = min(buffer_time, gap_to_next // 2)
                    end_time += adjustment
                    print(f"Adjusted end_time for segment {i} by {adjustment}ms due to small gap: New end_time = {end_time}")
            else:
                end_time += buffer_time
                print(f"Added buffer time to the last segment {i}: New end_time = {end_time}")

            audio_segment = audio[start_time:end_time]
            duration = len(audio_segment)
            filename = f'{audio_name[:-4]}_{i}.wav'

            temp_path = os.path.join(output_dir, "temp_" + filename)
            audio_segment.export(temp_path, format='wav')

            # Calculate WVMOS score
            wvmos_score = calculate_wvmos_score(temp_path)

            if 1850 <= duration <= 12000 and wvmos_score >= 2.5:  # Adjust these values as needed
                os.rename(temp_path, os.path.join(output_dir, filename))
                out_file.write(f'{filename}|{sub.text}|1\n')
            else:
                os.rename(temp_path, os.path.join(bad_audio_dir, filename))

            # Remove temporary file if it still exists
            if os.path.exists(temp_path):
                os.remove(temp_path)

print("Processing complete.")
