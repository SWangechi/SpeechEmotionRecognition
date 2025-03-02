import os
import shutil
from fastapi import UploadFile

UPLOAD_FOLDER = "temp_audio"

def save_audio_temp(audio: UploadFile):
    """
    Saves uploaded audio file temporarily.
    """
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, audio.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    return file_path
