import sys
import os

# OpenVoice 디렉토리 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'OpenVoice'))

import subprocess
import torch
from OpenVoice.openvoice import se_extractor
from OpenVoice.openvoice.api import BaseSpeakerTTS, ToneColorConverter

def extract_audio_from_video(video_path, audio_path):
    command = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'libmp3lame', '-ab', '192k', audio_path]
    subprocess.run(command, check=True)
    print(f"Audio extracted and saved to {audio_path}")

def generate_speech_with_tts(text, reference_speaker, output_audio_path, device="cpu"):
    # 모델과 체크포인트 설정
    ckpt_base = 'checkpoints/base_speakers/EN'
    ckpt_converter = 'checkpoints/converter'
    output_dir = os.path.dirname(output_audio_path)

    # 모델 로딩
    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    os.makedirs(output_dir, exist_ok=True)

    # 스타일 추출
    source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)

    # 텍스트 음성 생성
    src_path = f'{output_dir}/tmp.mp3'
    base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=1.0)

    # 톤 색상 변환
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=output_audio_path,
        message=encode_message
    )

    print(f"Generated speech saved to {output_audio_path}")

def lip_sync_with_wav2lip(video_path, audio_path, output_video_path, checkpoint_path):
    # Wav2Lip의 정확한 경로를 지정합니다.
    command = [
        'python', 'Wav2Lip/inference.py',  # 정확한 경로
        '--checkpoint_path', checkpoint_path,
        '--face', video_path,
        '--audio', audio_path
    ]
    subprocess.run(command, check=True)
    print(f"Lip sync completed and saved to {output_video_path}")

def main():
    video_path = 'sample.mp4'  # 입력 비디오 경로
    extracted_audio_path = 'extracted_audio.mp3'  # 비디오에서 추출된 오디오 저장 경로
    reference_speaker = extracted_audio_path  # 비디오에서 추출한 오디오를 reference_speaker로 사용
    tts_audio_path = 'outputs/output_en_default.mp3'  # OpenVoice로 생성된 변형 음성 저장 경로
    output_video_path = 'outputs/output_lip_synced.mp4'  # 입술 동기화된 비디오 저장 경로
    wav2lip_checkpoint_path = 'checkpoints/wav2lip.pth'  # Wav2Lip 체크포인트 경로

    # 1. 비디오에서 음성 추출
    extract_audio_from_video(video_path, extracted_audio_path)

    # 2. OpenVoice TTS로 음성 변형
    text = """Hi there! I hope you're doing well today. I just wanted to say how great it is to see your dedication and hard work. Keep pushing forward, and always remember that every step you take is progress. No matter the challenges, you’re doing amazing. Take it one day at a time and trust the process. Keep going, you're on the right path! Let's continue to move forward and keep striving for success. Together, we can make incredible things happen!"""

    generate_speech_with_tts(text, reference_speaker, tts_audio_path)

    # 3. Wav2Lip으로 입술 동기화
    lip_sync_with_wav2lip(video_path, tts_audio_path, output_video_path, wav2lip_checkpoint_path)

if __name__ == "__main__":
    main()