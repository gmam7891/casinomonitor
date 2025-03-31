import os
from utils import capturar_frame_ffmpeg_imageio

def test_ffmpeg_capture():
    test_url = "https://vod-secure.twitch.tv/..."  # coloque um m3u8 válido aqui
    output_path = "tests/testdata/test_frame.jpg"
    result = capturar_frame_ffmpeg_imageio(test_url, output_path, skip_seconds=5)
    assert result and os.path.exists(result)
    os.remove(output_path)
