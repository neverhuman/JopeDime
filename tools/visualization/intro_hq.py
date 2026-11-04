"""Encoder commands for native UHD JAIN masters and compatible share copies."""

from __future__ import annotations


UHD_PROFILE = ("prores", "yuv444p12le", "pcm_s24le")
SHARE_PROFILE = ("h264", "yuv420p", "aac")
REC709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
RGB_TO_REC709 = "scale=out_color_matrix=bt709:out_range=tv"
TAG_REC709_FRAMES = "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709"


def master_command(ffmpeg, output, audio, chapters, seconds, hardware):
    command = [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-n", "-f", "rawvideo",
               "-pixel_format", "rgb48le", "-video_size", "3840x2160", "-framerate", "30", "-i", "pipe:0",
               "-i", str(audio), "-f", "ffmetadata", "-i", str(chapters), "-map_chapters", "2",
               "-map", "0:v:0", "-map", "1:a:0", "-t", str(seconds)]
    if hardware:
        codec = ["-vf", f"{RGB_TO_REC709},format=p416le,{TAG_REC709_FRAMES}", "-c:v", "prores_videotoolbox",
                 "-profile:v", "xq", "-allow_sw", "false"]
    else:
        codec = ["-vf", f"{RGB_TO_REC709},{TAG_REC709_FRAMES}", "-c:v", "prores_ks", "-profile:v", "5", "-pix_fmt", "yuv444p10le",
                 "-alpha_bits", "0", "-vendor", "apl0"]
    return [*command, *codec, *REC709, "-c:a", "pcm_s24le", "-movflags", "+faststart+write_colr", str(output)]


def trim_command(ffmpeg, source, output, seconds):
    return [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-n", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0",
            "-map_chapters", "-1", "-t", str(seconds), "-c:v", "copy", "-af", "afade=t=out:st=11.25:d=0.75",
            "-c:a", "pcm_s24le", "-movflags", "+faststart+write_colr", str(output)]


def share_command(ffmpeg, source, output):
    return [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-n", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "libx264", "-crf", "12", "-preset", "slower", "-pix_fmt", "yuv420p", *REC709,
            "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart+write_colr", str(output)]
