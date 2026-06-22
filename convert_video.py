import subprocess

input_file = r"/mnt/nvme/opt/Tracking_research/assets/ps_Long/1_2026-04-23_144019.3gp"
output_file = r"/mnt/nvme/opt/Tracking_research/assets/ps_Long/1_2026-04-23_144019.mp4"

command = [
    "ffmpeg",
    "-i", input_file,
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-r", "30",
    "-preset", "fast",
    "-c:a", "aac",
    output_file
]

result = subprocess.run(command)

if result.returncode == 0:
    print("Convert completed!")
else:
    print("Conversion failed!")