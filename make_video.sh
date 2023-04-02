# ffmpeg -framerate 15 -pattern_type glob -i '/source/sihun/latent-nerf-test/experiments/full_moon2/vis/eval/*_rgb.png' \
    # -c:v libx264 -pix_fmt yuv420p /source/sihun/latent-nerf-test/experiments/full_moon2/results/out.mp4
ffmpeg -framerate 15 -t 2 -loop 1 -pattern_type glob \
    -i '/source/sihun/latent-nerf-test/experiments/lego_man/vis/eval/5000*_rgb.png' \
    -c:v libx264 -pix_fmt yuv420p /source/sihun/latent-nerf-test/experiments/lego_man/results/out_fin.mp4
# ffmpeg -framerate 15 -t 2 -loop 1 -pattern_type glob \
#     -i '/source/sihun/latent-nerf-test/experiments/rugby_ball/vis/eval/*10000_*_rgb.png' \
#     -c:v libx264 -pix_fmt yuv420p /source/sihun/latent-nerf-test/experiments/rugby_ball/results/out_last.mp4