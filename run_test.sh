CUDA_VISIBLE_DEVICES=0 python -m scripts.train_latent_paint_mesh \
--log.exp_name monster_ball14-grey_mask \
--guide.text "pokemon, monster ball, red and white color, on a white back ground" \
--guide.shape_path shapes/sphere.obj \
--guide.image /source/kseo/diffusion_playground/latent-nerf-test/data/monster_ball.jpg
# --optim.iters 10000 \
# --optim.resume True \
# --optim.ckpt /source/kseo/diffusion_playground/latent-nerf-test/experiments/monster_ball4-grey_mask/checkpoints/step_005000.pth
