# CUDA_VISIBLE_DEVICES=0 python -m scripts.train_latent_paint_mesh \
# --log.exp_name monster_ball14-grey_mask \
# --guide.text "pokemon, monster ball, red and white color, on a white back ground" \
# --guide.shape_path shapes/sphere.obj \
# --guide.image /source/kseo/diffusion_playground/latent-nerf-test/data/monster_ball.jpg
# # --optim.iters 10000 \
# # --optim.resume True \
# # --optim.ckpt /source/kseo/diffusion_playground/latent-nerf-test/experiments/monster_ball4-grey_mask/checkpoints/step_005000.pth

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_latent_paint_mesh \
--log.exp_name smpl-grey_mask-SD13 \
--guide.text "a photo of a single person posing and wearing a t-shirt and short pants, full body" \
--guide.shape_path shapes/smpl-aaron_posed_005.obj \
--guide.dy "0.5" \
--guide.shape_scale "0.8" \
--optim.iters 2000 \
--optim.use_SD True \
--guide.image img/rp_aaron_posed_005_A.jpg
# --optim.use_opt_txt True \