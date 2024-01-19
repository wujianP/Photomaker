conda activate /discobox/wjpeng/env/photomaker
cd /discobox/wjpeng/code/202312/PhotoMaker

python main_real.py \
--output='/DDN_ROOT/wjpeng/zoo/photomaker/output' \
--input='/DDN_ROOT/wjpeng/zoo/photomaker/input/wj' \
--exp-name='wj' \
--prompt="sci-fi, closeup portrait photo of an aisa man img wearing the sunglasses in Iron man suit, face, slim body, high quality, film grain" \
--neg-prompt="(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"
