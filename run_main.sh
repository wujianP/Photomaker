conda activate /discobox/wjpeng/env/photomaker
cd /discobox/wjpeng/code/202312/PhotoMaker

python main_real.py \
--output='/DDN_ROOT/wjpeng/zoo/photomaker/output' \
--input='/DDN_ROOT/wjpeng/zoo/photomaker/input/ym' \
--exp-name='ym' \
--prompt="sci-fi, closeup portrait photo of an aisa man img in superhero suit, draped in a cape, and wielding a longsword, strong muscle, face, high quality, film grain" \
--neg-prompt="(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth" \
--n-images=2

