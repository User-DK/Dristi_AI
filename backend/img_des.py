"""
    Small script for testing on few generic images given the model weights.
    In order to minimize the requirements, it runs only on CPU and images are
    processed one by one.
"""

import torch
import argparse
import pickle
from argparse import Namespace
import warnings

# Suppress specific warnings



from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description


def generate_description(image_paths, load_path):
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    model_dim=512
    N_enc=3
    N_dec=3
    max_seq_len=74
    device='cpu'
    beam_size=5


    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)
    model_args = Namespace(model_dim=model_dim,
                           N_enc=N_enc,
                           N_dec=N_dec,
                           drop_args=drop_args)

    with open('/home/yashraj/tts_image_description_app/models/Expansion_new/ExpansionNet_v2/demo_material/demo_coco_tokens.pickle', 'rb') as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
        eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

    # print("Dictionary loaded ...")

    img_size = 384
    model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                swin_use_checkpoint=False,
                                final_swin_dim=1536,

                                d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=coco_tokens['word2idx_dict'],
                                output_idx2word=coco_tokens['idx2word_list'],
                                max_seq_len=max_seq_len, drop_args=model_args.drop_args,
                                rank=device)
    checkpoint = torch.load(load_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    # print("Model loaded ...")

    input_images = []
    for path in image_paths:
        input_images.append(preprocess_image(path, img_size))

    result = ""

    # print("Generating captions ...\n")
    for i in range(len(input_images)):
        path = image_paths[i]
        image = input_images[i]
        beam_search_kwargs = {'beam_size': beam_size,
                              'beam_max_seq_len': max_seq_len,
                              'sample_or_max': 'max',
                              'how_many_outputs': 1,
                              'sos_idx': sos_idx,
                              'eos_idx': eos_idx}
        with torch.no_grad():
            pred, _ = model(enc_x=image,
                            enc_x_num_pads=[0],
                            mode='beam_search', **beam_search_kwargs)
        pred = tokens2description(pred[0][0], coco_tokens['idx2word_list'], sos_idx, eos_idx)
        # print(path + ' \n\tDescription: ' + pred + '\n')
        result = pred

    # print("Closed.")
    return result


parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--image_paths', type=str,
                        nargs='+')
parser.add_argument('--load_path', type=str)

args = parser.parse_args()
result = generate_description(image_paths=args.image_paths, load_path=args.load_path)
print(result)


# _MFA$ export PYTHONPATH=/home/yashraj/tts_image_description_app/models/Expansion_new/ExpansionNet_v2:$PYTHONPATH
# python3 /home/yashraj/tts_image_description_app/backend/img_des.py  --load_path /home/yashraj/tts_image_description_app/models/Expansion_new/model_files/rf_model.pth --image_paths /home/yashraj/tts_image_description_app/backend/bear.jpeg --image_paths /home/yashraj/tts_image_description_app/backend/bear.jpeg

# python inference.py --sample_text "श्रीलंका और पाकिस्तान में खेला जा रहा एशिया कप अब तक का सबसे विवादित टूर्नामेंट होता जा रहा है।" --language hindi --gender male --output_file male_hindi_output.wav