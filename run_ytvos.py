import alphaclip
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from utils import *
from ssa_module import compute_ssa_scores
import argparse
import os
import cv2
import json
import numpy as np
from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')


def test(args):

    # initialize EVF-SAM
    tokenizer, evfsam = init_models()

    # initialize Alpha-CLIP
    clip, clip_preprocess = alphaclip.load('ViT-L/14@336px', alpha_vision_ckpt_pth=args.alpha_clip_ckpt, device='cuda')
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

    # initialize Cutie
    cutie = get_default_model(config='ytvos_config')
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    # load videos
    output_dir = 'outputs'
    save_path_prefix = os.path.join(output_dir, 'Ref_YTVOS_val')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    root = args.data_root
    img_folder = os.path.join(root, args.img_folder_rel)
    meta_file = os.path.join(root, args.meta_file_rel)
    with open(meta_file, 'r') as f:
        data = json.load(f)['videos']
    valid_test_videos = set(data.keys())
    test_meta_file_rel = args.meta_file_rel.replace('valid', 'test')
    test_meta_file = os.path.join(root, test_meta_file_rel)
    test_videos = set()
    if os.path.exists(test_meta_file):
        with open(test_meta_file, 'r') as f:
            test_data = json.load(f)['videos']
        test_videos = set(test_data.keys())
    else:
        fallback_test_meta_file = os.path.join(root, 'meta_expressions', 'test', 'meta_expressions.json')
        if os.path.exists(fallback_test_meta_file):
            with open(fallback_test_meta_file, 'r') as f:
                test_data = json.load(f)['videos']
            test_videos = set(test_data.keys())
        else:
            print(f"Warning: Test meta file not found at {test_meta_file} or {fallback_test_meta_file}. No videos will be excluded from the validation list.")
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])

    # inference
    for idx_, video in enumerate(video_list):
        print(f"[{idx_+1}/{len(video_list)}] Processing Video: {video}")
        metas = []
        expressions = data[video]['expressions']
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        for i in range(num_expressions):
            meta = {}
            meta['video'] = video
            meta['exp'] = expressions[expression_list[i]]['exp']
            meta['exp_id'] = expression_list[i]
            meta['frames'] = data[video]['frames']
            metas.append(meta)
        meta = metas
        video_name = video
        frames = data[video]['frames']
        video_len = len(frames)

        # input pre-process
        imgs_beit = []
        imgs_sam = []
        imgs_clip = []
        imgs_cutie = []
        for i in range(video_len):
            img_path = os.path.join(img_folder, video_name, frames[i] + '.jpg')
            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            # BEiT pre-process
            img_beit = beit3_preprocess(Image.open(img_path), 224)
            imgs_beit.append(img_beit)

            # SAM pre-process
            img_sam, resize_shape = sam_preprocess(image_np)
            imgs_sam.append(img_sam)

            # Alpha-CLIP pre-process
            img_clip = clip_preprocess(Image.open(img_path))
            imgs_clip.append(img_clip)

            # Cutie pre-process
            img_cutie = tv.transforms.ToTensor()(Image.open(img_path))
            imgs_cutie.append(img_cutie)

        # for each language
        for e in range(num_expressions):

            # make files
            video_name = meta[e]['video']
            exp = meta[e]['exp']
            exp_id = meta[e]['exp_id']
            frames = meta[e]['frames']
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # per-frame mask prediction
            ref_masks = []
            ref_scores_seg = []
            ref_scores_vl = []
            image_features_list = []
            ref_num = 10
            for ref_idx in range(ref_num):
                i = int(ref_idx * (video_len - 1) / (ref_num - 1))
                words = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()
                ref_mask, ref_score = evfsam.inference(imgs_sam[i].unsqueeze(0).cuda(), imgs_beit[i].unsqueeze(0).cuda(), words, resize_shape, original_size_list)
                ref_mask = (ref_mask > 0).float()
                ref_masks.append(ref_mask)
                ref_scores_seg.append(ref_score)

                # consider vision-text alignment in addition to segmentation confidence
                clip_text = alphaclip.tokenize([exp]).cuda()
                alpha = clip_preprocess_mask(ref_mask).cuda()
                image_features = clip.visual(imgs_clip[i].unsqueeze(0).cuda(), alpha.unsqueeze(0))
                text_features = clip.encode_text(clip_text)
                
                # Keep original features for SSA (Inter-Frame Semantic Consistency)
                image_features_list.append(image_features.detach())

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                vl_score = torch.matmul(image_features, text_features.transpose(0, 1))[0]
                ref_scores_vl.append(vl_score)

            # SSA Score integration
            ssa_scores = compute_ssa_scores(image_features_list)
            
            # Min-Max Normalization to solve score scale dominance
            seg_vals = torch.stack(ref_scores_seg)
            vl_vals = torch.stack(ref_scores_vl)
            ssa_vals = ssa_scores
            
            seg_min, seg_max = seg_vals.min(), seg_vals.max()
            vl_min, vl_max = vl_vals.min(), vl_vals.max()
            ssa_min, ssa_max = ssa_vals.min(), ssa_vals.max()
            
            seg_range = (seg_max - seg_min).clamp(min=1e-6)
            vl_range = (vl_max - vl_min).clamp(min=1e-6)
            ssa_range = (ssa_max - ssa_min).clamp(min=1e-6)
            
            norm_seg = (seg_vals - seg_min) / seg_range
            norm_vl = (vl_vals - vl_min) / vl_range
            norm_ssa = (ssa_vals - ssa_min) / ssa_range
            
            # Combine scores (Segmentation Confidence, Vision-Language Alignment, SSA Consistency)
            w1, w2, w3 = 0.4, 0.4, 0.2
            ref_scores = []
            for idx in range(ref_num):
                total_score = w1 * norm_seg[idx] + w2 * norm_vl[idx] + w3 * norm_ssa[idx]
                ref_scores.append(total_score)

            # select multiple reference frames using temporal diversity filtering
            candidate_indices = [int(ref_idx * (video_len - 1) / (ref_num - 1)) for ref_idx in range(ref_num)]
            scores_list = [s.item() for s in ref_scores]
            
            # Sort candidate indices by combined score descending
            sorted_indices = np.argsort(scores_list)[::-1]
            selected_candidate_indices = []
            
            for idx in sorted_indices:
                if len(selected_candidate_indices) >= args.num_references:
                    break
                current_frame_pos = candidate_indices[idx]
                diverse = True
                for sel_idx in selected_candidate_indices:
                    sel_frame_pos = candidate_indices[sel_idx]
                    if abs(current_frame_pos - sel_frame_pos) < args.min_frame_distance:
                        diverse = False
                        break
                if diverse:
                    selected_candidate_indices.append(idx)
            
            # Fallback if we couldn't find enough diverse references
            if len(selected_candidate_indices) < args.num_references:
                for idx in sorted_indices:
                    if len(selected_candidate_indices) >= args.num_references:
                        break
                    if idx not in selected_candidate_indices:
                        selected_candidate_indices.append(idx)
            
            # Sort selected references chronologically
            selected_candidate_indices.sort()
            selected_refs = [candidate_indices[idx] for idx in selected_candidate_indices]
            earliest_ref_idx = selected_refs[0]
            earliest_candidate_idx = selected_candidate_indices[0]
            
            print(f"  => Selected Reference Frames for Tracking: {selected_refs}")

            # forward pass
            for i in range(earliest_ref_idx, video_len):
                if i in selected_refs:
                    ref_list_idx = selected_refs.index(i)
                    cand_idx = selected_candidate_indices[ref_list_idx]
                    mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[cand_idx].squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                # clear memory for each sequence
                if i == video_len - 1:
                    processor.clear_memory()

                # convert format
                mask = mask.detach().cpu().numpy().astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask.save(save_file)

            # backward pass
            for i in range(earliest_ref_idx, -1, -1):
                if i == earliest_ref_idx:
                    cand_idx = earliest_candidate_idx
                    mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[cand_idx].squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                # clear memory for each sequence
                if i == 0:
                    processor.clear_memory()

                # convert format
                mask = mask.detach().cpu().numpy().astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask.save(save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../DB/RVOS/YTVOS')
    parser.add_argument('--alpha_clip_ckpt', type=str, default='weights/clip_l14_336_grit_20m_4xe.pth')
    parser.add_argument('--img_folder_rel', type=str, default='valid/JPEGImages')
    parser.add_argument('--meta_file_rel', type=str, default='meta_expressions/valid/meta_expressions.json')
    # Accept other arguments silently if passed, so the user's command doesn't crash
    parser.add_argument('--use_temporal_score', action='store_true')
    parser.add_argument('--tracker', type=str, default='cutie')
    parser.add_argument('--num_references', type=int, default=3)
    parser.add_argument('--min_frame_distance', type=int, default=15)
    parser.add_argument('--multi_reference', action='store_true')
    
    args = parser.parse_args()

    torch.cuda.set_device(0)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        test(args)
