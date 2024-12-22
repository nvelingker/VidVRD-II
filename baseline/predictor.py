import os
import json
import multiprocessing
from itertools import chain, product
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
# from . import llava_clip_model
from .llava_clip_model import PredicateModel
# from .llava_clip_model import PredicateModel as llava_clip_model
import common
from common import Trajectory, VideoRelation
from video_object_detection.object_tracklet_proposal import get_object_tracklets
from .feature import extract_object_feature, extract_relation_feature
from .model import IndependentClassifier, CascadeClassifier, IterativeClassifier
import cv2
import sys
sys.path.append("/home/nvelingker/LASER/VidVRD-II/baseline")
from llava_clip_model import PredicateModel
def load_video_frames(video_path, start=0, end=None, transpose=False):
    """
    Load frames from a video file within a specified frame range.

    Parameters:
    - video_path (str): Path to the video file.
    - start (int): Index of the first frame to load (0-based).
    - end (int or None): Index of the last frame to load (exclusive).
      If None, load until the end of the video.
    - transpose (bool): If True, transpose the frames from (height, width, 3) to (width, height, 3).

    Returns:
    - frames (list of np.ndarray): A list of frames in RGB format.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    current_frame = start
    while True:
        # If end is specified and current_frame reaches or exceeds 'end', stop reading
        if end is not None and current_frame >= end:
            break

        ret, frame = cap.read()
        if not ret:
            # No more frames to read
            break

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if transpose:
            # Transpose frame shape from (H, W, 3) to (W, H, 3)
            frame_rgb = np.transpose(frame_rgb, (1, 0, 2))

        frames.append(frame_rgb)
        current_frame += 1

    cap.release()
    return frames


class TestDataset(Dataset):
    def __init__(self, raw_dataset, split, **param):
        self.raw_dataset = raw_dataset
        self.split = split
        self.temporal_propagate_threshold = param['inference_temporal_propagate_threshold']
        self.n_workers = param['inference_n_workers']
        self.video_segments = self._get_testing_segments()

    def __len__(self):
        return len(self.video_segments)

    def __getitem__(self, i):
        # Index represents a single video segment
        vid, fstart, fend = self.video_segments[i]
        anno = self.raw_dataset.get_anno(vid)

        # Load features and pairs from relation extraction
        pairs, sub_feats, obj_feats, pred_pos_feats, pred_vis_feats, iou, trackid = extract_relation_feature(
            self.raw_dataset.name, vid, fstart, fend, anno)
        
        # Convert features to float32 if needed
        sub_feats = sub_feats.astype(np.float32)
        obj_feats = obj_feats.astype(np.float32)
        pred_pos_feats = pred_pos_feats.astype(np.float32)
        pred_vis_feats = pred_vis_feats.astype(np.float32)

        # Load tracklets for objects in this segment
        tracklets = self._get_object_tracklets(self.raw_dataset.name, vid, fstart, fend, anno)

        # Build transition matrix if needed (as originally)
        trans_mat = self._build_transition_matrix(i, tracklets, pairs)

        # 1. Video IDs
        batched_video_ids = [vid]

        # 2. Video frames: load frames for this segment
        video_path = self.raw_dataset.get_video_path(vid)
        batched_videos = load_video_frames(video_path, fstart, fend)
        H, W, _ = batched_videos[0].shape if len(batched_videos) > 0 else (720, 1280, 3)

        # 3. Object information
        batched_object_ids = []  # (video_id_idx, frame_id_in_segment, obj_id)
        batched_bboxes = []
        batched_masks = []

        # Map track id (tid) to category name
        objid2category = {}
        for so in anno['subject/objects']:
            objid2category[so['tid']] = so['category']

        frame_count = fend - fstart

        # Check if tracklets have masks
        has_direct_masks = (len(tracklets) > 0 and hasattr(tracklets[0], 'masks'))

        # We also need to build a frame->object mapping to find object pairs' frame IDs
        obj_to_frames = defaultdict(set)

        for obj_id, trac in enumerate(tracklets):
            for offset, bbox in enumerate(trac.rois):
                frame_abs = trac.pstart + offset
                if fstart <= frame_abs < fend:
                    frame_rel = frame_abs - fstart
                    batched_object_ids.append((0, frame_rel, obj_id))
                    batched_bboxes.append(bbox)

                    # Create or retrieve mask
                    if has_direct_masks:
                        mask = trac.masks[offset]  # Assuming trac.masks aligned with trac.rois
                    else:
                        # overlap_bbox is [x1, y1, x2, y2] with floats in [0,1].
                        # W, H are the width and height of the image.

                        x1, y1, x2, y2 = bbox

                        # Convert normalized coords (0.0–1.0) to integer pixel coords
                        x1 = int(x1 * W)
                        y1 = int(y1 * H)
                        x2 = int(x2 * W)
                        y2 = int(y2 * H)

                        # Clamp to stay within the image
                        x1 = max(0, min(x1, W - 1))
                        y1 = max(0, min(y1, H - 1))
                        x2 = max(x1 + 1, min(x2, W))
                        y2 = max(y1 + 1, min(y2, H))

                        # Create the mask
                        mask = np.zeros((H, W, 1), dtype=np.uint8)
                        mask[y1:y2, x1:x2, :] = 1

                        batched_masks.append(mask)


                    # Record that this object appears in frame_rel
                    obj_to_frames[obj_id].add(frame_rel)

        # 4. Object names
        bns = list(self.raw_dataset.so2soid.keys())
        batched_names = [[b.replace("_", " ") for b in bns]]

        # 5. Unary keywords
        batched_unary_kws = [[]]

        # 6. Binary keywords
        bkws = list(self.raw_dataset.pred2pid.keys())

        batched_binary_kws = [[b.replace("_", " ") for b in bkws]]

        # 7. Object pairs
        batched_obj_pairs = []
        for (ti, tj) in pairs:
            common_frames = obj_to_frames[ti].intersection(obj_to_frames[tj])
            if len(common_frames) > 0:
                fid = min(common_frames)  # pick earliest frame
                batched_obj_pairs.append((0, fid, (ti, tj)))

        # 8. Video splits
        batched_video_splits = [0]

        # 9. Binary predicates
        batched_binary_predicates = [[]]
        for rel_inst in anno['relation_instances']:
            sub_tid = rel_inst['subject_tid']
            obj_tid = rel_inst['object_tid']
            pred_name = rel_inst['predicate']
            sub_name = objid2category.get(sub_tid, "unknown")
            obj_name = objid2category.get(obj_tid, "unknown")
            batched_binary_predicates[0].append((pred_name, sub_name, obj_name))

        index = (vid, fstart, fend)

        return (batched_video_ids,
                batched_videos,
                batched_masks,
                batched_bboxes,
                batched_names,
                batched_object_ids,
                batched_unary_kws,
                batched_binary_kws,
                batched_obj_pairs,
                batched_video_splits,
                batched_binary_predicates,
                pairs, sub_feats, obj_feats, pred_pos_feats, pred_vis_feats, iou, tracklets, trans_mat, index)



    def _get_object_tracklets(self, dname, vid, fstart, fend, anno):
        tracklets = []
        _tracklets, _, _ = extract_object_feature(dname, vid, fstart, fend, anno)
        for _trac in _tracklets:
            pstart = fstart + int(_trac['fstart'])
            pend = pstart + len(_trac['bboxes'])
            trac = Trajectory(pstart, pend, _trac['bboxes'], score=_trac['score'])
            tracklets.append(trac)
        return tracklets

    def _build_transition_matrix(self, i, tracklets, pairs):
        # Reuse original logic for temporal propagation if needed
        if self.temporal_propagate_threshold == 1:
            return None
        # Otherwise, replicate logic from original __getitem__
        # Since we are simplifying, let's return None for now unless you need it.
        return None

    def _get_testing_segments(self):
        print('[info] preparing video segments from {} set for testing'.format(self.split))
        video_segments = dict()
        video_indices = self.raw_dataset.get_index(split=self.split)

        if self.n_workers > 0:
            with tqdm(total=len(video_indices)) as pbar:
                pool = multiprocessing.Pool(processes=self.n_workers)
                for vid in video_indices:
                    anno = self.raw_dataset.get_anno(vid)
                    video_segments[vid] = pool.apply_async(_get_testing_segments_for_video,
                            args=(self.raw_dataset.name, vid, anno),
                            callback=lambda _: pbar.update())
                pool.close()
                pool.join()
            for vid in video_segments.keys():
                res = video_segments[vid].get()
                video_segments[vid] = res
        else:
            for vid in tqdm(video_indices):
                anno = self.raw_dataset.get_anno(vid)
                res = _get_testing_segments_for_video(self.raw_dataset.name, vid, anno)
                video_segments[vid] = res
            
        return list(chain.from_iterable(video_segments.values()))



def _is_precede_video_segment(last_index, index):
    return last_index[0] == index[0] and last_index[2] >= index[1]


def _get_testing_segments_for_video(dname, vid, anno):
    video_segments = []
    segs = common.segment_video(0, anno['frame_count'])
    for fstart, fend in segs:
        tracklets, _, _ = extract_object_feature(dname, vid, fstart, fend, anno)
        # if multiple objects detected and the relation features extracted
        if len(tracklets) > 1:
            video_segments.append((vid, fstart, fend))
    return video_segments


@torch.no_grad()
def predict(raw_dataset, split, use_cuda=False, output_json=True, **param):
    test_dataset = TestDataset(raw_dataset, split, **param)
    # data_generator = DataLoader(test_dataset, batch_size=1, num_workers=param['inference_n_workers'], collate_fn=lambda bs: bs[0])
    data_generator = DataLoader(test_dataset, batch_size=1, collate_fn=lambda bs: bs[0])

    model_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'weights',
                              '{}{}'.format(param['model'].get('dump_file_prefix', ''), param['model']['dump_file']))
    print('[info] loading model from file: {}'.format(model_path))

    if param['model']['name'] == 'independent_classifier':
        model = IndependentClassifier(**param)
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, location: storage))
        model.infer_zero_shot_preference(strategy=param['model'].get('zero_shot_preference', 'none'))
        if use_cuda:
            model.cuda()

    elif param['model']['name'] == 'cascade_classifier':
        model = CascadeClassifier(**param)
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, location: storage))
        model.infer_zero_shot_preference(strategy=param['model'].get('zero_shot_preference', 'none'))
        if use_cuda:
            model.cuda()

    elif param['model']['name'] == 'iterative_classifier':
        model = IterativeClassifier(**param)
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, location: storage))
        model.infer_zero_shot_preference(strategy=param['model'].get('zero_shot_preference', 'none'))
        if use_cuda:
            model.cuda()

    elif param['model']['name'] == 'predicate_model':
        # Introduce local hardcoded variables
        model_name = param['model'].get('model_cpkt')
        epoch_num = param['model'].get('epoch')  # Set to a specific epoch number if desired, else None will load the latest
        model_dir = os.path.join(param['model'].get('dir'))

        device = param['model'].get('device', 'error')
        test_num_top_pairs = param['model'].get('num_top_pairs', 30)
        clip_model_name = param['model'].get('clip_model_name', "openai/clip-vit-base-patch32")

        # Check if model directory exists and contains checkpoint
        if not os.path.exists(model_dir) or len(os.listdir(model_dir)) == 0:
            raise FileNotFoundError(f"No model directory or no checkpoints found at {model_dir}")

        print(f"Loading Model: {model_dir}")
        # Find models with model_name in filename
        current_model_names = [existing_model_name for existing_model_name in os.listdir(model_dir) if model_name in existing_model_name]

        if len(current_model_names) == 0:
            raise FileNotFoundError(f"No model files found with base name '{model_name}' in {model_dir}")

        # Extract model IDs
        model_ids = [mn.split('.')[-2] for mn in current_model_names]
        digital_model_ids = [int(mid) for mid in model_ids if mid.isdigit()]

        # Determine which model epoch to load
        if epoch_num is not None:
            latest_model_id = epoch_num
        else:
            if len(digital_model_ids) == 0:
                raise FileNotFoundError(f"No valid model checkpoints with numeric IDs found in {model_dir} and no epoch specified.")
            latest_model_id = max(digital_model_ids)

        final_model_name = model_name + f'.{latest_model_id}.model'
        final_model_path = os.path.join(model_dir, final_model_name)

        if not os.path.exists(final_model_path):
            raise FileNotFoundError(f"Model file {final_model_path} does not exist.")
        sys.path.append("/home/nvelingker/LASER/VidVRD-II/baseline")
        model_info = torch.load(final_model_path, map_location=device)

        if isinstance(model_info, PredicateModel):
            predicate_model = model_info
        elif isinstance(model_info, torch.nn.Module):
            # If we get here and the model is a different module (like a DistributedDataParallel),
            # we are instructed not to add distributed option, so raise an error if distributed
            # or unknown module is found:
            if isinstance(model_info, torch.nn.parallel.distributed.DistributedDataParallel):
                raise RuntimeError("Distributed model detected, not supported in this code.")
            else:
                # Assume it's a state_dict if it's not our PredicateModel instance directly
                predicate_model = PredicateModel(hidden_dim=0,
                                                num_top_pairs=test_num_top_pairs,
                                                device=device,
                                                model_name=clip_model_name).to(device)
                predicate_model.load_state_dict(model_info)
        else:
            # If it's not a PredicateModel or a state_dict, raise an error
            raise RuntimeError(f"Unexpected model info type: {type(model_info)}")

        predicate_model.use_sparse = False
        predicate_model.device = device
        print(f"Loading: {final_model_name}")
        # If we want to track epoch:
        # self.epoch_ct = latest_model_id

        model = predicate_model
        model.eval()


    else:
        raise ValueError(param['model']['name'])

    print('[info] predicting visual relation segments')
    if param['model']['name'] != 'predicate_model':
        model.eval()

    relation_segments = dict()
    for data in tqdm(data_generator):
        (batched_video_ids,
                batched_videos,
                batched_masks,
                batched_bboxes,
                batched_names,
                batched_object_ids,
                batched_unary_kws,
                batched_binary_kws,
                batched_obj_pairs,
                batched_video_splits,
                batched_binary_predicates,
                # Also return what the original code returned if needed:
                pairs, sub_feats, obj_feats, pred_pos_feats, pred_vis_feats, iou, tracklets, trans_mat, index) = data

        if param['model']['name'] == 'predicate_model':
            # Here we assume you can obtain or already have the following:
            # batched_video_ids, batched_videos, batched_masks, batched_bboxes, batched_names, 
            # batched_object_ids, batched_unary_kws, batched_binary_kws, batched_obj_pairs,
            # batched_video_splits, batched_binary_predicates
            #
            # These should come from your data pipeline. If they are not directly provided by TestDataset,
            # you will need to modify TestDataset to provide them.
            
            # Run PredicateModel forward
            (batched_image_cate_probs,
            batched_image_unary_probs,
            batched_image_binary_probs) = model(
                batched_video_ids,
                batched_videos,
                batched_masks,
                batched_bboxes,
                batched_names,
                batched_object_ids,
                batched_unary_kws,
                batched_binary_kws,
                batched_obj_pairs,
                batched_video_splits,
                batched_binary_predicates
            )
            
            # Convert the PredicateModel outputs into s_prob, o_prob, p_prob arrays
            # similar to how IndependentClassifier's predictions are processed.
            # Pseudocode:
            pairs_np = pairs if not isinstance(pairs, torch.Tensor) else pairs.cpu().numpy()
            num_pairs = len(pairs_np)
            num_obj_classes = raw_dataset.get_object_num()
            num_pred_classes = raw_dataset.get_predicate_num()

            s_prob = np.zeros((num_pairs, num_obj_classes), dtype=np.float32)
            o_prob = np.zeros((num_pairs, num_obj_classes), dtype=np.float32)
            p_prob = np.zeros((num_pairs, num_pred_classes), dtype=np.float32)

            # Fill s_prob, o_prob from batched_image_cate_probs
            # Assuming batched_image_cate_probs[0] is a dict { (oid, cate_name): prob }
            # and that sub_id, obj_id correspond to oids from pairs.
            for pair_id, (sub_id, obj_id) in enumerate(pairs_np):
                # subject
                for (oid, cate_name), val in batched_image_cate_probs[0].items():
                    if oid == sub_id:
                        c_id = raw_dataset.get_object_id(cate_name)
                        s_prob[pair_id, c_id] = float(val)
                # object
                for (oid, cate_name), val in batched_image_cate_probs[0].items():
                    if oid == obj_id:
                        c_id = raw_dataset.get_object_id(cate_name)
                        o_prob[pair_id, c_id] = float(val)

            # Fill p_prob from batched_image_binary_probs
            # Assuming keys: (fid, (foid, toid), predicate_name) -> prob
            pred_dict = batched_image_binary_probs[0]
            for pair_id, (sub_id, obj_id) in enumerate(pairs_np):
                pred_probs = {}
                for (fid, (foid, toid), pname), val in pred_dict.items():
                    if foid == sub_id and toid == obj_id:
                        p_id = raw_dataset.get_predicate_id(pname)
                        p_val = float(val)
                        if p_id not in pred_probs:
                            pred_probs[p_id] = []
                        pred_probs[p_id].append(p_val)
                for p_id, vals in pred_probs.items():
                    p_prob[pair_id, p_id] = max(vals)  # or average

            # Apply thresholds and filtering as original code
            obj_background_id = raw_dataset.get_object_num() - 1
            s_max_idx = np.argmax(s_prob, 1)
            o_max_idx = np.argmax(o_prob, 1)
            valid_pair = (s_max_idx != obj_background_id) & (o_max_idx != obj_background_id)
            pairs_np = pairs_np[valid_pair]
            s_prob = s_prob[valid_pair, :-1]
            o_prob = o_prob[valid_pair, :-1]
            p_prob = p_prob[valid_pair]

            # Build predictions
            predictions = []
            object_conf_thres = param['inference_object_conf_threshold']
            predicate_conf_thres = param['inference_predicate_conf_threshold']
            from itertools import product
            for pair_id in range(len(pairs_np)):
                top_s_inds = np.where(s_prob[pair_id] > object_conf_thres)[0]
                top_p_inds = np.where(p_prob[pair_id] > predicate_conf_thres)[0]
                top_o_inds = np.where(o_prob[pair_id] > object_conf_thres)[0]
                for s_class_id, p_class_id, o_class_id in product(top_s_inds, top_p_inds, top_o_inds):
                    s_score = s_prob[pair_id, s_class_id]
                    p_score = p_prob[pair_id, p_class_id]
                    o_score = o_prob[pair_id, o_class_id]
                    r_score = s_score * p_score * o_score
                    sub_id, obj_id = pairs_np[pair_id]
                    predictions.append({
                        'sub_id': sub_id,
                        'obj_id': obj_id,
                        'triplet': (s_class_id, p_class_id, o_class_id),
                        'score': r_score,
                        'triplet_scores': (s_score, p_score, o_score)
                    })

            # Sort and apply NMS
            predictions = sorted(predictions, key=lambda r: r['score'], reverse=True)[:param['inference_topk']]
            if param['inference_nms'] < 1:
                predictions = relation_nms(predictions, iou, param['inference_nms'])

            # Convert class ids to names and form VideoRelation objects
            final_predictions = []
            for r in predictions:
                sub = raw_dataset.get_object_name(r['triplet'][0])
                pred = raw_dataset.get_predicate_name(r['triplet'][1])
                obj = raw_dataset.get_object_name(r['triplet'][2])
                final_predictions.append(VideoRelation(sub, pred, obj, tracklets[r['sub_id']], tracklets[r['obj_id']], r['score']))

            vsig = common.get_segment_signature(*index)
            if output_json:
                relation_segments[vsig] = [rel.serialize(allow_misalign=True) for rel in final_predictions if rel.serialize(allow_misalign=True) is not None]
            else:
                relation_segments[vsig] = final_predictions

        else:
            # Original logic for other models (no change)
            pairs = torch.from_numpy(pairs)
            sub_feats = torch.from_numpy(sub_feats)
            obj_feats = torch.from_numpy(obj_feats)
            pred_pos_feats = torch.from_numpy(pred_pos_feats)
            pred_vis_feats = torch.from_numpy(pred_vis_feats)
            trans_mat = None if trans_mat is None else torch.from_numpy(trans_mat)
            if use_cuda:
                pairs = pairs.cuda()
                sub_feats = sub_feats.cuda()
                obj_feats = obj_feats.cuda()
                pred_pos_feats = pred_pos_feats.cuda()
                pred_vis_feats = pred_vis_feats.cuda()
                trans_mat = None if trans_mat is None else trans_mat.cuda()

            model_predictions = model.predict(pairs, sub_feats, obj_feats, pred_pos_feats, pred_vis_feats, trans_mat=trans_mat,
                                              inference_steps=param['inference_steps'],
                                              inference_problistic=param['inference_problistic'],
                                              inference_object_conf_thres=param['inference_object_conf_threshold'],
                                              inference_predicate_conf_thres=param['inference_predicate_conf_threshold'])

            # supression
            model_predictions = sorted(model_predictions, key=lambda r: r['score'], reverse=True)[:param['inference_topk']]
            if param['inference_nms'] < 1:
                model_predictions = relation_nms(model_predictions, iou, param['inference_nms'])

            predictions = []
            for r in model_predictions:
                sub = raw_dataset.get_object_name(r['triplet'][0])
                pred = raw_dataset.get_predicate_name(r['triplet'][1])
                obj = raw_dataset.get_object_name(r['triplet'][2])
                predictions.append(VideoRelation(sub, pred, obj, tracklets[r['sub_id']], tracklets[r['obj_id']], r['score']))

            vsig = common.get_segment_signature(*index)
            if output_json:
                relation_segments[vsig] = [r.serialize(allow_misalign=True) for r in predictions]
            else:
                relation_segments[vsig] = predictions

    return relation_segments



def relation_nms(relations, iou, suppress_threshold=0.9, max_n_return=None):
    if len(relations) == 0:
        return []
    
    order = sorted(range(len(relations)), key=lambda i: relations[i]['score'], reverse=True)
    if max_n_return is None:
        max_n_return = len(order)

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(relations[i])
        if len(keep) >= max_n_return:
            break
        triplet = relations[i]['triplet']
        sub_id, obj_id = relations[i]['sub_id'], relations[i]['obj_id']
        new_order = []
        for j in order:
            supress = False
            if triplet == relations[j]['triplet']:
                sub_id_j, obj_id_j = relations[j]['sub_id'], relations[j]['obj_id']
                supress = iou[sub_id_j, sub_id]>suppress_threshold and iou[obj_id_j, obj_id]>suppress_threshold
            if not supress:
                new_order.append(j)
        order = new_order

    return keep
