from torch import nn
import torch
import torch.nn.functional as F
# from utils import static_preds, unary_preds, binary_preds
import sys
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel, AutoProcessor
sys.path.append("/home/nvelingker/LASER/VidVRD-II/baseline")
from model_utils import extract_single_object, extract_object_subject, crop_image_contain_bboxes

def get_print_hook(name):
    def print_hook(grad):
        print(f"{name}: \n {grad} \n")
        return grad
    return print_hook

def segment_list(l, n=5):
    current_seg = []
    all_segs = []
    
    for item in l:
        current_seg.append(item)
        if len(current_seg) >= n:
            all_segs.append(current_seg)
            current_seg = []
    
    if not len(current_seg) == 0:
        all_segs.append(current_seg)
        
    return all_segs

def get_tensor_size(a):
    return a.element_size() * a.nelement()
    
def comp_diff(v1, v2): 
    return 2 * torch.abs(v1 - v2) / (v1 + v2)

def gather_names(pred_res):
    all_names = set()
    for name, _ in pred_res:
        all_names.add(name)
    return list(all_names)

def extract_nl_feats(tokenizer, model, names, device):
    if len(names) == 0:
        features = []
    else:
        name_tokens = tokenizer(names, padding=True, return_tensors="pt").to(device)
        features = model.get_text_features(**name_tokens)
    return features

def extract_all_nl_feats(tokenizer, model, batch_size, batched_names, batched_unary_kws, batched_binary_kws, device):
    batched_obj_name_features = [[] for _ in range(batch_size)]
    batched_unary_nl_features = [[] for _ in range(batch_size)]
    batched_binary_nl_features = [[] for _ in range(batch_size)]
    
    # Step 1: compare the video objects with the nouns in the natural language
    # Memory usage should be small
        
    for vid, (object_names, unary_kws, binary_kws) in \
        enumerate(zip(batched_names, batched_unary_kws, batched_binary_kws)):
        
        obj_name_features = extract_nl_feats(tokenizer, model, object_names, device)
        batched_obj_name_features[vid] = obj_name_features
        
        unary_features = extract_nl_feats(tokenizer, model, unary_kws, device)
        batched_unary_nl_features[vid] = unary_features
        
        binary_features = extract_nl_feats(tokenizer, model, binary_kws, device)
        batched_binary_nl_features[vid] = binary_features
        
    return batched_obj_name_features, batched_unary_nl_features, batched_binary_nl_features
                    
def single_object_crop(batch_size, batched_videos, batched_object_ids, batched_bboxes, batched_video_splits):
    batched_frame_bboxes = {}
    batched_cropped_objs = [[] for _ in range(batch_size)]
    
    for (video_id, frame_id, obj_id), bbox in zip(batched_object_ids, batched_bboxes):
        overall_frame_id = batched_video_splits[video_id] + frame_id
        if type(bbox) == dict:
            bx1, by1, bx2, by2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        else:
            bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
        assert by2 > by1
        assert bx2 > bx1
        batched_cropped_objs[video_id].append((batched_videos[overall_frame_id][by1:by2, bx1:bx2]))
        batched_frame_bboxes[video_id, frame_id, obj_id] = (bx1, by1, bx2, by2)
    
    return batched_cropped_objs, batched_frame_bboxes

                    
class PredicateModel(nn.Module):
    
    def __init__(self, hidden_dim, num_top_pairs, device,  model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.device = device
        
        self.clip_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.clip_model = AutoModel.from_pretrained(model_name).to(device)
        self.clip_processor = AutoProcessor.from_pretrained(model_name)
        self.save_debug_log_path = f"/home/jianih/research/LASER/data/LLaVA-Video-178K-v2/outputs/debug/processed_{device}.log"
    
    def clip_sim(self, nl_feat, img_feat):
        img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)  
        nl_feat = nl_feat / nl_feat.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale
        logits_per_text = torch.matmul(nl_feat, img_feat.t()) * logit_scale.exp()

        return logits_per_text
    
   
    def forward(self, 
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
                batched_binary_predicates,
                unary_segment_size=None,
                binary_segment_size=None, 
                alpha=0.5, 
                white_alpha=0.8,
                topk_cate=3):
        
        batched_obj_name_features = []
        batched_unary_nl_features = []
        batched_binary_nl_features = []
        batched_object_ids_lookup = {}
        batch_size = len(batched_video_ids)
        for video_id in range(len(batched_video_ids)):
            batched_object_ids_lookup[video_id] = []
        
        # with open(self.save_debug_log_path, 'a') as f:
        #     f.write(f"start forwarding predicate model: {batched_video_ids}\n")
            
        # Step 1: compare the video objects with the nouns in the natural language
        for object_names, unary_kws, binary_preds in \
            zip(batched_names, batched_unary_kws, batched_binary_kws):

            if len(object_names) == 0:
                batched_obj_name_features.append([])
            else:
                obj_name_tokens = self.clip_tokenizer(object_names, return_tensors="pt", max_length=75, truncation=True, padding='max_length',).to(self.device)
                # obj_name_tokens = self.clip_tokenizer(object_names, return_tensors="pt",  padding=True,).to(self.device)

                obj_name_features = self.clip_model.get_text_features(**obj_name_tokens)
                batched_obj_name_features.append(obj_name_features)

            if len(unary_kws) == 0:
                batched_unary_nl_features.append([])
            else:
                # unary_tokens = self.clip_tokenizer(list(unary_kws), return_tensors="pt", padding=True,).to(self.device)
                unary_tokens = self.clip_tokenizer(list(unary_kws), return_tensors="pt", max_length=75, truncation=True, padding='max_length',).to(self.device)

                unary_features = self.clip_model.get_text_features(**unary_tokens)
                batched_unary_nl_features.append(unary_features)

            if len(binary_preds) == 0:
                batched_binary_nl_features.append([])
            else:
                # nl_tokens = self.clip_tokenizer(list(binary_preds), return_tensors="pt", padding=True,).to(self.device)
                nl_tokens = self.clip_tokenizer(list(binary_preds), return_tensors="pt", max_length=75, truncation=True, padding='max_length',).to(self.device)

                nl_features = self.clip_model.get_text_features(**nl_tokens)
                batched_binary_nl_features.append(nl_features)

        # with open(self.save_debug_log_path, 'a') as f:
        #     f.write(f"generated nl features predicate model: {batched_video_ids}\n")
            
        # Step 2: crop the objects and obtain the embedding for videos
        norm_boxes = []
        batched_frame_masks = {}
        batched_frame_bboxes = {}
        batched_cropped_objs = {}
        
        for vid in range(batch_size):
            batched_cropped_objs[vid] = []
            
        current_vid, current_frame_id = -1, -1
        batched_video_splits = [0] + batched_video_splits

        for (video_id, frame_id, obj_id), mask, bbox in zip(batched_object_ids, batched_masks, batched_bboxes):
            
            overall_frame_id = batched_video_splits[video_id] + frame_id
            try:
                object_img = extract_single_object(batched_videos[overall_frame_id], mask, white_alpha)
            except:
                print(f"Error: {batched_video_ids}")
                continue
                
            cropped_object_img = crop_image_contain_bboxes(object_img, [bbox], batched_video_ids)

            current_vid = video_id
            batched_frame_masks[video_id, frame_id, obj_id] = mask
            batched_frame_bboxes[video_id, frame_id, obj_id] = bbox
            
            batched_object_ids_lookup[video_id].append((frame_id, obj_id))
            batched_cropped_objs[current_vid].append(cropped_object_img)
            
        # with open(self.save_debug_log_path, 'a') as f:
        #     f.write(f"generated object crops: {batched_video_ids}\n")
            
        # Step 3: get the similarity for nl and single objects
        batched_image_unary_probs = {}
        batched_image_cate_probs = {}
        batched_obj_cate_features = {}
        batched_obj_per_cate = {}
        
        for vid in range(batch_size):
            batched_image_unary_probs[vid] = {}
            batched_image_cate_probs[vid] = {}
            batched_obj_cate_features[vid] = {}
            batched_obj_per_cate[vid] = {}
            
        for vid_id, (unary_nl_feats, object_name_feats, cate, unary_pred, binary_predicates) \
            in enumerate(zip(batched_unary_nl_features, batched_obj_name_features, batched_names, batched_unary_kws, batched_binary_predicates)):
            cropped_objs  = batched_cropped_objs[vid_id]
            
            if not len(cropped_objs) == 0:
                inputs = self.clip_processor(images=cropped_objs, return_tensors="pt")
                inputs = inputs.to(self.device)
                obj_clip_features = self.clip_model.get_image_features(**inputs)
                batched_obj_cate_features[vid_id] = obj_clip_features
            else:
                batched_obj_cate_features[vid_id] = torch.tensor([])

            if len(object_name_feats) == 0 or len(batched_object_ids_lookup[vid_id]) == 0 or len(cropped_objs) == 0:
                cate_logits_per_text = torch.tensor([])
            else:
                cate_logits_per_text = self.clip_sim(object_name_feats, obj_clip_features)
                cate_logits_per_text = cate_logits_per_text.softmax(dim=0)

            # Put up the categorical probabilities per object base
            object_ids = batched_object_ids_lookup[vid_id]
            if not (len(object_ids) == 0 or (len(cate_logits_per_text.shape) == 2 and cate_logits_per_text.shape[1] == len(object_ids))):
                print('here')
            assert len(object_name_feats) == 0 or len(object_ids) == 0 or (len(cate_logits_per_text.shape) == 2 and cate_logits_per_text.shape[1] == len(object_ids)), f"Mismatched object id and cate logic: {batched_video_ids}"
            
            cate_prob_per_obj = {}
            for cate_name, probs in zip(cate, cate_logits_per_text):
                for prob, (fid, oid) in zip(probs, object_ids):
                    if not oid in cate_prob_per_obj:
                        cate_prob_per_obj[oid] = {}
                    if not cate_name in cate_prob_per_obj[oid]:
                        cate_prob_per_obj[oid][cate_name] = []
                    cate_prob_per_obj[oid][cate_name].append(prob)
                    
            new_cate_prob_per_obj = {}
            obj_per_cate = {}
            for oid, object_cate_info in cate_prob_per_obj.items():
                for cate_name, prob in object_cate_info.items():
                    if not cate_name in obj_per_cate:
                        obj_per_cate[cate_name] = []
                    prob = torch.mean(torch.stack(prob))
                    obj_per_cate[cate_name].append((prob, oid))
                    new_cate_prob_per_obj[(oid, cate_name)] = prob
                    
            for cate_name in obj_per_cate:
                obj_per_cate[cate_name] = sorted(obj_per_cate[cate_name], reverse=True)
            
            # Not require grad on general unary
            if len(unary_nl_feats) == 0 or len(cropped_objs) == 0:
                unary_logits_per_text = torch.tensor([])
            else:
                unary_logits_per_text = self.clip_sim(unary_nl_feats, obj_clip_features)
                unary_logits_per_text = unary_logits_per_text.softmax(dim=0).detach()

            unary_prob_per_obj = {}
            for unary_name, probs in zip(unary_pred, unary_logits_per_text):
                for prob, (fid, oid) in zip(probs, object_ids):
                    unary_prob_per_obj[(fid, oid, unary_name)] = prob
            
            batched_image_cate_probs[vid_id] = new_cate_prob_per_obj
            batched_image_unary_probs[vid_id] = unary_prob_per_obj
            batched_obj_per_cate[vid_id] = obj_per_cate
        
        # with open(self.save_debug_log_path, 'a') as f:
        #     f.write(f"generated unary and categorical: {batched_video_ids}\n")
            
        # Step 4: get the similarity for object pairs
        batched_cropped_obj_pairs = {}
        frame_splits = {}
        current_info = (0, 0)
        frame_splits[current_info] = {'start': 0}

        batched_topk_cate_candidates = {}
        for video_id in range(batch_size):
            batched_topk_cate_candidates[video_id] = {}
            
        for vid, obj_per_cate in batched_obj_per_cate.items():
            topk_cate_candidates = {}
            for cate_name, pred_oid_ls in obj_per_cate.items():
                for _, oid in pred_oid_ls[:topk_cate]:
                    if not cate_name in topk_cate_candidates:
                        topk_cate_candidates[cate_name] = []
                    topk_cate_candidates[cate_name].append(oid)
            batched_topk_cate_candidates[vid] = topk_cate_candidates
        
        obj_pair_lookup = {}
        for video_id in range(batch_size):
            obj_pair_lookup[video_id] = {}
             
        for (vid, fid, (from_oid, to_oid)) in batched_obj_pairs:
            if not (from_oid, to_oid) in obj_pair_lookup[vid]:
                obj_pair_lookup[vid][(from_oid, to_oid)] = []
            obj_pair_lookup[vid][(from_oid, to_oid)].append(fid)
            
        selected_pairs = set()
        # selected_pairs_with_binary = set()
        
        if batched_binary_predicates[0] is None:
            selected_pairs = batched_obj_pairs
        else:
            for bp_vid, binary_predicates in enumerate(batched_binary_predicates):
                topk_cate_candidates = batched_topk_cate_candidates[bp_vid]
                for (rel_name, from_obj_name, to_obj_name) in binary_predicates:
                    if from_obj_name in topk_cate_candidates and to_obj_name in topk_cate_candidates:
                        from_oids = topk_cate_candidates[from_obj_name]
                        to_oids = topk_cate_candidates[to_obj_name]
                        for from_oid in from_oids:
                            for to_oid in to_oids:
                                if bp_vid in obj_pair_lookup and (from_oid, to_oid) in obj_pair_lookup[bp_vid]:
                                    for fid in obj_pair_lookup[bp_vid][(from_oid, to_oid)]:
                                        selected_pairs.add((bp_vid, fid, (from_oid, to_oid)))
                                        # selected_pairs_with_binary.add(((bp_vid, fid, (from_oid, to_oid)), rel_name))
                
        selected_pairs = list(selected_pairs)

        new_select_pairs = {}
        for video_id in range(len(batched_video_ids)):
            new_select_pairs[video_id] = []
        for (vid, fid, (from_oid, to_oid)) in selected_pairs:
            new_select_pairs[vid].append((vid, fid, (from_oid, to_oid)))
        
        # with open(self.save_debug_log_path, 'a') as f:
        #     f.write(f"selected object pair number: {batched_video_ids}: {len(selected_pairs)}\n")
            
        for vid in range(len(batched_video_ids)):
                batched_cropped_obj_pairs[vid] = []
                
        for (vid, fid, (from_id, to_id)) in selected_pairs:
            overall_frame_id = batched_video_splits[vid] + fid
            mask1 = batched_frame_masks[(vid, fid, from_id)]
            mask2 = batched_frame_masks[(vid, fid, to_id)]
            bbox1 = batched_frame_bboxes[(vid, fid, from_id)]
            bbox2 = batched_frame_bboxes[(vid, fid, to_id)]
            bb_pop_image = extract_object_subject(batched_videos[overall_frame_id], mask1, mask2, alpha=0.05, white_alpha=0.3)
            cropped_bb_pop_image = crop_image_contain_bboxes(img=bb_pop_image, bbox_ls=[bbox1, bbox2], data_id=batched_video_ids)
            
            batched_cropped_obj_pairs[vid].append(cropped_bb_pop_image)

        batched_image_binary_probs = []
        if len(batched_cropped_obj_pairs) == 0:
            batched_image_binary_probs.append({})
        else:
            for vid, binary_nl_features in enumerate(batched_binary_nl_features):

                if len(binary_nl_features) == 0:
                    batched_image_binary_probs.append({})
                    continue

                binary_kws = batched_binary_kws[vid]

                cropped_obj_pairs = batched_cropped_obj_pairs[vid]
                if len(cropped_obj_pairs) == 0:
                    batched_image_binary_probs.append({})
                    continue
                
                inputs = self.clip_processor(images=cropped_obj_pairs, return_tensors="pt")
                inputs = inputs.to(self.device)

                obj_features = self.clip_model.get_image_features(**inputs)
                obj_clip_features = obj_features / obj_features.norm(p=2, dim=-1, keepdim=True)
                binary_nl_features = binary_nl_features / binary_nl_features.norm(p=2, dim=-1, keepdim=True)

                logit_scale = self.clip_model.logit_scale
                binary_logits_per_text = torch.matmul(binary_nl_features, obj_clip_features.t()) * logit_scale.exp()
                # Compute row-wise min and max
                row_min = binary_logits_per_text.min(dim=1, keepdim=True)[0]  # Shape: (batch_size, 1)
                row_max = binary_logits_per_text.max(dim=1, keepdim=True)[0]  # Shape: (batch_size, 1)

                # Compute the range (max - min)
                row_range = row_max - row_min  # Shape: (batch_size, 1)

                # Perform min-max normalization
                binary_logits_per_text = (binary_logits_per_text - row_min) / row_range

                binary_prob_per_obj = {}
                for binary_name, probs in zip(binary_kws, binary_logits_per_text):
                    for prob, (vid, fid, obj_pair) in zip(probs, new_select_pairs[vid]):
                        # choose_tp = ((vid, fid, obj_pair), binary_name)
                        # if choose_tp in selected_pairs_with_binary:
                        binary_prob_per_obj[(fid, obj_pair, binary_name)] = prob
                batched_image_binary_probs.append(binary_prob_per_obj)

        # with open(self.save_debug_log_path, 'a') as f:
        #     f.write(f"generated binary probs: {batched_video_ids}: {len(selected_pairs)}\n")
            
        return batched_image_cate_probs, batched_image_unary_probs, batched_image_binary_probs
    