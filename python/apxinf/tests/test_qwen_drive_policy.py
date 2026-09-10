"""Public VQA preprocessing and generation-option regression coverage."""
import numpy as np
import pytest

from apxinf.policies.impls.qwen_drive import QwenDrivePolicy


class Tokenizer:
    def token_id(self, token):
        return {'<|im_start|>': 10001, '<|im_end|>': 10002}[token]

    def encode(self, text):
        return list(text.encode())

    def decode(self, ids, skip_special_tokens=True):
        return 'answer'


class Model:
    def generate_tokens(self, ids, pixels, grids, max_new, min_new, eos):
        self.observed = (ids, pixels, grids, max_new)
        return [7, 8]


@pytest.mark.parametrize('surface', ['views', 'images'])
def test_vqa_preserves_frame_target_size_and_requested_generation_limit(surface):
    model = Model()
    config = {
        'vlm_config': {'image_token_id': 10003, 'vision_start_token_id': 10004,
                       'vision_end_token_id': 10005},
        'image_patch_size': 16, 'image_spatial_merge_size': 2,
        'image_temporal_patch_size': 2, 'history_image_pixels': 1024,
        'current_image_pixels': 1024, 'trajectory_scale': [1, 1, 1],
        'num_future_points': 2, 'trajectory_point_dim': 3,
    }
    policy = QwenDrivePolicy(model, config=config, tokenizer=Tokenizer(), mode='vqa',
                             eos_token_ids=[8], seed=42, max_new_tokens=2048,
                             min_new_tokens=0, num_steps=10)
    frame = {'image': np.zeros((64, 96, 3), dtype=np.uint8), 'target_size': [128, 64]}
    observation = {'mode': 'vqa', 'question': 'What is here?', 'max_new_tokens': 2048}
    observation[surface] = {'<FRONT VIEW>': [frame]} if surface == 'views' else [frame]
    result = policy.infer(observation)
    ids, pixels, grids, limit = model.observed
    assert grids == [[1, 4, 8]]
    assert pixels.shape == (32, 1536)
    assert np.all(pixels == -1)
    assert ids.count(10003) == 8
    assert limit == 2048
    assert result['token_ids'] == [7, 8]
