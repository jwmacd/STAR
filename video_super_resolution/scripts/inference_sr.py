import os
import torch
from argparse import ArgumentParser, Namespace
import json
from typing import Any, Dict, List, Mapping, Tuple
from easydict import EasyDict

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(base_path)
from video_to_video.video_to_video_model import VideoToVideo_sr
from video_to_video.utils.seed import setup_seed
from video_to_video.utils.logger import get_logger
from video_super_resolution.color_fix import adain_color_fix

from inference_utils import *

logger = get_logger()


class STAR():
    def __init__(self, 
                 result_dir='./results/',
                 file_name='000_video.mp4',
                 model_path='',
                 solver_mode='fast',
                 steps=15,
                 guide_scale=7.5,
                 upscale=4,
                 max_chunk_len=32,
                 batch_size=1,
                 device='cuda:0'
                 ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f'Using device: {self.device}')
        if self.device.type == 'cuda':
            logger.info(f'Total GPU memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.2f} GB')

        self.result_dir = result_dir
        self.file_name = file_name
        self.model_path = model_path
        logger.info(f'checkpoint_path: {self.model_path}')
        os.makedirs(self.result_dir, exist_ok=True)

        model_cfg = EasyDict(__name__='model_cfg')
        model_cfg.model_path = self.model_path
        # Initialize model with device specified
        self.model = VideoToVideo_sr(model_cfg, device=self.device)  # Pass device to model initialization
        
        self.batch_size = batch_size
        self.solver_mode = solver_mode
        self.steps = 15 if solver_mode == 'fast' else steps
        self.guide_scale = guide_scale
        self.upscale = upscale
        self.max_chunk_len = max_chunk_len

    def enhance_a_video(self, video_path, prompt):
        logger.info('input video path: {}'.format(video_path))
        text = prompt
        logger.info('text: {}'.format(text))
        caption = text + self.model.positive_prompt

        input_frames, input_fps = load_video(video_path)
        logger.info('input fps: {}'.format(input_fps))

        video_data = preprocess(input_frames)
        _, _, h, w = video_data.shape
        logger.info('input resolution: {}'.format((h, w)))
        target_h, target_w = h * self.upscale, w * self.upscale   # adjust_resolution(h, w, up_scale=4)
        logger.info('target resolution: {}'.format((target_h, target_w)))

        total_noise_levels = 900
        setup_seed(666)

        # Process video in chunks
        chunk_size = min(self.max_chunk_len, len(video_data))
        outputs = []
        
        for i in range(0, len(video_data), chunk_size):
            chunk = video_data[i:i + chunk_size]
            pre_data = {
                'video_data': chunk, 
                'y': caption,
                'target_res': (target_h, target_w)
            }
            
            try:
                with torch.cuda.amp.autocast():  # Enable automatic mixed precision
                    with torch.no_grad():
                        data_tensor = collate_fn(pre_data, self.device)
                        chunk_output = self.model.test(
                            data_tensor, 
                            total_noise_levels, 
                            steps=self.steps,
                            solver_mode=self.solver_mode, 
                            guide_scale=self.guide_scale,
                            max_chunk_len=chunk_size
                        )
                        outputs.append(chunk_output.cpu())  # Move to CPU immediately
                
                # Clear cache after each chunk
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    logger.error(f"OOM error processing chunk {i}. Try reducing chunk_size or upscale factor.")
                raise e

        # Concatenate all chunks
        output = torch.cat(outputs, dim=0)
        output = tensor2vid(output)

        # Using color fix
        output = adain_color_fix(output, video_data)

        save_video(output, self.result_dir, self.file_name, fps=input_fps)
        return os.path.join(self.result_dir, self.file_name)
    

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--input_path", required=True, type=str, help="input video path")
    parser.add_argument("--save_dir", type=str, default='results', help="save directory")
    parser.add_argument("--file_name", type=str, help="file name")
    parser.add_argument("--model_path", type=str, default='./pretrained_weight/model.pt', help="model path")
    parser.add_argument("--prompt", type=str, default='a good video', help="prompt")
    parser.add_argument("--upscale", type=int, default=4, help='up-scale')
    parser.add_argument("--max_chunk_len", type=int, default=32, help='max_chunk_len')

    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--solver_mode", type=str, default='fast', help='fast | normal')
    parser.add_argument("--steps", type=int, default=15)

    parser.add_argument("--batch_size", type=int, default=1, help="batch size for processing")
    parser.add_argument("--device", type=str, default='cuda:0', help="device to use (cuda:0, cpu)")

    return parser.parse_args()

def main():
    
    args = parse_args()

    input_path = args.input_path
    prompt = args.prompt
    model_path = args.model_path
    save_dir = args.save_dir
    file_name = args.file_name
    upscale = args.upscale
    max_chunk_len = args.max_chunk_len

    steps = args.steps
    solver_mode = args.solver_mode
    guide_scale = args.cfg

    assert solver_mode in ('fast', 'normal')

    star = STAR(
                result_dir=save_dir,
                file_name=file_name,
                model_path=model_path,
                solver_mode=solver_mode,
                steps=steps,
                guide_scale=guide_scale,
                upscale=upscale,
                max_chunk_len=max_chunk_len,
                batch_size=args.batch_size,
                device=args.device
                )

    star.enhance_a_video(input_path, prompt)


if __name__ == '__main__':
    main()
