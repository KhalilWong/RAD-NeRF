import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from .utils import *

from .asr import ASR

import cv2

class NeRFNoGUILive:
    def __init__(self, opt, trainer, data_loader, debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.debug = debug

        self.trainer = trainer
        self.data_loader = data_loader

        # override with dataloader's intrinsics
        self.W = data_loader._data.W
        self.H = data_loader._data.H

        # use dataloader's pose
        pose_init = data_loader._data.poses[0]

        # use dataloader's bg
        bg_img = data_loader._data.bg_img #.view(1, -1, 3)
        if self.H != bg_img.shape[0] or self.W != bg_img.shape[1]:
            bg_img = F.interpolate(bg_img.permute(2, 0, 1).unsqueeze(0).contiguous(), (self.H, self.W), mode='bilinear').squeeze(0).permute(1, 2, 0).contiguous()
        self.bg_color = bg_img.view(1, -1, 3)

        # audio features (from dataloader, only used in non-playing mode)
        self.audio_features = data_loader._data.auds # [N, 29, 16]
        self.audio_idx = 0

        # control eye
        self.eye_area = None if not self.opt.exp_eye else data_loader._data.eye_area.mean().item()

        # playing seq from dataloader, or pause.
        self.playing = True# False
        self.loader = iter(data_loader)

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel
        self.mode = 'image' # choose from ['image', 'depth']

        self.downscale = 1

        self.ind_index = 0
        self.ind_num = trainer.model.individual_codes.shape[0]

        # build asr
        if self.opt.asr:
            self.asr = ASR(opt)
        
        # self.test_step()
        

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.opt.asr:
            self.asr.stop()

    def prepare_buffer(self, outputs):
        if self.mode == 'image':
            return outputs['image']
        else:
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)

    def test_step(self):

        if self.need_update or self.spp < self.opt.max_spp:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            if self.playing:
                try:
                    data = next(self.loader)
                except StopIteration:
                    self.loader = iter(self.data_loader)
                    data = next(self.loader)
                
                if self.opt.asr:
                    # use the live audio stream
                    data['auds'] = self.asr.get_next_feat()

                outputs = self.trainer.test_gui_with_data(data, self.W, self.H)
            
            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            if self.need_update:
                self.render_buffer = self.prepare_buffer(outputs)
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
                self.spp += 1
            
            if self.playing:
                self.need_update = True
            
            return int(1000/t), cv2.cvtColor(self.render_buffer, cv2.COLOR_BGR2RGB)

    def render(self):
        if self.opt.asr:
            self.asr.warm_up()
        while True:
            # update every frame
            # audio stream thread...
            if self.opt.asr and self.playing:
                # run 2 ASR steps (audio is at 50FPS, video is at 25FPS)
                for _ in range(2):
                    self.asr.run_step()
            fps, image = self.test_step()
            #print(fps)
            cv2.putText(image, '%.2f' % fps, (5,5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow('MyLive', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if self.opt.asr:
            self.asr.stop()
        cv2.destroyAllWindows()