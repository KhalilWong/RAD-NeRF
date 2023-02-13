import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from .utils import *

from .asr import ASR

import cv2
import time
import re

from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig
import azure.cognitiveservices.speech as speechsdk

from threading import Thread, Event

class NeRFNoGUILive:
    def __init__(self, opt, trainer, data_loader, debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.debug = debug

        self.trainer = trainer
        self.data_loader = data_loader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        print(min(data_loader._data.eye_area), max(data_loader._data.eye_area))
        self.eye_area = 0.5
        self.dynamic_area = self.eye_area
        self.blinking = False
        self.canblink = True
        self.blinkspeed = 0.1
        self.blinkprocessing = 0.0

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
        # if self.opt.asr:
        #     self.asr = ASR(opt)
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
                data['eye'] = torch.FloatTensor([self.eye_area]).view(1, 1).to(self.device)
                data['auds'] = self.asr.get_next_feat()
                if self.blinking:
                    self.blinkprocessing += self.blinkspeed
                    if self.blinkprocessing >= 1.0:
                        self.blinkprocessing = 0.0
                        self.blinking = False
                    self.dynamic_area = (np.cos(self.blinkprocessing * 2.0 * np.pi) + 1) / 2 * self.eye_area
                    data['eye'] = torch.FloatTensor([self.dynamic_area]).view(1, 1).to(self.device)
                #print('DATA EYE:', data['eye'])
                outputs = self.trainer.test_gui_with_data(data, self.W, self.H)
                #print('test camera: ', data['poses_matrix'][0].detach().cpu().numpy())
                #self.cam.update_pose(data['poses_matrix'][0].detach().cpu().numpy())
            
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
        self.asr.warm_up()
        Thread(target=self.BlinkCtrl).start()
        while True:
            # update every frame
            # audio stream thread...
            #s = time.time()
            if self.playing:
                # run 2 ASR steps (audio is at 50FPS, video is at 25FPS)
                for _ in range(2):
                    self.asr.run_step()
            #t = time.time()
            #print('ASR:', t - s)
            fps, image = self.test_step()
            #print(fps)
            cv2.putText(image, '%.2f' % fps, (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow('MyLive', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.canblink = False
        self.asr.stop()
        cv2.destroyAllWindows()
    ###############################################################################################
    def InitTTS(self):
        speech_key, service_region = "41e78ceb208e49e9884e54b1f664f22b", "eastasia"
        speech_config = speechsdk.SpeechConfig(subscription = speech_key, region = service_region)

        speech_config.speech_synthesis_language = "zh-CN"
        speech_config.speech_synthesis_voice_name ="zh-CN-XiaochenNeural"
        
        #speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm)
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm)
        #print('sssssssss', speech_config.output_format)

        audio_config = AudioOutputConfig(use_default_speaker=True)
        synthesizer = SpeechSynthesizer(speech_config = speech_config, audio_config = None)
        return synthesizer
    ###############################################################################################
    def BlinkCtrl(self):
        while self.canblink:
            print('眨眼：y or n, 音频(路径或文本）')
            text = input()
            tlist = text.split(',')
            if len(tlist) > 0:
                if tlist[0] == 'y':
                    self.blinking = True
                if len(tlist) > 1:
                    if tlist[1] != '':
                        if self.opt.asr_nogui == 1:
                            self.asr.opt.asr_wav = tlist[1]
                        elif self.opt.asr_nogui == 2:
                            synthesizer = self.InitTTS()
                            self.Text2Audio(synthesizer, tlist[1])
    ###############################################################################################
    def Text2Audio(self, in_synthesizer, text):
        TTSFrames = re.split('[.][\s]+|[。][\s]*|[\n][\s]*', text)
        print(TTSFrames)
        for i in range(len(TTSFrames)):
            if TTSFrames[i] == '':
                continue
            while 1:
                result = in_synthesizer.speak_text_async(TTSFrames[i]).get()
                #print(result.properties.get_property(0))
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    break
            self.asr.tts_queue.put(result.audio_data[256:-30])#16, -30
            time.sleep(0.5)
