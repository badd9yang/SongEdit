# Copyright (c) 2025 by badd9yang


__doc__ = """ 
Step. 1. Source Separation and DeEcho; return singing voice, accompany, MFA and MIDI result.
"""

import base64
import codecs
import csv
import glob
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import types
import warnings
from collections import OrderedDict, namedtuple
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from functools import partial
from math import isclose
from multiprocessing import cpu_count
from pathlib import Path
from typing import (Any, Callable, Dict, List, Literal, NamedTuple, Optional,
                    Text, Tuple, Union)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", ".*Tensor Cores.*")

from contextlib import contextmanager

import cn2an
import colorlog
import faster_whisper
import jieba
import librosa
import lightning as pl
import numpy as np
import onnxruntime
import pandas as pd
import parselmouth
import soundfile as sf
import textgrid
import torch
import torch.nn.functional as F
import torchaudio
import tqdm
import yaml
from chardet import detect
from cryptography.exceptions import InvalidSignature
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from einops import pack, rearrange, reduce, repeat, unpack
from librosa.filters import mel
from ml_collections import ConfigDict
from packaging import version
from pyannote.audio import Audio
from pyannote.audio.core.inference import BaseInference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import CACHE_DIR, Model
from pyannote.audio.utils.reproducibility import fix_reproducibility
from pyannote.core.utils.helper import get_class_by_name
from pyannote.database import FileFinder, ProtocolFile
from pyannote.pipeline import Pipeline as _Pipeline
from pydub import AudioSegment
from pypinyin import lazy_pinyin
from rotary_embedding_torch import RotaryEmbedding
from textgrid import TextGrid
from torch import einsum, nn
from torch.utils.data import Dataset
from transformers.pipelines.base import ChunkPipeline
from whisperx.asr import (FasterWhisperPipeline, WhisperModel,
                          find_numeral_symbol_tokens)
from whisperx.audio import (N_SAMPLES, SAMPLE_RATE, load_audio,
                            log_mel_spectrogram)
from whisperx.types import SingleSegment, TranscriptionResult

#########################################################
#                     Logging Set                       #
#########################################################

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

jieba.setLogLevel(logging.ERROR)
onnxruntime.set_default_logger_severity(4)

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

class TqdmToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.last_message = ""
        self.buffer = ""

    def write(self, buf):
        self.buffer += buf
        # 只处理完整的行（包含 \r 或 \n）
        if "\r" in self.buffer or "\n" in self.buffer:
            # 取最后一个有效的进度信息
            lines = self.buffer.replace("\r", "\n").split("\n")
            for line in lines:
                line = line.strip()
                if line and line != self.last_message and "%|" in line:
                    # 清理 ANSI 转义序列
                    clean_line = self._clean_ansi(line)
                    # 检查是否完成（100%）
                    if "100%" in clean_line:
                        # 完成时换行输出
                        print(
                            f"\r\033[92m{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO     - {clean_line}\033[0m"
                        )
                    else:
                        # 进行中时覆盖当前行
                        print(
                            f"\r\033[92m{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO     - {clean_line}\033[0m",
                            end="",
                            flush=True,
                        )
                    self.last_message = line
            self.buffer = ""

    def _clean_ansi(self, text):
        """清理 ANSI 转义序列"""
        import re

        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    def flush(self):
        pass

tqdm_out = TqdmToLogger(logger, level=logging.INFO)

class AdvancedSafeTqdmHandler:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.last_message = ""
        self.buffer = ""
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    
    def get_terminal_width(self):
        """获取终端宽度"""
        try:
            return os.get_terminal_size().columns
        except:
            return 80
    
    def parse_tqdm_line(self, line):
        """解析 tqdm 行的各个组件"""
        parts = {
            'description': '',
            'percentage': '',
            'bar': '',
            'numbers': '',
            'time_info': ''
        }
        
        # 提取百分比
        percent_match = re.search(r'(\d+%)', line)
        if percent_match:
            parts['percentage'] = percent_match.group(1)
        
        # 提取进度条
        bar_match = re.search(r'\|([█▉▊▋▌▍▎▏ ]*)\|', line)
        if bar_match:
            parts['bar'] = bar_match.group(1)
        
        # 提取数字信息 (如 "1000/2000")
        numbers_match = re.search(r'(\d+/\d+)', line)
        if numbers_match:
            parts['numbers'] = numbers_match.group(1)
        
        # 提取时间和速度信息
        time_match = re.search(r'\[([^\]]+)\]', line)
        if time_match:
            parts['time_info'] = time_match.group(1)
        
        # 提取描述（在第一个冒号之前）
        if ':' in line:
            parts['description'] = line.split(':')[0].strip()
        
        return parts
    
    def rebuild_tqdm_line(self, parts, available_width):
        """根据可用宽度重建 tqdm 行"""
        # 必需的组件及其优先级
        essential = []
        
        if parts['percentage']:
            essential.append(parts['percentage'])
        
        if parts['numbers']:
            essential.append(parts['numbers'])
        
        # 计算必需组件的长度
        essential_text = ' '.join(essential)
        essential_len = len(essential_text)
        
        # 为描述预留空间
        desc_len = min(len(parts['description']), available_width // 4) if parts['description'] else 0
        
        # 计算进度条可用的宽度
        separator_len = 4  # " |" + "| "
        remaining_width = available_width - essential_len - desc_len - separator_len - 10  # 10 for spacing
        
        if remaining_width < 10:
            remaining_width = 10
        
        # 构建新的进度条
        bar_char = '█'
        empty_char = ' '
        
        if parts['percentage']:
            try:
                percent_val = int(parts['percentage'].rstrip('%'))
                filled_len = int(remaining_width * percent_val / 100)
                bar = bar_char * filled_len + empty_char * (remaining_width - filled_len)
            except:
                bar = parts['bar'][:remaining_width] if parts['bar'] else empty_char * remaining_width
        else:
            bar = parts['bar'][:remaining_width] if parts['bar'] else empty_char * remaining_width
        
        # 组装最终的行
        result_parts = []
        
        if parts['description'] and desc_len > 0:
            result_parts.append(parts['description'][:desc_len])
        
        if bar:
            result_parts.append(f"|{bar}|")
        
        if essential_text:
            result_parts.append(essential_text)
        
        return ' '.join(result_parts)
    
    def format_progress_line(self, clean_line, timestamp):
        """格式化进度条行"""
        terminal_width = self.get_terminal_width()
        prefix = f"{timestamp} - INFO     - "
        prefix_len = len(prefix)
        color_codes_len = len("\033[92m\033[0m")
        available_width = terminal_width - prefix_len - color_codes_len - 2
        
        if available_width <= 0:
            available_width = 50
        
        # 解析并重建 tqdm 行
        parts = self.parse_tqdm_line(clean_line)
        formatted_line = self.rebuild_tqdm_line(parts, available_width)
        
        return formatted_line
    
    def write(self, buf):
        if "%|" not in buf:
            self.original_stdout.write(buf)
            return
        
        self.buffer += buf
        if "\r" in self.buffer or "\n" in self.buffer:
            lines = self.buffer.replace("\r", "\n").split("\n")
            for line in lines:
                line = line.strip()
                if line and line != self.last_message and "%|" in line:
                    clean_line = self.ansi_escape.sub("", line)
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    formatted_line = self.format_progress_line(clean_line, timestamp)
                    
                    if "100%" in clean_line:
                        self.original_stdout.write(f"\r\033[92m{timestamp} - INFO     - {formatted_line}\033[0m\n")
                    else:
                        self.original_stdout.write(f"\r\033[92m{timestamp} - INFO     - {formatted_line}\033[0m")
                    self.original_stdout.flush()
                    
                    self.last_message = line
            self.buffer = ""
    
    def flush(self):
        self.original_stdout.flush()

@contextmanager
def tqdm_logger():
    """used for lightning predict method"""
    original_stdout = sys.stdout
    try:
        sys.stdout =  AdvancedSafeTqdmHandler(original_stdout)
        yield
    finally:
        sys.stdout = original_stdout


#########################################################
#                     Constants                         #
#########################################################

VAD_THRESHOLD = 20
SAMPLING_RATE = 16000

uvr5_band_v2 = {
    "bins": 672,
    "unstable_bins": 8,
    "reduction_bins": 637,
    "band": {
        1: {
            "sr": 7350,
            "hl": 80,
            "n_fft": 640,
            "crop_start": 0,
            "crop_stop": 85,
            "lpf_start": 25,
            "lpf_stop": 53,
            "res_type": "polyphase",
        },
        2: {
            "sr": 7350,
            "hl": 80,
            "n_fft": 320,
            "crop_start": 4,
            "crop_stop": 87,
            "hpf_start": 25,
            "hpf_stop": 12,
            "lpf_start": 31,
            "lpf_stop": 62,
            "res_type": "polyphase",
        },
        3: {
            "sr": 14700,
            "hl": 160,
            "n_fft": 512,
            "crop_start": 17,
            "crop_stop": 216,
            "hpf_start": 48,
            "hpf_stop": 24,
            "lpf_start": 139,
            "lpf_stop": 210,
            "res_type": "polyphase",
        },
        4: {
            "sr": 44100,
            "hl": 480,
            "n_fft": 960,
            "crop_start": 78,
            "crop_stop": 383,
            "hpf_start": 130,
            "hpf_stop": 86,
            "res_type": "kaiser_fast",
        },
    },
    "sr": 44100,
    "pre_filter_start": 668,
    "pre_filter_stop": 672,
}

uvr5_band_v3 = {
    "bins": 672,
    "unstable_bins": 8,
    "reduction_bins": 530,
    "band": {
        1: {
            "sr": 7350,
            "hl": 80,
            "n_fft": 640,
            "crop_start": 0,
            "crop_stop": 85,
            "lpf_start": 25,
            "lpf_stop": 53,
            "res_type": "polyphase",
        },
        2: {
            "sr": 7350,
            "hl": 80,
            "n_fft": 320,
            "crop_start": 4,
            "crop_stop": 87,
            "hpf_start": 25,
            "hpf_stop": 12,
            "lpf_start": 31,
            "lpf_stop": 62,
            "res_type": "polyphase",
        },
        3: {
            "sr": 14700,
            "hl": 160,
            "n_fft": 512,
            "crop_start": 17,
            "crop_stop": 216,
            "hpf_start": 48,
            "hpf_stop": 24,
            "lpf_start": 139,
            "lpf_stop": 210,
            "res_type": "polyphase",
        },
        4: {
            "sr": 44100,
            "hl": 480,
            "n_fft": 960,
            "crop_start": 78,
            "crop_stop": 383,
            "hpf_start": 130,
            "hpf_stop": 86,
            "res_type": "kaiser_fast",
        },
    },
    "sr": 44100,
    "pre_filter_start": 668,
    "pre_filter_stop": 672,
}

mel_band_roformer = {
    "model": {
        "dim": 384,
        "depth": 6,
        "stereo": True,
        "num_stems": 1,
        "time_transformer_depth": 1,
        "freq_transformer_depth": 1,
        "num_bands": 60,
        "dim_head": 64,
        "heads": 8,
        "attn_dropout": 0,
        "ff_dropout": 0,
        "flash_attn": True,
        "dim_freqs_in": 1025,
        "sample_rate": 44100,
        "stft_n_fft": 2048,
        "stft_hop_length": 441,
        "stft_win_length": 2048,
        "stft_normalized": False,
        "mask_estimator_depth": 2,
        "multi_stft_resolution_loss_weight": 1.0,
        "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
        "multi_stft_hop_size": 147,
        "multi_stft_normalized": False,
    },
    "training": {"instruments": ["vocals", "other"], "target_instrument": "vocals"},
    "inference": {"num_overlap": 2, "chunk_size": 352800},
}

dia_pipeline_config = {
    "version": "3.1.0",
    "pipeline": {
        "name": "pyannote.audio.pipelines.SpeakerDiarization",
        "params": {
            "clustering": "AgglomerativeClustering",
            "embedding": "checkpoints/pyannote/resnet34-LM.bin",
            "embedding_batch_size": 32,
            "embedding_exclude_overlap": True,
            "segmentation": "checkpoints/pyannote/pytorch_model.bin",
            "segmentation_batch_size": 32,
        },
    },
    "params": {
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 12,
            "threshold": 0.7045654963945799,
        },
        "segmentation": {"min_duration_off": 0.0},
    },
}

# TODO set DS_KEY to dataclass
DS_KEYS = [
    "offset",
    "offset_end",
    "text",
    "word_seq",
    "word2ph",
    "word_dur",
    "ph_seq",
    "ph_dur",
    "ph_num",
    "note_seq",
    "note_dur",
    "note_slur",
    "f0_seq",
    "f0_timestep",
]

dictionary = """a	a
ai	ai
an	an
ang	ang
ao	ao
ba	b a
bai	b ai
ban	b an
bang	b ang
bao	b ao
be	b e
bei	b ei
ben	b en
beng	b eng
ber	b er
bi	b i
bia	b ia
bian	b ian
biang	b iang
biao	b iao
bie	b ie
bin	b in
bing	b ing
biong	b iong
biu	b iu
bo	b o
bong	b ong
bou	b ou
bu	b u
bua	b ua
buai	b uai
buan	b uan
buang	b uang
bui	b ui
bun	b un
bv	b v
bve	b ve
ca	c a
cai	c ai
can	c an
cang	c ang
cao	c ao
ce	c e
cei	c ei
cen	c en
ceng	c eng
cer	c er
cha	ch a
chai	ch ai
chan	ch an
chang	ch ang
chao	ch ao
che	ch e
chei	ch ei
chen	ch en
cheng	ch eng
cher	ch er
chi	ch ir
chong	ch ong
chou	ch ou
chu	ch u
chua	ch ua
chuai	ch uai
chuan	ch uan
chuang	ch uang
chui	ch ui
chun	ch un
chuo	ch uo
chv	ch v
chyi	ch i
ci	c i0
cong	c ong
cou	c ou
cu	c u
cua	c ua
cuai	c uai
cuan	c uan
cuang	c uang
cui	c ui
cun	c un
cuo	c uo
cv	c v
cyi	c i
da	d a
dai	d ai
dan	d an
dang	d ang
dao	d ao
de	d e
dei	d ei
den	d en
deng	d eng
der	d er
di	d i
dia	d ia
dian	d ian
diang	d iang
diao	d iao
die	d ie
din	d in
ding	d ing
diong	d iong
diu	d iu
dong	d ong
dou	d ou
du	d u
dua	d ua
duai	d uai
duan	d uan
duang	d uang
dui	d ui
dun	d un
duo	d uo
dv	d v
dve	d ve
e	e
ei	ei
en	en
eng	eng
er	er
fa	f a
fai	f ai
fan	f an
fang	f ang
fao	f ao
fe	f e
fei	f ei
fen	f en
feng	f eng
fer	f er
fi	f i
fia	f ia
fian	f ian
fiang	f iang
fiao	f iao
fie	f ie
fin	f in
fing	f ing
fiong	f iong
fiu	f iu
fo	f o
fong	f ong
fou	f ou
fu	f u
fua	f ua
fuai	f uai
fuan	f uan
fuang	f uang
fui	f ui
fun	f un
fv	f v
fve	f ve
ga	g a
gai	g ai
gan	g an
gang	g ang
gao	g ao
ge	g e
gei	g ei
gen	g en
geng	g eng
ger	g er
gi	g i
gia	g ia
gian	g ian
giang	g iang
giao	g iao
gie	g ie
gin	g in
ging	g ing
giong	g iong
giu	g iu
gong	g ong
gou	g ou
gu	g u
gua	g ua
guai	g uai
guan	g uan
guang	g uang
gui	g ui
gun	g un
guo	g uo
gv	g v
gve	g ve
ha	h a
hai	h ai
han	h an
hang	h ang
hao	h ao
he	h e
hei	h ei
hen	h en
heng	h eng
her	h er
hi	h i
hia	h ia
hian	h ian
hiang	h iang
hiao	h iao
hie	h ie
hin	h in
hing	h ing
hiong	h iong
hiu	h iu
hong	h ong
hou	h ou
hu	h u
hua	h ua
huai	h uai
huan	h uan
huang	h uang
hui	h ui
hun	h un
huo	h uo
hv	h v
hve	h ve
ji	j i
jia	j ia
jian	j ian
jiang	j iang
jiao	j iao
jie	j ie
jin	j in
jing	j ing
jiong	j iong
jiu	j iu
ju	j v
juan	j van
jue	j ve
jun	j vn
ka	k a
kai	k ai
kan	k an
kang	k ang
kao	k ao
ke	k e
kei	k ei
ken	k en
keng	k eng
ker	k er
ki	k i
kia	k ia
kian	k ian
kiang	k iang
kiao	k iao
kie	k ie
kin	k in
king	k ing
kiong	k iong
kiu	k iu
kong	k ong
kou	k ou
ku	k u
kua	k ua
kuai	k uai
kuan	k uan
kuang	k uang
kui	k ui
kun	k un
kuo	k uo
kv	k v
kve	k ve
la	l a
lai	l ai
lan	l an
lang	l ang
lao	l ao
le	l e
lei	l ei
len	l en
leng	l eng
ler	l er
li	l i
lia	l ia
lian	l ian
liang	l iang
liao	l iao
lie	l ie
lin	l in
ling	l ing
liong	l iong
liu	l iu
lo	l o
long	l ong
lou	l ou
lu	l u
lua	l ua
luai	l uai
luan	l uan
luang	l uang
lui	l ui
lun	l un
luo	l uo
lv	l v
lve	l ve
ma	m a
mai	m ai
man	m an
mang	m ang
mao	m ao
me	m e
mei	m ei
men	m en
meng	m eng
mer	m er
mi	m i
mia	m ia
mian	m ian
miang	m iang
miao	m iao
mie	m ie
min	m in
ming	m ing
miong	m iong
miu	m iu
mo	m o
mong	m ong
mou	m ou
mu	m u
mua	m ua
muai	m uai
muan	m uan
muang	m uang
mui	m ui
mun	m un
mv	m v
mve	m ve
na	n a
nai	n ai
nan	n an
nang	n ang
nao	n ao
ne	n e
nei	n ei
nen	n en
neng	n eng
ner	n er
ni	n i
nia	n ia
nian	n ian
niang	n iang
niao	n iao
nie	n ie
nin	n in
ning	n ing
niong	n iong
niu	n iu
nong	n ong
nou	n ou
nu	n u
nua	n ua
nuai	n uai
nuan	n uan
nuang	n uang
nui	n ui
nun	n un
nuo	n uo
nv	n v
nve	n ve
o	o
ong	ong
ou	ou
pa	p a
pai	p ai
pan	p an
pang	p ang
pao	p ao
pe	p e
pei	p ei
pen	p en
peng	p eng
per	p er
pi	p i
pia	p ia
pian	p ian
piang	p iang
piao	p iao
pie	p ie
pin	p in
ping	p ing
piong	p iong
piu	p iu
po	p o
pong	p ong
pou	p ou
pu	p u
pua	p ua
puai	p uai
puan	p uan
puang	p uang
pui	p ui
pun	p un
pv	p v
pve	p ve
qi	q i
qia	q ia
qian	q ian
qiang	q iang
qiao	q iao
qie	q ie
qin	q in
qing	q ing
qiong	q iong
qiu	q iu
qu	q v
quan	q van
que	q ve
qun	q vn
ra	r a
rai	r ai
ran	r an
rang	r ang
rao	r ao
re	r e
rei	r ei
ren	r en
reng	r eng
rer	r er
ri	r ir
rong	r ong
rou	r ou
ru	r u
rua	r ua
ruai	r uai
ruan	r uan
ruang	r uang
rui	r ui
run	r un
ruo	r uo
rv	r v
ryi	r i
sa	s a
sai	s ai
san	s an
sang	s ang
sao	s ao
se	s e
sei	s ei
sen	s en
seng	s eng
ser	s er
sha	sh a
shai	sh ai
shan	sh an
shang	sh ang
shao	sh ao
she	sh e
shei	sh ei
shen	sh en
sheng	sh eng
sher	sh er
shi	sh ir
shong	sh ong
shou	sh ou
shu	sh u
shua	sh ua
shuai	sh uai
shuan	sh uan
shuang	sh uang
shui	sh ui
shun	sh un
shuo	sh uo
shv	sh v
shyi	sh i
si	s i0
song	s ong
sou	s ou
su	s u
sua	s ua
suai	s uai
suan	s uan
suang	s uang
sui	s ui
sun	s un
suo	s uo
sv	s v
syi	s i
ta	t a
tai	t ai
tan	t an
tang	t ang
tao	t ao
te	t e
tei	t ei
ten	t en
teng	t eng
ter	t er
ti	t i
tia	t ia
tian	t ian
tiang	t iang
tiao	t iao
tie	t ie
tin	t in
ting	t ing
tiong	t iong
tong	t ong
tou	t ou
tu	t u
tua	t ua
tuai	t uai
tuan	t uan
tuang	t uang
tui	t ui
tun	t un
tuo	t uo
tv	t v
tve	t ve
wa	w a
wai	w ai
wan	w an
wang	w ang
wao	w ao
we	w e
wei	w ei
wen	w en
weng	w eng
wer	w er
wi	w i
wo	w o
wong	w ong
wou	w ou
wu	w u
xi	x i
xia	x ia
xian	x ian
xiang	x iang
xiao	x iao
xie	x ie
xin	x in
xing	x ing
xiong	x iong
xiu	x iu
xu	x v
xuan	x van
xue	x ve
xun	x vn
ya	y a
yai	y ai
yan	y En
yang	y ang
yao	y ao
ye	y E
yei	y ei
yi	y i
yin	y in
ying	y ing
yo	y o
yong	y ong
you	y ou
yu	y v
yuan	y van
yue	y ve
yun	y vn
ywu	y u
za	z a
zai	z ai
zan	z an
zang	z ang
zao	z ao
ze	z e
zei	z ei
zen	z en
zeng	z eng
zer	z er
zha	zh a
zhai	zh ai
zhan	zh an
zhang	zh ang
zhao	zh ao
zhe	zh e
zhei	zh ei
zhen	zh en
zheng	zh eng
zher	zh er
zhi	zh ir
zhong	zh ong
zhou	zh ou
zhu	zh u
zhua	zh ua
zhuai	zh uai
zhuan	zh uan
zhuang	zh uang
zhui	zh ui
zhun	zh un
zhuo	zh uo
zhv	zh v
zhyi	zh i
zi	z i0
zong	z ong
zou	z ou
zu	z u
zua	z ua
zuai	z uai
zuan	z uan
zuang	z uang
zui	z ui
zun	z un
zuo	z uo
zv	z v
zyi	z i
AP	AP
SP	SP"""


#########################################################
#                           ASR                         #
#########################################################


class VadFreeFasterWhisperPipeline(FasterWhisperPipeline):
    """
    FasterWhisperModel without VAD
    """

    def __init__(
        self,
        model,
        options: NamedTuple,
        tokenizer=None,
        device: Union[int, str, "torch.device"] = -1,
        framework="pt",
        language: Optional[str] = None,
        suppress_numerals: bool = False,
        **kwargs,
    ):
        """
        Initialize the VadFreeFasterWhisperPipeline.

        Args:
            model: The Whisper model instance.
            options: Transcription options.
            tokenizer: The tokenizer instance.
            device: Device to run the model on.
            framework: The framework to use ('pt' for PyTorch).
            language: The language for transcription.
            suppress_numerals: Whether to suppress numeral tokens.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(
            model=model,
            vad=None,
            vad_params={},
            options=options,
            tokenizer=tokenizer,
            device=device,
            framework=framework,
            language=language,
            suppress_numerals=suppress_numerals,
            **kwargs,
        )

    def detect_language(self, audio: np.ndarray):
        """
        Detect the language of the audio.

        Args:
            audio (np.ndarray): The input audio signal.

        Returns:
            tuple: Detected language and its probability.
        """
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        if audio.shape[0] > N_SAMPLES:
            # Randomly sample N_SAMPLES from the audio array
            start_index = np.random.randint(0, audio.shape[0] - N_SAMPLES)
            audio_sample = audio[start_index : start_index + N_SAMPLES]
        else:
            audio_sample = audio[:N_SAMPLES]
        padding = 0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0]
        segment = log_mel_spectrogram(
            audio_sample,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=padding,
        )
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        return language, language_probability

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def pass_call(self, inputs, *args, num_workers=None, batch_size=None, **kwargs):

        if num_workers is None:
            if self._num_workers is None:
                num_workers = 0
            else:
                num_workers = self._num_workers
        if batch_size is None:
            if self._batch_size is None:
                batch_size = 1
            else:
                batch_size = self._batch_size

        (
            preprocess_params,
            forward_params,
            postprocess_params,
        ) = self._sanitize_parameters(**kwargs)

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        self.call_count += 1
        if (
            self.call_count > 10
            and self.framework == "pt"
            and self.device.type == "cuda"
        ):
            raise Warning(
                "You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a"
                " dataset",
            )

        is_dataset = Dataset is not None and isinstance(inputs, Dataset)
        is_generator = isinstance(inputs, types.GeneratorType)
        is_list = isinstance(inputs, list)

        is_iterable = is_dataset or is_generator or is_list
        # TODO make the get_iterator work also for `tf` (and `flax`).
        can_use_iterator = self.framework == "pt" and (
            is_dataset or is_generator or is_list
        )

        if is_list:
            if can_use_iterator:
                final_iterator = self.get_iterator(
                    inputs,
                    num_workers,
                    batch_size,
                    preprocess_params,
                    forward_params,
                    postprocess_params,
                )
                outputs = list(final_iterator)
                return outputs
            else:
                return self.run_multi(
                    inputs, preprocess_params, forward_params, postprocess_params
                )
        elif can_use_iterator:
            return self.get_iterator(
                inputs,
                num_workers,
                batch_size,
                preprocess_params,
                forward_params,
                postprocess_params,
            )
        elif is_iterable:
            return self.iterate(
                inputs, preprocess_params, forward_params, postprocess_params
            )
        elif self.framework == "pt" and isinstance(self, ChunkPipeline):
            return next(
                iter(
                    self.get_iterator(
                        [inputs],
                        num_workers,
                        batch_size,
                        preprocess_params,
                        forward_params,
                        postprocess_params,
                    )
                )
            )
        else:
            return self.run_single(
                inputs, preprocess_params, forward_params, postprocess_params
            )

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        vad_segments: List[dict],
        batch_size=None,
        num_workers=0,
        language=None,
        task=None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
    ) -> TranscriptionResult:
        """
        Transcribe the audio into text.

        Args:
            audio (Union[str, np.ndarray]): The input audio signal or path to audio file.
            vad_segments (List[dict]): List of VAD segments.
            batch_size (int, optional): Batch size for transcription. Defaults to None.
            num_workers (int, optional): Number of workers for loading data. Defaults to 0.
            language (str, optional): Language for transcription. Defaults to None.
            task (str, optional): Task type ('transcribe' or 'translate'). Defaults to None.
            chunk_size (int, optional): Size of chunks for processing. Defaults to 30.
            print_progress (bool, optional): Whether to print progress. Defaults to False.
            combined_progress (bool, optional): Whether to combine progress. Defaults to False.

        Returns:
            TranscriptionResult: The transcription result containing segments and language.
        """
        if isinstance(audio, str):
            audio = load_audio(audio)

        # def data(audio, segments):
        #     for seg in segments:
        #         f1 = int(seg["start"] * SAMPLE_RATE)
        #         f2 = int(seg["end"] * SAMPLE_RATE)
        #         yield {"inputs": audio[f1:f2]}
        def data(audio, segments):
            result = []
            for seg in segments:
                f1 = int(seg["start"] * SAMPLE_RATE)
                f2 = int(seg["end"] * SAMPLE_RATE)
                result.append({"inputs": audio[f1:f2]})
            return result  # 返回列表而不是生成器

        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = faster_whisper.tokenizer.Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=language,
            )
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = faster_whisper.tokenizer.Tokenizer(
                    self.model.hf_tokenizer,
                    self.model.model.is_multilingual,
                    task=task,
                    language=language,
                )

        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = self.options._replace(suppress_tokens=new_suppressed_tokens)

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        progress = tqdm.tqdm(total=total_segments, desc="Transcribing")
        for idx, out in enumerate(
            self.pass_call(
                data(audio, vad_segments),
                batch_size=batch_size,
                num_workers=num_workers,
            )
        ):
            if print_progress:
                progress.update(1)
            text = out["text"]
            if batch_size in [0, 1, None]:
                text = text[0]
            segments.append(
                {
                    "text": text,
                    "start": round(vad_segments[idx]["start"], 3),
                    "end": round(vad_segments[idx]["end"], 3),
                    "speaker": vad_segments[idx].get("speaker", None),
                }
            )

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

        # revert suppressed tokens if suppress_numerals is enabled
        if self.suppress_numerals:
            self.options = self.options._replace(
                suppress_tokens=previous_suppress_tokens
            )

        return {"segments": segments, "language": language}


def load_asr_model(
    whisper_arch: str,
    device: str,
    device_index: int = 0,
    compute_type: str = "float16",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    model: Optional[WhisperModel] = None,
    task: str = "transcribe",
    download_root: Optional[str] = None,
    threads: int = 4,
) -> VadFreeFasterWhisperPipeline:
    """
    Load a Whisper model for inference.

    Args:
        whisper_arch (str): The name of the Whisper model to load.
        device (str): The device to load the model on.
        device_index (int, optional): The device index. Defaults to 0.
        compute_type (str, optional): The compute type to use for the model. Defaults to "float16".
        asr_options (Optional[dict], optional): Options for ASR. Defaults to None.
        language (Optional[str], optional): The language of the model. Defaults to None.
        vad_model: The VAD model instance. Defaults to None.
        vad_options: Options for VAD. Defaults to None.
        model (Optional[WhisperModel], optional): The WhisperModel instance to use. Defaults to None.
        task (str, optional): The task type ('transcribe' or 'translate'). Defaults to "transcribe".
        download_root (Optional[str], optional): The root directory to download the model to. Defaults to None.
        threads (int, optional): The number of CPU threads to use per worker. Defaults to 4.

    Returns:
        VadFreeFasterWhisperPipeline: The loaded Whisper pipeline.

    Raises:
        ValueError: If the whisper architecture is not recognized.
    """

    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or WhisperModel(
        whisper_arch,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        download_root=download_root,
        cpu_threads=threads,
    )
    if language is not None:
        tokenizer = faster_whisper.tokenizer.Tokenizer(
            model.hf_tokenizer,
            model.model.is_multilingual,
            task=task,
            language=language,
        )
    else:
        # print(
        #     "No language specified, language will be detected for each audio file (increases inference time)."
        # )
        tokenizer = None

    default_asr_options = {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = faster_whisper.transcribe.TranscriptionOptions(
        **default_asr_options
    )

    return VadFreeFasterWhisperPipeline(
        model=model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
    )


#########################################################
#                           VAD                         #
#########################################################
class SileroVadOnnxWrapper:
    def __init__(self, path, force_onnx_cpu=False):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if (
            force_onnx_cpu
            and "CPUExecutionProvider" in onnxruntime.get_available_providers()
        ):
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"], sess_options=opts
            )
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.reset_states()
        self.sample_rates = [8000, 16000]

    def _validate_input(self, x, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(
                f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)"
            )
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        self._state = torch.zeros((2, batch_size, 128)).float()
        self._context = torch.zeros(0)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):

        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = torch.zeros(batch_size, context_size)

        x = torch.cat([self._context, x], dim=1)
        if sr in [8000, 16000]:
            ort_inputs = {
                "input": x.numpy(),
                "state": self._state.numpy(),
                "sr": np.array(sr, dtype="int64"),
            }
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs
            self._state = torch.from_numpy(state)
        else:
            raise ValueError()

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        out = torch.from_numpy(out)
        return out

    def audio_forward(self, x, sr: int):
        outs = []
        x, sr = self._validate_input(x, sr)
        self.reset_states()
        num_samples = 512 if sr == 16000 else 256

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = torch.nn.functional.pad(x, (0, pad_num), "constant", value=0.0)

        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i : i + num_samples]
            out_chunk = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)

        stacked = torch.cat(outs, dim=1)
        return stacked.cpu()

class SileroVAD:
    """
    Voice Activity Detection (VAD) using Silero-VAD.
    """

    def __init__(
        self, model_path: str = "", device=torch.device("cpu"),
    ):
        """
        Initialize the VAD object.

        Args:
            local (bool, optional): Whether to load the model locally. Defaults to False.
            model (str, optional): The VAD model name to load. Defaults to "silero_vad".
            device (torch.device, optional): The device to run the model on. Defaults to 'cpu'.

        Returns:
            None

        Raises:
            RuntimeError: If loading the model fails.
        """
        try:
            self.device = device
            torch.set_num_threads(1)
            self.vad_model = SileroVadOnnxWrapper(model_path, True)

        except Exception as e:
            raise RuntimeError(f"Failed to load VAD model: {e}")

    @torch.no_grad()
    def get_speech_timestamps(
        self,
        audio: torch.Tensor,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float("inf"),
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        return_seconds: bool = False,
        visualize_probs: bool = False,
        progress_tracking_callback: Callable[[float], None] = None,
        window_size_samples: int = 512,
    ):

        """
        This method is used for splitting long audios into speech chunks using silero VAD

        Parameters
        ----------
        audio: torch.Tensor, one dimensional
            One dimensional float torch.Tensor, other types are casted to torch if possible

        model: preloaded .jit/.onnx silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates

        min_speech_duration_ms: int (default - 250 milliseconds)
            Final speech chunks shorter min_speech_duration_ms are thrown out

        max_speech_duration_s: int (default -  inf)
            Maximum duration of speech chunks in seconds
            Chunks longer than max_speech_duration_s will be split at the timestamp of the last silence that lasts more than 100ms (if any), to prevent agressive cutting.
            Otherwise, they will be split aggressively just before max_speech_duration_s.

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)

        visualize_probs: bool (default - False)
            whether draw prob hist or not

        progress_tracking_callback: Callable[[float], None] (default - None)
            callback function taking progress in percents as an argument

        window_size_samples: int (default - 512 samples)
            !!! DEPRECATED, DOES NOTHING !!!

        Returns
        ----------
        speeches: list of dicts
            list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
        """

        if not torch.is_tensor(audio):
            try:
                audio = torch.Tensor(audio)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        if len(audio.shape) > 1:
            for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
                audio = audio.squeeze(0)
            if len(audio.shape) > 1:
                raise ValueError(
                    "More than one dimension in audio. Are you trying to process audio with 2 channels?"
                )

        if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
            step = sampling_rate // 16000
            sampling_rate = 16000
            audio = audio[::step]
            warnings.warn(
                "Sampling rate is a multiply of 16000, casting to 16000 manually!"
            )
        else:
            step = 1

        if sampling_rate not in [8000, 16000]:
            raise ValueError(
                "Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates"
            )

        window_size_samples = 512 if sampling_rate == 16000 else 256

        model.reset_states()
        min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
        speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        max_speech_samples = (
            sampling_rate * max_speech_duration_s
            - window_size_samples
            - 2 * speech_pad_samples
        )
        min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

        audio_length_samples = len(audio)

        speech_probs = []
        for current_start_sample in range(0, audio_length_samples, window_size_samples):
            chunk = audio[
                current_start_sample : current_start_sample + window_size_samples
            ]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, int(window_size_samples - len(chunk)))
                )
            speech_prob = model(chunk, sampling_rate).item()
            speech_probs.append(speech_prob)
            # caculate progress and seng it to callback function
            progress = current_start_sample + window_size_samples
            if progress > audio_length_samples:
                progress = audio_length_samples
            progress_percent = (progress / audio_length_samples) * 100
            if progress_tracking_callback:
                progress_tracking_callback(progress_percent)

        triggered = False
        speeches = []
        current_speech = {}
        neg_threshold = threshold - 0.15
        temp_end = 0  # to save potential segment end (and tolerate some silence)
        prev_end = (
            next_start
        ) = 0  # to save potential segment limits in case of maximum segment size reached

        for i, speech_prob in enumerate(speech_probs):
            if (speech_prob >= threshold) and temp_end:
                temp_end = 0
                if next_start < prev_end:
                    next_start = window_size_samples * i

            if (speech_prob >= threshold) and not triggered:
                triggered = True
                current_speech["start"] = window_size_samples * i
                continue

            if (
                triggered
                and (window_size_samples * i) - current_speech["start"]
                > max_speech_samples
            ):
                if prev_end:
                    current_speech["end"] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if (
                        next_start < prev_end
                    ):  # previously reached silence (< neg_thres) and is still not speech (< thres)
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = next_start = temp_end = 0
                else:
                    current_speech["end"] = window_size_samples * i
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = window_size_samples * i
                if (
                    (window_size_samples * i) - temp_end
                ) > min_silence_samples_at_max_speech:  # condition to avoid cutting in very short silence
                    prev_end = temp_end
                if (window_size_samples * i) - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech["end"] = temp_end
                    if (
                        current_speech["end"] - current_speech["start"]
                    ) > min_speech_samples:
                        speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

        if (
            current_speech
            and (audio_length_samples - current_speech["start"]) > min_speech_samples
        ):
            current_speech["end"] = audio_length_samples
            speeches.append(current_speech)

        for i, speech in enumerate(speeches):
            if i == 0:
                speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]["start"] - speech["end"]
                if silence_duration < 2 * speech_pad_samples:
                    speech["end"] += int(silence_duration // 2)
                    speeches[i + 1]["start"] = int(
                        max(0, speeches[i + 1]["start"] - silence_duration // 2)
                    )
                else:
                    speech["end"] = int(
                        min(audio_length_samples, speech["end"] + speech_pad_samples)
                    )
                    speeches[i + 1]["start"] = int(
                        max(0, speeches[i + 1]["start"] - speech_pad_samples)
                    )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )

        if return_seconds:
            for speech_dict in speeches:
                speech_dict["start"] = round(speech_dict["start"] / sampling_rate, 1)
                speech_dict["end"] = round(speech_dict["end"] / sampling_rate, 1)
        elif step > 1:
            for speech_dict in speeches:
                speech_dict["start"] *= step
                speech_dict["end"] *= step

        if visualize_probs:
            self.make_visualization(speech_probs, window_size_samples / sampling_rate)

        return speeches

    def make_visualization(probs, step):
        pd.DataFrame(
            {"probs": probs}, index=[x * step for x in range(len(probs))]
        ).plot(
            figsize=(16, 8),
            kind="area",
            ylim=[0, 1.05],
            xlim=[0, len(probs) * step],
            xlabel="seconds",
            ylabel="speech probability",
            colormap="tab20",
        )

    def segment_speech(self, audio_segment, start_time, end_time, sampling_rate):
        """
        Segment speech from an audio segment and return a list of timestamps.

        Args:
            audio_segment (np.ndarray): The audio segment to be segmented.
            start_time (int): The start time of the audio segment in frames.
            end_time (int): The end time of the audio segment in frames.
            sampling_rate (int): The sampling rate of the audio segment.

        Returns:
            list: A list of timestamps, each containing the start and end times of speech segments in frames.

        Raises:
            ValueError: If the audio segment is invalid.
        """
        if audio_segment is None or not isinstance(audio_segment, (np.ndarray, list)):
            raise ValueError("Invalid audio segment")

        speech_timestamps = self.get_speech_timestamps(
            audio_segment, self.vad_model, sampling_rate=sampling_rate, threshold=0.55
        )

        adjusted_timestamps = [
            (ts["start"] + start_time, ts["end"] + start_time)
            for ts in speech_timestamps
        ]
        if not adjusted_timestamps:
            return []

        intervals = [
            end[0] - start[1]
            for start, end in zip(adjusted_timestamps[:-1], adjusted_timestamps[1:])
        ]

        segments = []

        def split_timestamps(start_index, end_index):
            if (
                start_index == end_index
                or adjusted_timestamps[end_index][1]
                - adjusted_timestamps[start_index][0]
                < 20 * sampling_rate
            ):
                segments.append([start_index, end_index])
            else:
                if not intervals[start_index:end_index]:
                    return
                max_interval_index = intervals[start_index:end_index].index(
                    max(intervals[start_index:end_index])
                )
                split_index = start_index + max_interval_index
                split_timestamps(start_index, split_index)
                split_timestamps(split_index + 1, end_index)

        split_timestamps(0, len(adjusted_timestamps) - 1)

        merged_timestamps = [
            [adjusted_timestamps[start][0], adjusted_timestamps[end][1]]
            for start, end in segments
        ]
        return merged_timestamps

    def vad(self, speakerdia, audio):
        """
        Process the audio based on the given speaker diarization dataframe.

        Args:
            speakerdia (pd.DataFrame): The diarization dataframe containing start, end, and speaker info.
            audio (dict): A dictionary containing the audio waveform and sample rate.

        Returns:
            list: A list of dictionaries containing processed audio segments with start, end, and speaker.
        """
        sampling_rate = audio["sample_rate"]
        audio_data = audio["waveform"]

        out = []
        last_end = 0
        speakers_seen = set()
        count_id = 0

        for index, row in speakerdia.iterrows():
            start = float(row["start"])
            end = float(row["end"])

            if end <= last_end:
                continue
            last_end = end

            start_frame = int(start * sampling_rate)
            end_frame = int(end * sampling_rate)

            if row["speaker"] not in speakers_seen:
                speakers_seen.add(row["speaker"])

            if end - start <= VAD_THRESHOLD:
                out.append(
                    {
                        "index": str(count_id).zfill(5),
                        "start": start,  # in seconds
                        "end": end,
                        "speaker": row["speaker"],  # same for all
                    }
                )
                count_id += 1

                continue

            temp_audio = audio_data[start_frame:end_frame]

            # resample from 24k to 16k
            temp_audio_resampled = librosa.resample(
                temp_audio, orig_sr=sampling_rate, target_sr=SAMPLING_RATE
            )

            for start_frame_sub, end_frame_sub in self.segment_speech(
                temp_audio_resampled,
                int(start * SAMPLING_RATE),
                int(end * SAMPLING_RATE),
                SAMPLING_RATE,
            ):
                out.append(
                    {
                        "index": str(count_id).zfill(5),
                        "start": start_frame_sub / SAMPLING_RATE,  # in seconds
                        "end": end_frame_sub / SAMPLING_RATE,
                        "speaker": row["speaker"],  # same for all
                    }
                )

                count_id += 1

        return out


#########################################################
#                    Dialog Pipeline                    #
#########################################################

class DiaPipeline(_Pipeline):
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[Text, Path],
        hparams_file: Union[Text, Path] = None,
        use_auth_token: Union[Text, None] = None,
        cache_dir: Union[Path, Text] = CACHE_DIR,
    ) -> "DiaPipeline":
        """Load pretrained pipeline

        Parameters
        ----------
        checkpoint_path : Path or str
            Path to pipeline checkpoint, or a remote URL,
            or a pipeline identifier from the huggingface.co model hub.
        hparams_file: Path or str, optional
        use_auth_token : str, optional
            When loading a private huggingface.co pipeline, set `use_auth_token`
            to True or to a string containing your hugginface.co authentication
            token that can be obtained by running `huggingface-cli login`
        cache_dir: Path or str, optional
            Path to model cache directory. Defauorch/pyannote" when unset.
        """

        checkpoint_path = str(checkpoint_path)
        global dia_pipeline_config

        config = dia_pipeline_config
        dia_pipeline_config["pipeline"]["params"]["embedding"] = os.path.join(
            checkpoint_path, "resnet34-LM.bin"
        )
        dia_pipeline_config["pipeline"]["params"]["segmentation"] = os.path.join(
            checkpoint_path, "pytorch_model.bin"
        )
        # initialize pipeline
        pipeline_name = config["pipeline"]["name"]
        Klass = get_class_by_name(
            pipeline_name, default_module_name="pyannote.pipeline.blocks"
        )
        params = config["pipeline"].get("params", {})
        params.setdefault("use_auth_token", use_auth_token)
        pipeline = Klass(**params)

        # freeze  parameters
        if "freeze" in config:
            params = config["freeze"]
            pipeline.freeze(params)

        if "params" in config:
            pipeline.instantiate(config["params"])

        if hparams_file is not None:
            pipeline.load_params(hparams_file)

        if "preprocessors" in config:
            preprocessors = {}
            for key, preprocessor in config.get("preprocessors", {}).items():
                # preprocessors:
                #    key:
                #       name: package.module.ClassName
                #       params:
                #          param1: value1
                #          param2: value2
                if isinstance(preprocessor, dict):
                    Klass = get_class_by_name(
                        preprocessor["name"], default_module_name="pyannote.audio"
                    )
                    params = preprocessor.get("params", {})
                    preprocessors[key] = Klass(**params)
                    continue

                try:
                    # preprocessors:
                    #    key: /path/to/database.yml
                    preprocessors[key] = FileFinder(database_yml=preprocessor)

                except FileNotFoundError:
                    # preprocessors:
                    #    key: /path/to/{uri}.wav
                    template = preprocessor
                    preprocessors[key] = template

            pipeline.preprocessors = preprocessors

        # send pipeline to specified device
        if "device" in config:
            device = torch.device(config["device"])
            try:
                pipeline.to(device)
            except RuntimeError as e:
                print(e)

        return pipeline

    def __init__(self):
        super().__init__()
        self._models: Dict[str, Model] = OrderedDict()
        self._inferences: Dict[str, BaseInference] = OrderedDict()

    def __getattr__(self, name):
        """(Advanced) attribute getter

        Adds support for Model and Inference attributes,
        which are iterated over by Pipeline.to() method.

        See pyannote.pipeline.Pipeline.__getattr__.
        """

        if "_models" in self.__dict__:
            _models = self.__dict__["_models"]
            if name in _models:
                return _models[name]

        if "_inferences" in self.__dict__:
            _inferences = self.__dict__["_inferences"]
            if name in _inferences:
                return _inferences[name]

        return super().__getattr__(name)

    def __setattr__(self, name, value):
        """(Advanced) attribute setter

        Adds support for Model and Inference attributes,
        which are iterated over by Pipeline.to() method.

        See pyannote.pipeline.Pipeline.__setattr__.
        """

        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        _parameters = self.__dict__.get("_parameters")
        _instantiated = self.__dict__.get("_instantiated")
        _pipelines = self.__dict__.get("_pipelines")
        _models = self.__dict__.get("_models")
        _inferences = self.__dict__.get("_inferences")

        if isinstance(value, Model):
            if _models is None:
                msg = "cannot assign models before Pipeline.__init__() call"
                raise AttributeError(msg)
            remove_from(
                self.__dict__, _inferences, _parameters, _instantiated, _pipelines
            )
            _models[name] = value
            return

        if isinstance(value, BaseInference):
            if _inferences is None:
                msg = "cannot assign inferences before Pipeline.__init__() call"
                raise AttributeError(msg)
            remove_from(self.__dict__, _models, _parameters, _instantiated, _pipelines)
            _inferences[name] = value
            return

        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._models:
            del self._models[name]

        elif name in self._inferences:
            del self._inferences[name]

        else:
            super().__delattr__(name)

    @staticmethod
    def setup_hook(file: AudioFile, hook: Optional[Callable] = None) -> Callable:
        def noop(*args, **kwargs):
            return

        return partial(hook or noop, file=file)

    def default_parameters(self):
        raise NotImplementedError()

    def classes(self) -> Union[List, Iterator]:
        """Classes returned by the pipeline

        Returns
        -------
        classes : list of string or string iterator
            Finite list of strings when classes are known in advance
            (e.g. ["MALE", "FEMALE"] for gender classification), or
            infinite string iterator when they depend on the file
            (e.g. "SPEAKER_00", "SPEAKER_01", ... for speaker diarization)

        Usage
        -----
        >>> from collections.abc import Iterator
        >>> classes = pipeline.classes()
        >>> if isinstance(classes, Iterator):  # classes depend on the input file
        >>> if isinstance(classes, list):      # classes are known in advance

        """
        raise NotImplementedError()

    def __call__(self, file: AudioFile, **kwargs):
        fix_reproducibility(getattr(self, "device", torch.device("cpu")))

        if not self.instantiated:
            # instantiate with default parameters when available
            try:
                default_parameters = self.default_parameters()
            except NotImplementedError:
                raise RuntimeError(
                    "A pipeline must be instantiated with `pipeline.instantiate(parameters)` before it can be applied."
                )

            try:
                self.instantiate(default_parameters)
            except ValueError:
                raise RuntimeError(
                    "A pipeline must be instantiated with `pipeline.instantiate(paramaters)` before it can be applied. "
                    "Tried to use parameters provided by `pipeline.default_parameters()` but those are not compatible. "
                )

            warnings.warn(
                f"The pipeline has been automatically instantiated with {default_parameters}."
            )

        file = Audio.validate_file(file)

        if hasattr(self, "preprocessors"):
            file = ProtocolFile(file, lazy=self.preprocessors)

        return self.apply(file, **kwargs)

    def to(self, device: torch.device):
        """Send pipeline to `device`"""

        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        for _, pipeline in self._pipelines.items():
            if hasattr(pipeline, "to"):
                _ = pipeline.to(device)

        for _, model in self._models.items():
            _ = model.to(device)

        for _, inference in self._inferences.items():
            _ = inference.to(device)

        self.device = device

        return self

#########################################################
#                   Source Separation                   #
#########################################################

FlashAttentionConfig = namedtuple(
    "FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


class Attend(nn.Module):
    def __init__(self, dropout=0.0, flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        # Check if there is a compatible device for flash attention

        config = self.cuda_config if q.is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, rotary_embed=None, flash=True
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(
            self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        )

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_embed=rotary_embed,
                            flash=flash_attn,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class BandSplit(nn.Module):
    def __init__(self, dim, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(nn.Module):
    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = nn.ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth), nn.GLU(dim=-1)
            )
            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


class MelBandRoformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        num_bands=60,
        dim_head=64,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        flash_attn=True,
        dim_freqs_in=1025,
        sample_rate=44100,  # needed for mel filter bank from librosa
        stft_n_fft=2048,
        stft_hop_length=512,
        # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Optional[Callable] = None,
        mask_estimator_depth=1,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (
            4096,
            2048,
            1024,
            512,
            256,
        ),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: Callable = torch.hann_window,
        match_input_audio_length=False,  # if True, pad output tensor to match length of input tensor
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        self.layers = nn.ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Transformer(
                            depth=time_transformer_depth,
                            rotary_embed=time_rotary_embed,
                            **transformer_kwargs,
                        ),
                        Transformer(
                            depth=freq_transformer_depth,
                            rotary_embed=freq_rotary_embed,
                            **transformer_kwargs,
                        ),
                    ]
                )
            )

        self.stft_window_fn = partial(
            default(stft_window_fn, torch.hann_window), stft_win_length
        )

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        freqs = torch.stft(
            torch.randn(1, 4096), **self.stft_kwargs, return_complex=True
        ).shape[1]

        # create mel filter bank
        # with librosa.filters.mel as in section 2 of paper

        mel_filter_bank_numpy = mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        # for some reason, it doesn't include the first freq? just force a value for now

        mel_filter_bank[0][0] = 1.0

        # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
        # so let's force a positive value

        mel_filter_bank[-1, -1] = 1.0

        # binary as in paper (then estimated masks are averaged for overlapping regions)

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(
            dim=0
        ).all(), "all frequencies need to be covered by all bands for now"

        repeated_freq_indices = repeat(torch.arange(freqs), "f -> b f", b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, "f -> f s", s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, "f s -> (f s)")

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
        num_bands_per_freq = reduce(freqs_per_band, "b f -> f", "sum")

        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        # band split and mask estimator

        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in num_freqs_per_band.tolist()
        )

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
            )

            self.mask_estimators.append(mask_estimator)

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size, normalized=multi_stft_normalized
        )

        self.match_input_audio_length = match_input_audio_length

    def forward(self, raw_audio, target=None, return_loss_breakdown=False):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")

        batch, channels, raw_audio_length = raw_audio.shape

        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and channels == 1) or (
            self.stereo and channels == 2
        ), "stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)"

        # to stft

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, "* t")

        stft_window = self.stft_window_fn(device=device)

        stft_repr = torch.stft(
            raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True
        )
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, "* f t c")
        stft_repr = rearrange(
            stft_repr, "b s f t c -> b (f s) t c"
        )  # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting

        # index out all frequencies for all frequency ranges across bands ascending in one go

        batch_arange = torch.arange(batch, device=device)[..., None]

        # account for stereo

        x = stft_repr[batch_arange, self.freq_indices]

        # fold the complex (real and imag) into the frequencies dimension

        x = rearrange(x, "b f t c -> b t (f c)")

        x = self.band_split(x)

        # axial / hierarchical attention

        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")

            x = time_transformer(x)

            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")

            x = freq_transformer(x)

            (x,) = unpack(x, ps, "* f d")

        num_stems = len(self.mask_estimators)

        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

        # modulate frequency representation

        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)

        masks = masks.type(stft_repr.dtype)

        # need to average the estimated mask for the overlapped frequencies

        scatter_indices = repeat(
            self.freq_indices,
            "f -> b n f t",
            b=batch,
            n=num_stems,
            t=stft_repr.shape[-1],
        )

        stft_repr_expanded_stems = repeat(stft_repr, "b 1 ... -> b n ...", n=num_stems)
        masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(
            2, scatter_indices, masks
        )

        denom = repeat(self.num_bands_per_freq, "f -> (f r) 1", r=channels)

        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        # modulate stft repr with estimated mask

        stft_repr = stft_repr * masks_averaged

        # istft

        stft_repr = rearrange(
            stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels
        )

        recon_audio = torch.istft(
            stft_repr,
            **self.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=istft_length,
        )

        recon_audio = rearrange(
            recon_audio,
            "(b n s) t -> b n s t",
            b=batch,
            s=self.audio_channels,
            n=num_stems,
        )

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 s t -> b s t")

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, "... t -> ... 1 t")

        target = target[
            ..., : recon_audio.shape[-1]
        ]  # protect against lost length on istft

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.0

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(
                    window_size, self.multi_stft_n_fft
                ),  # not sure what n_fft is across multi resolution stft
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(
                rearrange(recon_audio, "... s t -> (... s) t"), **res_stft_kwargs
            )
            target_Y = torch.stft(
                rearrange(target, "... s t -> (... s) t"), **res_stft_kwargs
            )

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(
                recon_Y, target_Y
            )

        weighted_multi_resolution_loss = (
            multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        )

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)


class Predictor:
    """
    Predictor class for source separation using MelBandRoformer
    """

    def __init__(self, model_sate_dict, device):
        global mel_band_roformer
        self.config = ConfigDict(mel_band_roformer)

        torch.backends.cudnn.benchmark = True
        self.model = MelBandRoformer(**self.config.model)
        self.model.load_state_dict(model_sate_dict)
        self.device = device
        if device == "cuda":
            self.model = self.model.to(device)

        elif device == "cpu":
            logger.warning(
                "CUDA is not available. Run inference on CPU. It will be very slow..."
            )
            self.model = self.model.to(device)

    def predict(self, mix):
        if len(mix.shape) == 2 and mix.shape[1] <= 2:
            mixture = torch.tensor(mix.T, dtype=torch.float32)
        else:
            mixture = torch.tensor(mix, dtype=torch.float32)[None, ...].expand(2, -1)
        first_chunk_time = None
        res, first_chunk_time = self.demix_track(
            self.config, self.model, mixture, self.device, first_chunk_time
        )
        vocals = res["vocals"].T
        instrumental = mix - vocals
        return (vocals, instrumental)

    @staticmethod
    def get_model_from_config(model_type, config):
        if model_type == "mel_band_roformer":

            model = MelBandRoformer(**dict(config.model))
        else:
            print("Unknown model: {}".format(model_type))
            model = None
        return model

    @staticmethod
    def get_windowing_array(window_size, fade_size, device):
        fadein = torch.linspace(0, 1, fade_size)
        fadeout = torch.linspace(1, 0, fade_size)
        window = torch.ones(window_size)
        window[-fade_size:] *= fadeout
        window[:fade_size] *= fadein
        return window.to(device)

    def demix_track(self, config, model, mix, device, first_chunk_time=None):
        C = config.inference.chunk_size
        N = config.inference.num_overlap
        step = C // N
        fade_size = C // 10
        border = C - step

        if mix.shape[1] > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

        windowing_array = self.get_windowing_array(C, fade_size, device)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                if config.training.target_instrument is not None:
                    req_shape = (1,) + tuple(mix.shape)
                else:
                    req_shape = (len(config.training.instruments),) + tuple(mix.shape)

                mix = mix.to(device)
                result = torch.zeros(req_shape, dtype=torch.float32).to(device)
                counter = torch.zeros(req_shape, dtype=torch.float32).to(device)

                i = 0
                last_i = 0
                total_length = mix.shape[1]
                num_chunks = (total_length + step - 1) // step

                if first_chunk_time is None:
                    start_time = time.time()
                    first_chunk = True
                else:
                    start_time = None
                    first_chunk = False
                with tqdm.tqdm(
                    total=total_length,
                    desc="Source Separation",
                    file=tqdm_out,
                    ascii=True,
                    ncols=os.get_terminal_size().columns - 35,
                    miniters=1,
                ) as pbar:
                    while i < total_length:
                        part = mix[:, i : i + C]
                        length = part.shape[-1]
                        if length < C:
                            if length > C // 2 + 1:
                                part = nn.functional.pad(
                                    input=part, pad=(0, C - length), mode="reflect"
                                )
                            else:
                                part = nn.functional.pad(
                                    input=part,
                                    pad=(0, C - length, 0, 0),
                                    mode="constant",
                                    value=0,
                                )

                        if first_chunk and i == 0:
                            chunk_start_time = time.time()

                        x = model(part.unsqueeze(0))[0]

                        window = windowing_array.clone()
                        if i == 0:
                            window[:fade_size] = 1
                        elif i + C >= total_length:
                            window[-fade_size:] = 1

                        result[..., i : i + length] += (
                            x[..., :length] * window[..., :length]
                        )
                        counter[..., i : i + length] += window[..., :length]
                        i += step

                        if first_chunk and i == step:
                            chunk_time = time.time() - chunk_start_time
                            first_chunk_time = chunk_time
                            first_chunk = False

                        current_step = min(step, total_length - last_i)
                        pbar.update(current_step)
                        last_i = i

                estimated_sources = result / counter
                estimated_sources = estimated_sources.cpu().numpy()
                np.nan_to_num(estimated_sources, copy=False, nan=0.0)
                if mix.shape[1] > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]

        if config.training.target_instrument is None:
            return (
                {k: v for k, v in zip(config.training.instruments, estimated_sources)},
                first_chunk_time,
            )
        else:
            return (
                {
                    k: v
                    for k, v in zip(
                        [config.training.target_instrument], estimated_sources
                    )
                },
                first_chunk_time,
            )


#########################################################
#                     DeEcho & DeReverb                 #
#########################################################


class Config:
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.device: str = "cuda:0"
        self.is_half: bool = True
        self.use_jit: bool = False
        self.n_cpu: int = cpu_count()
        self.gpu_name: str | None = None
        self.gpu_mem: int | None = None
        self.instead: str | None = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def has_xpu() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def params_config(self) -> tuple:
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
        elif self.is_half:
            # 6G PU_RAM conf
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G GPU_RAM conf
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41
        return x_pad, x_query, x_center, x_max

    def use_cuda(self) -> None:
        if self.has_xpu():
            self.device = self.instead = "xpu:0"
            self.is_half = True
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        self.gpu_mem = int(
            torch.cuda.get_device_properties(i_device).total_memory / 1024 / 1024 / 1024
            + 0.4
        )

    def use_cpu(self) -> None:
        self.device = self.instead = "cpu"
        self.is_half = False
        self.params_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            self.use_cuda()
        else:
            logger.warning("No supported Nvidia GPU found")
            self.device = self.instead = "cpu"
            self.is_half = False
        return self.params_config()


class ModelParameters(object):
    def __init__(self, config_dict):

        self.param = config_dict

        for k in [
            "mid_side",
            "mid_side_b",
            "mid_side_b2",
            "stereo_w",
            "stereo_n",
            "reverse",
        ]:
            if not k in self.param:
                self.param[k] = False


def crop_center(h1, h2):
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")
    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    h1 = h1[:, :, :, s_time:e_time]

    return h1


class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nout,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)

    def __call__(self, x, return_skip=True):
        skip = self.conv1(x)
        h = self.conv2(skip)
        if return_skip:
            return h, skip
        else:
            return h


class SeperableConv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(SeperableConv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nin,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=nin,
                bias=False,
            ),
            nn.Conv2d(nin, nout, kernel_size=1, bias=False),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, x):
        return self.conv(x)


class ASPPModule(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        dilations=(4, 8, 12),
        activ=nn.ReLU,
        ds_conv=False,
        dropout=False,
    ):
        super(ASPPModule, self).__init__()
        self.ds_conv = ds_conv

        if ds_conv:
            hidden = nin
            conv_type = SeperableConv2DBNActiv
        else:
            hidden = nout
            conv_type = Conv2DBNActiv

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            Conv2DBNActiv(nin, hidden, 1, 1, 0, activ=activ),
        )
        self.conv2 = Conv2DBNActiv(nin, hidden, 1, 1, 0, activ=activ)
        self.conv3 = conv_type(
            nin, hidden, 3, 1, dilations[0], dilations[0], activ=activ
        )
        self.conv4 = conv_type(
            nin, hidden, 3, 1, dilations[1], dilations[1], activ=activ
        )
        self.conv5 = conv_type(
            nin, hidden, 3, 1, dilations[2], dilations[2], activ=activ
        )
        if ds_conv:
            self.bottleneck = nn.Sequential(
                Conv2DBNActiv(nin * 5, nout, 1, 1, 0, activ=activ), nn.Dropout2d(0.1)
            )
        else:
            self.bottleneck = Conv2DBNActiv(hidden * 5, nout, 1, 1, 0, activ=activ)
            self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.bottleneck(out)

        if hasattr(self, "dropout"):
            out = self.dropout(out)

        return out


class Decoder(nn.Module):
    def __init__(
        self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False
    ):
        super(Decoder, self).__init__()
        self.conv = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            skip = crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)
        h = self.conv(x)

        if self.dropout is not None:
            h = self.dropout(h)

        return h


class NewDecoder(nn.Module):
    def __init__(
        self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False
    ):
        super(NewDecoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            skip = crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)

        h = self.conv1(x)

        if self.dropout is not None:
            h = self.dropout(h)

        return h


class BaseASPPNet(nn.Module):
    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.enc1 = Encoder(nin, ch, 3, 2, 1)
        self.enc2 = Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = Encoder(ch * 4, ch * 8, 3, 2, 1)

        self.aspp = ASPPModule(ch * 8, ch * 16, dilations, ds_conv=True)

        self.dec4 = Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        h = self.aspp(h)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


class LSTMModule(nn.Module):
    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super(LSTMModule, self).__init__()
        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
        self.lstm = nn.LSTM(
            input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True
        )
        self.dense = nn.Sequential(
            nn.Linear(nout_lstm, nin_lstm), nn.BatchNorm1d(nin_lstm), nn.ReLU()
        )

    def forward(self, x):
        N, _, nbins, nframes = x.size()
        h = self.conv(x)[:, 0]  # N, nbins, nframes
        h = h.permute(2, 0, 1)  # nframes, N, nbins
        h, _ = self.lstm(h)
        h = self.dense(h.reshape(-1, h.size()[-1]))  # nframes * N, nbins
        h = h.reshape(nframes, N, 1, nbins)
        h = h.permute(1, 2, 3, 0)

        return h


class BaseNet(nn.Module):
    def __init__(
        self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))
    ):
        super(BaseNet, self).__init__()
        self.enc1 = Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = Encoder(nout, nout * 2, 3, 2, 1)
        self.enc3 = Encoder(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = Encoder(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = Encoder(nout * 6, nout * 8, 3, 2, 1)

        self.aspp = ASPPModule(nout * 8, nout * 8, dilations, dropout=True)

        self.dec4 = NewDecoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = NewDecoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = NewDecoder(nout * (2 + 4), nout * 2, 3, 1, 1)
        self.lstm_dec2 = LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = NewDecoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1, False)
        e3 = self.enc3(e2, False)
        e4 = self.enc4(e3, False)
        e5 = self.enc5(e4, False)

        h = self.aspp(e5)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)

        return h


class CascadedNet(nn.Module):
    def __init__(self, n_fft, nout=32, nout_lstm=128):
        super(CascadedNet, self).__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64

        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, nout // 2, self.nin_lstm // 2, nout_lstm),
            Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0),
        )

        self.stg1_high_band_net = BaseNet(
            2, nout // 4, self.nin_lstm // 2, nout_lstm // 2
        )

        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + 2, nout, self.nin_lstm // 2, nout_lstm),
            Conv2DBNActiv(nout, nout // 2, 1, 1, 0),
        )
        self.stg2_high_band_net = BaseNet(
            nout // 4 + 2, nout // 2, self.nin_lstm // 2, nout_lstm // 2
        )

        self.stg3_full_band_net = BaseNet(
            3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm
        )

        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)

    def forward(self, x):
        x = x[:, :, : self.max_bin]

        bandw = x.size()[2] // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)

        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)
        aux2 = torch.cat([l2, h2], dim=2)

        f3_in = torch.cat([x, aux1, aux2], dim=1)
        f3 = self.stg3_full_band_net(f3_in)

        mask = torch.sigmoid(self.out(f3))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        if self.training:
            aux = torch.cat([aux1, aux2], dim=1)
            aux = torch.sigmoid(self.aux_out(aux))
            aux = F.pad(
                input=aux,
                pad=(0, 0, 0, self.output_bin - aux.size()[2]),
                mode="replicate",
            )
            return mask, aux
        else:
            return mask

    def predict_mask(self, x):
        mask = self.forward(x)

        if self.offset > 0:
            mask = mask[:, :, :, self.offset : -self.offset]
            assert mask.size()[3] > 0

        return mask

    def predict(self, x, aggressiveness=None):
        mask = self.forward(x)
        pred_mag = x * mask

        if self.offset > 0:
            pred_mag = pred_mag[:, :, :, self.offset : -self.offset]
            assert pred_mag.size()[3] > 0

        return pred_mag


class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        self.stg1_low_band_net = BaseASPPNet(2, 32)
        self.stg1_high_band_net = BaseASPPNet(2, 32)

        self.stg2_bridge = Conv2DBNActiv(34, 16, 1, 1, 0)
        self.stg2_full_band_net = BaseASPPNet(16, 32)

        self.stg3_bridge = Conv2DBNActiv(66, 32, 1, 1, 0)
        self.stg3_full_band_net = BaseASPPNet(32, 64)

        self.out = nn.Conv2d(64, 2, 1, bias=False)
        self.aux1_out = nn.Conv2d(32, 2, 1, bias=False)
        self.aux2_out = nn.Conv2d(32, 2, 1, bias=False)

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.offset = 128

    def forward(self, x, aggressiveness=None):
        mix = x.detach()
        x = x.clone()

        x = x[:, :, : self.max_bin]

        bandw = x.size()[2] // 2
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(x[:, :, :bandw]),
                self.stg1_high_band_net(x[:, :, bandw:]),
            ],
            dim=2,
        )

        h = torch.cat([x, aux1], dim=1)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        h = torch.cat([x, aux1, aux2], dim=1)
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        mask = torch.sigmoid(self.out(h))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        if self.training:
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(
                input=aux1,
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]),
                mode="replicate",
            )
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(
                input=aux2,
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]),
                mode="replicate",
            )
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            if aggressiveness:
                mask[:, :, : aggressiveness["split_bin"]] = torch.pow(
                    mask[:, :, : aggressiveness["split_bin"]],
                    1 + aggressiveness["value"] / 3,
                )
                mask[:, :, aggressiveness["split_bin"] :] = torch.pow(
                    mask[:, :, aggressiveness["split_bin"] :],
                    1 + aggressiveness["value"],
                )

            return mask * mix

    def predict(self, x_mag, aggressiveness=None):
        h = self.forward(x_mag, aggressiveness)

        if self.offset > 0:
            h = h[:, :, :, self.offset : -self.offset]
            assert h.size()[3] > 0

        return h


class Dereverb:
    def __init__(self, model_name, model_state_dict, agg=10, tta=False, device=None):
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": tta,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        self.config: Config = Config()
        self.config.device = device
        self.version = 3 if "DeEcho" not in model_name else 2
        self.mp: ModelParameters = ModelParameters(eval(f"uvr5_band_v{self.version}"))
        self.model = (
            CascadedASPPNet(self.mp.param["bins"] * 2)
            if self.version == 3
            else CascadedNet(
                self.mp.param["bins"] * 2, 64 if "DeReverb" in model_name else 48
            )
        )

        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        if self.config.is_half:
            self.model = self.model.half()
        self.model.to(self.config.device)

    @staticmethod
    def _wave_to_spectrogram_mt(
        wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False
    ):
        import threading

        if reverse:
            wave_left = np.flip(np.asfortranarray(wave[0]))
            wave_right = np.flip(np.asfortranarray(wave[1]))
        elif mid_side:
            wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
        elif mid_side_b2:
            wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
        else:
            wave_left = np.asfortranarray(wave[0])
            wave_right = np.asfortranarray(wave[1])

        def run_thread(**kwargs):
            global spec_left
            spec_left = librosa.stft(**kwargs)

        thread = threading.Thread(
            target=run_thread,
            kwargs={"y": wave_left, "n_fft": n_fft, "hop_length": hop_length},
        )
        thread.start()
        spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)
        thread.join()

        spec = np.asfortranarray([spec_left, spec_right])

        return spec

    @staticmethod
    def _fft_lp_filter(spec, bin_start, bin_stop):
        g = 1.0
        for b in range(bin_start, bin_stop):
            g -= 1 / (bin_stop - bin_start)
            spec[:, b, :] = g * spec[:, b, :]

        spec[:, bin_stop:, :] *= 0

        return spec

    def _combine_spectrograms(self, specs, mp):
        l = min([specs[i].shape[2] for i in specs])
        spec_c = np.zeros(shape=(2, mp.param["bins"] + 1, l), dtype=np.complex64)
        offset = 0
        bands_n = len(mp.param["band"])

        for d in range(1, bands_n + 1):
            h = mp.param["band"][d]["crop_stop"] - mp.param["band"][d]["crop_start"]
            spec_c[:, offset : offset + h, :l] = specs[d][
                :,
                mp.param["band"][d]["crop_start"] : mp.param["band"][d]["crop_stop"],
                :l,
            ]
            offset += h

        if offset > mp.param["bins"]:
            raise ValueError("Too much bins")

        # lowpass fiter
        if (
            mp.param["pre_filter_start"] > 0
        ):  # and mp.param['band'][bands_n]['res_type'] in ['scipy', 'polyphase']:
            if bands_n == 1:
                spec_c = self._fft_lp_filter(
                    spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"]
                )
            else:
                gp = 1
                for b in range(
                    mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]
                ):
                    g = math.pow(
                        10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0
                    )
                    gp = g
                    spec_c[:, b, :] *= g

        return np.asfortranarray(spec_c)

    def process(
        self, audio,
    ):
        """ 
        audio['waveform]: (samples, 2)
        """
        x_wave, y_wave, x_spec_s, y_spec_s = {}, {}, {}, {}
        bands_n = len(self.mp.param["band"])

        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:  # high-end band
                # librosa loading may be buggy for some audio. ffmpeg will solve this, but it's a pain
                x_wave[d] = librosa.core.resample(
                    audio["waveform"].T,
                    orig_sr=audio["sample_rate"],
                    target_sr=bp["sr"],
                    res_type=bp["res_type"],
                )
                if x_wave[d].ndim == 1:
                    x_wave[d] = np.asfortranarray([x_wave[d], x_wave[d]])

            else:  # lower bands
                x_wave[d] = librosa.core.resample(
                    x_wave[d + 1],
                    orig_sr=self.mp.param["band"][d + 1]["sr"],
                    target_sr=bp["sr"],
                    res_type=bp["res_type"],
                )
            # Stft of wave source
            x_spec_s[d] = self._wave_to_spectrogram_mt(
                x_wave[d],
                bp["hl"],
                bp["n_fft"],
                self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"],
                self.mp.param["reverse"],
            )
        input_high_end_h = (
            self.mp.param["band"][1]["n_fft"] // 2
            - self.mp.param["band"][1]["crop_stop"]
        ) + (self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"])
        input_high_end = x_spec_s[1][
            :,
            self.mp.param["band"][1]["n_fft"] // 2
            - input_high_end_h : self.mp.param["band"][1]["n_fft"] // 2,
            :,
        ]
        x_spec_m = self._combine_spectrograms(x_spec_s, self.mp)
        aggresive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggresive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
        }

        with torch.no_grad():
            pred, x_mag, x_phase = self.inference(
                x_spec_m, self.config.device, self.model, aggressiveness, self.data
            )
        # Postprocess
        if self.data["postprocess"]:
            pred_inv = np.clip(x_mag - pred, 0, np.inf)
            pred = self.mask_silence(pred, pred_inv)
        y_spec_m = pred * x_phase
        v_spec_m = x_spec_m - y_spec_m

        wav_instrument = self.cmb_spectrogram_to_wave(y_spec_m, self.mp)
        wav_vocals = self.cmb_spectrogram_to_wave(v_spec_m, self.mp)

        return (
            np.array(wav_instrument),
            np.array(wav_vocals),
            self.mp.param["sr"],
            self.data["agg"],
        )

    @staticmethod
    def spectrogram_to_wave(spec, hop_length, mid_side, mid_side_b2, reverse):
        spec_left = np.asfortranarray(spec[0])
        spec_right = np.asfortranarray(spec[1])

        wave_left = librosa.istft(spec_left, hop_length=hop_length)
        wave_right = librosa.istft(spec_right, hop_length=hop_length)

        if reverse:
            return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
        elif mid_side:
            return np.asfortranarray(
                [
                    np.add(wave_left, wave_right / 2),
                    np.subtract(wave_left, wave_right / 2),
                ]
            )
        elif mid_side_b2:
            return np.asfortranarray(
                [
                    np.add(wave_right / 1.25, 0.4 * wave_left),
                    np.subtract(wave_left / 1.25, 0.4 * wave_right),
                ]
            )
        else:
            return np.asfortranarray([wave_left, wave_right])

    @staticmethod
    def fft_hp_filter(spec, bin_start, bin_stop):
        g = 1.0
        for b in range(bin_start, bin_stop, -1):
            g -= 1 / (bin_start - bin_stop)
            spec[:, b, :] = g * spec[:, b, :]

        spec[:, 0 : bin_stop + 1, :] *= 0

        return spec

    @staticmethod
    def mask_silence(mag, ref, thres=0.2, min_range=64, fade_size=32):
        if min_range < fade_size * 2:
            raise ValueError("min_range must be >= fade_area * 2")

        mag = mag.copy()

        idx = np.where(ref.mean(axis=(0, 1)) < thres)[0]
        starts = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
        ends = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
        uninformative = np.where(ends - starts > min_range)[0]
        if len(uninformative) > 0:
            starts = starts[uninformative]
            ends = ends[uninformative]
            old_e = None
            for s, e in zip(starts, ends):
                if old_e is not None and s - old_e < fade_size:
                    s = old_e - fade_size * 2

                if s != 0:
                    weight = np.linspace(0, 1, fade_size)
                    mag[:, :, s : s + fade_size] += (
                        weight * ref[:, :, s : s + fade_size]
                    )
                else:
                    s -= fade_size

                if e != mag.shape[2]:
                    weight = np.linspace(1, 0, fade_size)
                    mag[:, :, e - fade_size : e] += (
                        weight * ref[:, :, e - fade_size : e]
                    )
                else:
                    e += fade_size

                mag[:, :, s + fade_size : e - fade_size] += ref[
                    :, :, s + fade_size : e - fade_size
                ]
                old_e = e

        return mag

    def cmb_spectrogram_to_wave(self, spec_m, mp, extra_bins_h=None, extra_bins=None):
        wave_band = {}
        bands_n = len(mp.param["band"])
        offset = 0

        for d in range(1, bands_n + 1):
            bp = mp.param["band"][d]
            spec_s = np.ndarray(
                shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex
            )
            h = bp["crop_stop"] - bp["crop_start"]
            spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[
                :, offset : offset + h, :
            ]

            offset += h
            if d == bands_n:  # higher
                if extra_bins_h:  # if --high_end_process bypass
                    max_bin = bp["n_fft"] // 2
                    spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[
                        :, :extra_bins_h, :
                    ]
                if bp["hpf_start"] > 0:
                    spec_s = self.fft_hp_filter(
                        spec_s, bp["hpf_start"], bp["hpf_stop"] - 1
                    )
                if bands_n == 1:
                    wave = self.spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    )
                else:
                    wave = np.add(
                        wave,
                        self.spectrogram_to_wave(
                            spec_s,
                            bp["hl"],
                            mp.param["mid_side"],
                            mp.param["mid_side_b2"],
                            mp.param["reverse"],
                        ),
                    )
            else:
                sr = mp.param["band"][d + 1]["sr"]
                if d == 1:  # lower
                    spec_s = self._fft_lp_filter(
                        spec_s, bp["lpf_start"], bp["lpf_stop"]
                    )
                    wave = librosa.resample(
                        self.spectrogram_to_wave(
                            spec_s,
                            bp["hl"],
                            mp.param["mid_side"],
                            mp.param["mid_side_b2"],
                            mp.param["reverse"],
                        ),
                        orig_sr=bp["sr"],
                        target_sr=sr,
                        res_type="sinc_fastest",
                    )
                else:  # mid
                    spec_s = self.fft_hp_filter(
                        spec_s, bp["hpf_start"], bp["hpf_stop"] - 1
                    )
                    spec_s = self._fft_lp_filter(
                        spec_s, bp["lpf_start"], bp["lpf_stop"]
                    )
                    wave2 = np.add(
                        wave,
                        self.spectrogram_to_wave(
                            spec_s,
                            bp["hl"],
                            mp.param["mid_side"],
                            mp.param["mid_side_b2"],
                            mp.param["reverse"],
                        ),
                    )
                    wave = librosa.core.resample(
                        wave2, orig_sr=bp["sr"], target_sr=sr, res_type="sinc_fastest"
                    )

        return wave.T

    @staticmethod
    def inference(X_spec, device, model, aggressiveness, data):
        """
        data ： dic configs
        """

        def _execute(
            X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half=True
        ):
            model.eval()
            with torch.no_grad():
                preds = []

                iterations = [n_window]

                with tqdm.tqdm(
                    total=n_window,
                    desc="DeEecho & DeReverb",
                    file=tqdm_out,
                    ascii=True,
                    ncols=os.get_terminal_size().columns - 35,
                    miniters=1,
                ) as pbar:

                    for i in range(n_window):
                        start = i * roi_size
                        X_mag_window = X_mag_pad[
                            None, :, :, start : start + data["window_size"]
                        ]
                        X_mag_window = torch.from_numpy(X_mag_window)
                        if is_half:
                            X_mag_window = X_mag_window.half()
                        X_mag_window = X_mag_window.to(device)

                        pred = model.predict(X_mag_window, aggressiveness)

                        pred = pred.detach().cpu().numpy()
                        preds.append(pred[0])
                        pbar.update(1)

                pred = np.concatenate(preds, axis=2)
            return pred

        def preprocess(X_spec):
            X_mag = np.abs(X_spec)
            X_phase = np.angle(X_spec)

            return X_mag, X_phase

        X_mag, X_phase = preprocess(X_spec)

        coef = X_mag.max()
        X_mag_pre = X_mag / coef

        n_frame = X_mag_pre.shape[2]

        def make_padding(width, cropsize, offset):
            left = offset
            roi_size = cropsize - left * 2
            if roi_size == 0:
                roi_size = cropsize
            right = roi_size - (width % roi_size) + left

            return left, right, roi_size

        pad_l, pad_r, roi_size = make_padding(
            n_frame, data["window_size"], model.offset
        )
        n_window = int(np.ceil(n_frame / roi_size))

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

        if list(model.state_dict().values())[0].dtype == torch.float16:
            is_half = True
        else:
            is_half = False
        pred = _execute(
            X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half
        )
        pred = pred[:, :, :n_frame]

        if data["tta"]:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            n_window += 1

            X_mag_pad = np.pad(
                X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant"
            )

            pred_tta = _execute(
                X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half
            )
            pred_tta = pred_tta[:, :, roi_size // 2 :]
            pred_tta = pred_tta[:, :, :n_frame]

            return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.0j * X_phase)
        else:
            return pred * coef, X_mag, np.exp(1.0j * X_phase)


#########################################################
#                       SOFA MFA                        #
#########################################################


class LoudnessSpectralcentroidAPDetector:
    def __init__(self):
        self.spectral_centroid_threshold = 40
        self.spl_threshold = 20

        self.device = "cpu" if not torch.cuda.is_available() else "cuda"

        self.sample_rate = 44100
        self.hop_length = 512
        self.n_fft = 2048
        self.win_length = 1024
        self.hann_window = torch.hann_window(self.win_length).to(self.device)

        self.conv = nn.Conv1d(
            1, 1, self.hop_length, self.hop_length, self.hop_length // 2, bias=False
        ).to(self.device)
        self.conv.requires_grad_(False)
        self.conv.weight.data.fill_(1.0 / self.hop_length)

    def _get_spl(self, wav):
        out = self.conv(wav.pow(2).unsqueeze(0).unsqueeze(0))
        out = 20 * torch.log10(out.sqrt() / 2 * 10e5)
        return out.squeeze(0).squeeze(0)

    def _get_spectral_centroid(self, wav):
        wav = nn.functional.pad(wav, (self.n_fft // 2, (self.n_fft + 1) // 2))
        fft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            center=False,
            return_complex=True,
        )
        magnitude = fft.abs().pow(2)
        magnitude = magnitude / magnitude.sum(dim=-2, keepdim=True)

        spectral_centroid = torch.sum(
            (1 + torch.arange(0, self.n_fft // 2 + 1))
            .float()
            .unsqueeze(-1)
            .to(self.device)
            * magnitude,
            dim=0,
        )

        return spectral_centroid

    def _get_diff_intervals(self, intervals_a, intervals_b):
        # get complement of interval_b
        if intervals_a.shape[0] <= 0:
            return np.array([])
        if intervals_b.shape[0] <= 0:
            return intervals_a
        intervals_b = np.stack(
            [
                np.concatenate([[0.0], intervals_b[:, 1]]),
                np.concatenate([intervals_b[:, 0], intervals_a[[-1], [-1]]]),
            ],
            axis=-1,
        )
        intervals_b = intervals_b[(intervals_b[:, 0] < intervals_b[:, 1]), :]

        idx_a = 0
        idx_b = 0
        intersection_intervals = []
        while idx_a < intervals_a.shape[0] and idx_b < intervals_b.shape[0]:
            start_a, end_a = intervals_a[idx_a]
            start_b, end_b = intervals_b[idx_b]
            if end_a <= start_b:
                idx_a += 1
                continue
            if end_b <= start_a:
                idx_b += 1
                continue
            intersection_intervals.append([max(start_a, start_b), min(end_a, end_b)])
            if end_a < end_b:
                idx_a += 1
            else:
                idx_b += 1

        return np.array(intersection_intervals)

    def _process_one(
        self,
        wav_path,
        wav_length,
        confidence,
        ph_seq,
        ph_intervals,
        word_seq,
        word_intervals,
    ):
        # input:
        #     wav_path: Path
        #     ph_seq: list of phonemes, SP is the silence phoneme.
        #     ph_intervals: np.ndarray of shape (n_ph, 2), ph_intervals[i] = [start, end]
        #                   means the i-th phoneme starts at start and ends at end.
        #     word_seq: list of words.
        #     word_intervals: np.ndarray of shape (n_word, 2), word_intervals[i] = [start, end]

        # output: same as the input.
        wav = load_wav(wav_path, self.device, self.sample_rate)
        wav = 0.01 * (wav - wav.mean()) / wav.std()

        # ap intervals
        spl = self._get_spl(wav)
        spectral_centroid = self._get_spectral_centroid(wav)
        ap_frame = (spl > self.spl_threshold) & (
            spectral_centroid > self.spectral_centroid_threshold
        )
        ap_frame_diff = torch.diff(
            torch.cat(
                [
                    torch.tensor([0], device=self.device),
                    ap_frame,
                    torch.tensor([0], device=self.device),
                ]
            ),
            dim=0,
        )
        ap_start_idx = torch.where(ap_frame_diff == 1)[0]
        ap_end_idx = torch.where(ap_frame_diff == -1)[0]
        ap_intervals = torch.stack([ap_start_idx, ap_end_idx], dim=-1) * (
            self.hop_length / self.sample_rate
        )
        ap_intervals = self._get_diff_intervals(
            ap_intervals.cpu().numpy(), word_intervals
        )
        if ap_intervals.shape[0] <= 0:
            return (
                wav_path,
                wav_length,
                confidence,
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
            )
        ap_intervals = ap_intervals[(ap_intervals[:, 1] - ap_intervals[:, 0]) > 0.1, :]

        # merge
        ap_tuple_list = [
            ("AP", ap_start, ap_end)
            for (ap_start, ap_end) in zip(ap_intervals[:, 0], ap_intervals[:, 1])
        ]
        word_tuple_list = [
            (word, word_start, word_end)
            for (word, (word_start, word_end)) in zip(word_seq, word_intervals)
        ]
        word_tuple_list.extend(ap_tuple_list)
        ph_tuple_list = [
            (ph, ph_start, ph_end)
            for (ph, (ph_start, ph_end)) in zip(ph_seq, ph_intervals)
        ]
        ph_tuple_list.extend(ap_tuple_list)

        # sort
        word_tuple_list.sort(key=lambda x: x[1])
        ph_tuple_list.sort(key=lambda x: x[1])

        ph_seq = [ph for (ph, _, _) in ph_tuple_list]
        ph_intervals = np.array([(start, end) for (_, start, end) in ph_tuple_list])

        word_seq = [word for (word, _, _) in word_tuple_list]
        word_intervals = np.array([(start, end) for (_, start, end) in word_tuple_list])

        return (
            wav_path,
            wav_length,
            confidence,
            ph_seq,
            ph_intervals,
            word_seq,
            word_intervals,
        )

    def process(self, predictions):
        res = []
        for (
            wav_path,
            wav_length,
            confidence,
            ph_seq,
            ph_intervals,
            word_seq,
            word_intervals,
        ) in predictions:
            prediction = self._process_one(
                wav_path,
                wav_length,
                confidence,
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
            )
            res.append(prediction)

        return res


class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataset = dataframe

    def __getitem__(self, index):
        return tuple(self.dataset.iloc[index])

    def __len__(self):
        return len(self.dataset)


class DictionaryG2P:
    def __init__(self, dictionary):

        self.dictionary = {
            item.split("\t")[0]: item.split("\t")[1].strip().split(" ")
            for item in dictionary.split("\n")
        }

    def __call__(self, text):
        ph_seq, word_seq, ph_idx_to_word_idx = self._g2p(text)

        # The first and last phonemes should be `SP`,
        # and there should not be more than two consecutive `SP`s at any position.
        assert ph_seq[0] == "SP" and ph_seq[-1] == "SP"
        assert all(
            ph_seq[i] != "SP" or ph_seq[i + 1] != "SP" for i in range(len(ph_seq) - 1)
        )
        return ph_seq, word_seq, ph_idx_to_word_idx

    def set_in_format(self, in_format):
        self.in_format = in_format

    @staticmethod
    def check_file_exists(wav_path, in_format):
        if wav_path.stem.endswith("_人声"):
            new_stem = wav_path.stem[: -len("_人声")]
            new_path = wav_path.with_stem(new_stem).with_suffix("." + in_format)
        else:
            new_path = wav_path.with_suffix("." + in_format)

        return new_path.exists(), new_path

    def get_dataset(self, wav_paths):
        # dataset is a pandas dataframe with columns: wav_path, ph_seq, word_seq, ph_idx_to_word_idx
        dataset = []
        for wav_path in wav_paths:
            try:
                exists, new_path = self.check_file_exists(wav_path, self.in_format)
                if exists:
                    with open(new_path, "r", encoding="utf-8") as f:
                        lab_text = f.read().strip()
                    if self.in_format == "txt":
                        raw_text, lab_text = chinese_to_ipa(lab_text)
                        raw_text = " ".join(raw_text)
                        lab_text = " ".join(lab_text)
                    ph_seq, word_seq, ph_idx_to_word_idx = self(lab_text)
                    dataset.append((wav_path, ph_seq, word_seq, ph_idx_to_word_idx))
            except Exception as e:
                e.args = (f" Error when processing {wav_path}: {e} ",)
                raise e
        if len(dataset) <= 0:
            raise ValueError("No valid data found.")

        dataset = pd.DataFrame(
            dataset, columns=["wav_path", "ph_seq", "word_seq", "ph_idx_to_word_idx"]
        )
        dataset = DataFrameDataset(dataset)
        return dataset

    def _g2p(self, input_text):
        word_seq_raw = input_text.strip().split(" ")
        word_seq = []
        word_seq_idx = 0
        ph_seq = ["SP"]
        ph_idx_to_word_idx = [-1]
        for word in word_seq_raw:
            if word not in self.dictionary:
                warnings.warn(f"Word {word} is not in the dictionary. Ignored.")
                continue
            word_seq.append(word)
            phones = self.dictionary[word]
            for i, ph in enumerate(phones):
                if (i == 0 or i == len(phones) - 1) and ph == "SP":
                    warnings.warn(
                        f"The first or last phoneme of word {word} is SP, which is not allowed. "
                        "Please check your dictionary."
                    )
                    continue
                ph_seq.append(ph)
                ph_idx_to_word_idx.append(word_seq_idx)
            if ph_seq[-1] != "SP":
                ph_seq.append("SP")
                ph_idx_to_word_idx.append(-1)
            word_seq_idx += 1

        return ph_seq, word_seq, ph_idx_to_word_idx


class Exporter:
    def __init__(self, predictions, log, out_folder):
        self.predictions = predictions
        self.log = log
        self.out_folder = out_folder

    def save_textgrids(self):
        logger.info("Saving TextGrids...")

        for (
            wav_path,
            wav_length,
            confidence,
            ph_seq,
            ph_intervals,
            word_seq,
            word_intervals,
        ) in self.predictions:
            tg = textgrid.TextGrid()
            word_tier = textgrid.IntervalTier(name="words")
            ph_tier = textgrid.IntervalTier(name="phones")

            for word, (start, end) in zip(word_seq, word_intervals):
                word_tier.add(start, end, word)

            for ph, (start, end) in zip(ph_seq, ph_intervals):
                ph_tier.add(minTime=float(start), maxTime=end, mark=ph)

            tg.append(word_tier)
            tg.append(ph_tier)

            label_path = (
                Path(self.out_folder)
                / "TextGrid"
                / wav_path.with_suffix(".TextGrid").name
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            tg.write(label_path)

    def save_htk(self):
        logger.info("Saving htk labels...")

        for (
            wav_path,
            wav_length,
            confidence,
            ph_seq,
            ph_intervals,
            word_seq,
            word_intervals,
        ) in self.predictions:
            label = ""
            for ph, (start, end) in zip(ph_seq, ph_intervals):
                start_time = int(float(start) * 10000000)
                end_time = int(float(end) * 10000000)
                label += f"{start_time} {end_time} {ph}\n"
            label_path = (
                Path(self.out_folder)
                / "htk"
                / "phones"
                / wav_path.with_suffix(".lab").name
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(label)
                f.close()

            label = ""
            for word, (start, end) in zip(word_seq, word_intervals):
                start_time = int(float(start) * 10000000)
                end_time = int(float(end) * 10000000)
                label += f"{start_time} {end_time} {word}\n"
            label_path = (
                Path(self.out_folder)
                / "htk"
                / "words"
                / wav_path.with_suffix(".lab").name
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(label)
                f.close()

    def save_transcriptions(self):
        logger.info("Saving transcriptions.csv...")

        folder_to_data = {}

        for (
            wav_path,
            wav_length,
            confidence,
            ph_seq,
            ph_intervals,
            word_seq,
            word_intervals,
        ) in self.predictions:
            folder = Path(self.out_folder)
            if folder in folder_to_data:
                curr_data = folder_to_data[folder]
            else:
                curr_data = {
                    "name": [],
                    "word_seq": [],
                    "word_dur": [],
                    "ph_seq": [],
                    "ph_dur": [],
                }

            name = wav_path.with_suffix("").name
            word_seq = " ".join(word_seq)
            ph_seq = " ".join(ph_seq)
            word_dur = []
            ph_dur = []

            last_word_end = 0
            for start, end in word_intervals:
                dur = np.round(end - last_word_end, 5)
                word_dur.append(dur)
                last_word_end += dur

            last_ph_end = 0
            for start, end in ph_intervals:
                dur = np.round(end - last_ph_end, 5)
                ph_dur.append(dur)
                last_ph_end += dur

            word_dur = " ".join([str(i) for i in word_dur])
            ph_dur = " ".join([str(i) for i in ph_dur])

            curr_data["name"].append(name)
            curr_data["word_seq"].append(word_seq)
            curr_data["word_dur"].append(word_dur)
            curr_data["ph_seq"].append(ph_seq)
            curr_data["ph_dur"].append(ph_dur)

            folder_to_data[folder] = curr_data

        for folder, data in folder_to_data.items():
            df = pd.DataFrame(data)
            path = folder / "transcriptions"
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path / "transcriptions.csv", index=False)

    def save_confidence_fn(self):
        logger.info("saving confidence...")

        folder_to_data = {}

        for (
            wav_path,
            wav_length,
            confidence,
            ph_seq,
            ph_intervals,
            word_seq,
            word_intervals,
        ) in self.predictions:
            folder = Path(self.out_folder)
            if folder in folder_to_data:
                curr_data = folder_to_data[folder]
            else:
                curr_data = {
                    "name": [],
                    "confidence": [],
                }

            name = wav_path.with_suffix("").name
            curr_data["name"].append(name)
            curr_data["confidence"].append(confidence)

            folder_to_data[folder] = curr_data

        for folder, data in folder_to_data.items():
            df = pd.DataFrame(data)
            path = folder / "confidence"
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path / "confidence.csv", index=False)

    def export(self, out_formats):
        if "textgrid" in out_formats or "praat" in out_formats:
            self.save_textgrids()
        if (
            "htk" in out_formats
            or "lab" in out_formats
            or "nnsvs" in out_formats
            or "sinsy" in out_formats
        ):
            self.save_htk()
        if (
            "trans" in out_formats
            or "transcription" in out_formats
            or "transcriptions" in out_formats
            or "transcriptions.csv" in out_formats
            or "diffsinger" in out_formats
        ):
            self.save_transcriptions()

        if "confidence" in out_formats:
            self.save_confidence_fn()


MIN_SP_LENGTH = 0.1
SP_MERGE_LENGTH = 0.3


def add_SP(word_seq, word_intervals, wav_length):
    word_seq_res = []
    word_intervals_res = []
    if len(word_seq) == 0:
        word_seq_res.append("SP")
        word_intervals_res.append([0, wav_length])
        return word_seq_res, word_intervals_res

    word_seq_res.append("SP")
    word_intervals_res.append([0, word_intervals[0, 0]])
    for word, (start, end) in zip(word_seq, word_intervals):
        if word_intervals_res[-1][1] < start:
            word_seq_res.append("SP")
            word_intervals_res.append([word_intervals_res[-1][1], start])
        word_seq_res.append(word)
        word_intervals_res.append([start, end])
    if word_intervals_res[-1][1] < wav_length:
        word_seq_res.append("SP")
        word_intervals_res.append([word_intervals_res[-1][1], wav_length])
    if word_intervals[0, 0] <= 0:
        word_seq_res = word_seq_res[1:]
        word_intervals_res = word_intervals_res[1:]

    return word_seq_res, word_intervals_res


def fill_small_gaps(word_seq, word_intervals, wav_length):
    if word_intervals[0, 0] > 0:
        if word_intervals[0, 0] < MIN_SP_LENGTH:
            word_intervals[0, 0] = 0

    for idx in range(len(word_seq) - 1):
        if word_intervals[idx, 1] < word_intervals[idx + 1, 0]:
            if word_intervals[idx + 1, 0] - word_intervals[idx, 1] < SP_MERGE_LENGTH:
                if word_seq[idx] == "AP":
                    if word_seq[idx + 1] == "AP":
                        # 情况1：gap的左右都是AP
                        mean = (word_intervals[idx, 1] + word_intervals[idx + 1, 0]) / 2
                        word_intervals[idx, 1] = mean
                        word_intervals[idx + 1, 0] = mean
                    else:
                        # 情况2：只有左边是AP
                        word_intervals[idx, 1] = word_intervals[idx + 1, 0]
                elif word_seq[idx + 1] == "AP":
                    # 情况3：只有右边是AP
                    word_intervals[idx + 1, 0] = word_intervals[idx, 1]
                else:
                    # 情况4：gap的左右都不是AP
                    if (
                        word_intervals[idx + 1, 0] - word_intervals[idx, 1]
                        < MIN_SP_LENGTH
                    ):
                        mean = (word_intervals[idx, 1] + word_intervals[idx + 1, 0]) / 2
                        word_intervals[idx, 1] = mean
                        word_intervals[idx + 1, 0] = mean

    if word_intervals[-1, 1] < wav_length:
        if wav_length - word_intervals[-1, 1] < MIN_SP_LENGTH:
            word_intervals[-1, 1] = wav_length

    return word_seq, word_intervals


def post_processing(predictions):
    logger.info("Post-processing...")

    res = []
    error_log = []
    for (
        wav_path,
        wav_length,
        confidence,
        ph_seq,
        ph_intervals,
        word_seq,
        word_intervals,
    ) in predictions:
        try:
            # fill small gaps
            word_seq, word_intervals = fill_small_gaps(
                word_seq, word_intervals, wav_length
            )
            ph_seq, ph_intervals = fill_small_gaps(ph_seq, ph_intervals, wav_length)
            # add SP
            word_seq, word_intervals = add_SP(word_seq, word_intervals, wav_length)
            ph_seq, ph_intervals = add_SP(ph_seq, ph_intervals, wav_length)

            res.append(
                [
                    wav_path,
                    wav_length,
                    confidence,
                    ph_seq,
                    ph_intervals,
                    word_seq,
                    word_intervals,
                ]
            )
        except Exception as e:
            error_log.append([wav_path, e])
    return res, error_log


class ResidualBasicBlock(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=None, n_groups=16):
        super(ResidualBasicBlock, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = (
            hidden_dims
            if hidden_dims is not None
            else max(n_groups * (output_dims // n_groups), n_groups)
        )
        self.n_groups = n_groups

        self.block = nn.Sequential(
            nn.Conv1d(
                self.input_dims, self.hidden_dims, kernel_size=3, padding=1, bias=False,
            ),
            nn.GroupNorm(self.n_groups, self.hidden_dims),
            nn.Hardswish(),
            nn.Conv1d(
                self.hidden_dims,
                self.output_dims,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

        self.shortcut = nn.Sequential(
            nn.Linear(self.input_dims, self.output_dims, bias=False)
            if self.input_dims != self.output_dims
            else nn.Identity()
        )

        self.out = nn.Sequential(nn.LayerNorm(self.output_dims), nn.Hardswish(),)

    def forward(self, x):
        x = self.block(x.transpose(1, 2)).transpose(1, 2) + self.shortcut(x)
        x = self.out(x)
        return x


class ResidualBottleNeckBlock(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=None, n_groups=16):
        super(ResidualBottleNeckBlock, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = (
            hidden_dims
            if hidden_dims is not None
            else max(n_groups * ((output_dims // 4) // n_groups), n_groups)
        )
        self.n_groups = n_groups

        self.input_proj = nn.Linear(self.input_dims, self.hidden_dims, bias=False)
        self.conv = nn.Sequential(
            nn.GroupNorm(self.n_groups, self.hidden_dims),
            nn.Hardswish(),
            nn.Conv1d(
                self.hidden_dims,
                self.hidden_dims,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(self.n_groups, self.hidden_dims),
            nn.Hardswish(),
        )
        self.output_proj = nn.Linear(self.hidden_dims, self.output_dims, bias=False)

        self.shortcut = nn.Sequential(
            nn.Linear(self.input_dims, self.output_dims)
            if self.input_dims != self.output_dims
            else nn.Identity()
        )

        self.out = nn.Sequential(nn.LayerNorm(self.output_dims), nn.Hardswish(),)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.output_proj(h)
        return self.out(h + self.shortcut(x))


class DownSampling(nn.Module):
    def __init__(self, input_dims, output_dims, down_sampling_factor=2):
        super(DownSampling, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.down_sampling_factor = down_sampling_factor

        self.conv = nn.Conv1d(
            self.input_dims,
            self.output_dims,
            kernel_size=down_sampling_factor,
            stride=down_sampling_factor,
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        padding_len = x.shape[-1] % self.down_sampling_factor
        if padding_len != 0:
            x = nn.functional.pad(x, (0, self.down_sampling_factor - padding_len))
        return self.conv(x).transpose(1, 2)


class UpSampling(nn.Module):
    def __init__(self, input_dims, output_dims, up_sampling_factor=2):
        super(UpSampling, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.up_sampling_factor = up_sampling_factor

        self.conv = nn.ConvTranspose1d(
            self.input_dims,
            self.output_dims,
            kernel_size=up_sampling_factor,
            stride=up_sampling_factor,
        )

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class UNetBackbone(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_dims,
        block,
        down_sampling,
        up_sampling,
        down_sampling_factor=2,
        down_sampling_times=5,
        channels_scaleup_factor=2,
        **kwargs,
    ):
        """_summary_

        Args:
            input_dims (int):
            output_dims (int):
            hidden_dims (int):
            block (nn.Module): shape: (B, T, C) -> shape: (B, T, C)
            down_sampling (nn.Module): shape: (B, T, C) -> shape: (B, T/down_sampling_factor, C*2)
            up_sampling (nn.Module): shape: (B, T, C) -> shape: (B, T*down_sampling_factor, C/2)
        """
        super(UNetBackbone, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        self.divisible_factor = down_sampling_factor ** down_sampling_times

        self.encoders = nn.ModuleList()
        self.encoders.append(block(input_dims, hidden_dims, **kwargs))
        for i in range(down_sampling_times - 1):
            i += 1
            self.encoders.append(
                nn.Sequential(
                    down_sampling(
                        int(channels_scaleup_factor ** (i - 1)) * hidden_dims,
                        int(channels_scaleup_factor ** i) * hidden_dims,
                        down_sampling_factor,
                    ),
                    block(
                        int(channels_scaleup_factor ** i) * hidden_dims,
                        int(channels_scaleup_factor ** i) * hidden_dims,
                        **kwargs,
                    ),
                )
            )

        self.bottle_neck = nn.Sequential(
            down_sampling(
                int(channels_scaleup_factor ** (down_sampling_times - 1)) * hidden_dims,
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                down_sampling_factor,
            ),
            block(
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                **kwargs,
            ),
            up_sampling(
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                int(channels_scaleup_factor ** (down_sampling_times - 1)) * hidden_dims,
                down_sampling_factor,
            ),
        )

        self.decoders = nn.ModuleList()
        for i in range(down_sampling_times - 1):
            i += 1
            self.decoders.append(
                nn.Sequential(
                    block(
                        int(channels_scaleup_factor ** (down_sampling_times - i))
                        * hidden_dims,
                        int(channels_scaleup_factor ** (down_sampling_times - i))
                        * hidden_dims,
                        **kwargs,
                    ),
                    up_sampling(
                        int(channels_scaleup_factor ** (down_sampling_times - i))
                        * hidden_dims,
                        int(channels_scaleup_factor ** (down_sampling_times - i - 1))
                        * hidden_dims,
                        down_sampling_factor,
                    ),
                )
            )
        self.decoders.append(block(hidden_dims, output_dims, **kwargs))

    def forward(self, x):
        T = x.shape[1]
        padding_len = T % self.divisible_factor
        if padding_len != 0:
            x = nn.functional.pad(x, (0, 0, 0, self.divisible_factor - padding_len))

        h = [x]
        for encoder in self.encoders:
            h.append(encoder(h[-1]))

        h_ = [self.bottle_neck(h[-1])]
        for i, decoder in enumerate(self.decoders):
            h_.append(decoder(h_[-1] + h[-1 - i]))

        out = h_[-1]
        out = out[:, :T, :]

        return out


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
                audio.device
            )
        if center:
            pad_left = n_fft_new // 2
            pad_right = (n_fft_new + 1) // 2
            audio = F.pad(audio, (pad_left, pad_right))

        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=False,
            return_complex=True,
        )
        magnitude = fft.abs()

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new

        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


melspec_transform = None


class MelSpecExtractor:
    def __init__(
        self,
        n_mels,
        sample_rate,
        win_length,
        hop_length,
        n_fft,
        fmin,
        fmax,
        clamp,
        device=None,
        scale_factor=None,
    ):
        global melspec_transform
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if melspec_transform is None:
            melspec_transform = MelSpectrogram(
                n_mel_channels=n_mels,
                sampling_rate=sample_rate,
                win_length=win_length,
                hop_length=hop_length,
                n_fft=n_fft,
                mel_fmin=fmin,
                mel_fmax=fmax,
                clamp=clamp,
            ).to(device)

    def __call__(self, waveform, key_shift=0):
        return melspec_transform(waveform.unsqueeze(0), key_shift).squeeze(0)


class LitForcedAlignmentTask(pl.LightningModule):
    def __init__(
        self,
        vocab_text,
        model_config,
        melspec_config,
        optimizer_config,
        loss_config,
        data_augmentation_enabled,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = yaml.safe_load(vocab_text)

        self.backbone = UNetBackbone(
            melspec_config["n_mels"],
            model_config["hidden_dims"],
            model_config["hidden_dims"],
            ResidualBasicBlock,
            DownSampling,
            UpSampling,
            down_sampling_factor=model_config["down_sampling_factor"],  # 3
            down_sampling_times=model_config["down_sampling_times"],  # 7
            channels_scaleup_factor=model_config["channels_scaleup_factor"],  # 1.5
        )
        self.head = nn.Linear(
            model_config["hidden_dims"], self.vocab["<vocab_size>"] + 2
        )
        self.melspec_config = melspec_config  # Required for inference
        self.optimizer_config = optimizer_config

        self.pseudo_label_ratio = loss_config["function"]["pseudo_label_ratio"]
        self.pseudo_label_auto_theshold = 0.5

        self.losses_names = [
            "ph_frame_GHM_loss",
            "ph_edge_GHM_loss",
            "ph_edge_EMD_loss",
            "ph_edge_diff_loss",
            "ctc_GHM_loss",
            "consistency_loss",
            "pseudo_label_loss",
            "total_loss",
        ]
        self.data_augmentation_enabled = data_augmentation_enabled

        # get_melspec
        self.get_melspec = None

        # validation_step_outputs
        self.validation_step_outputs = {"losses": []}

        self.inference_mode = "force"

    def load_pretrained(self, pretrained_model):
        self.backbone = pretrained_model.backbone
        if self.vocab["<vocab_size>"] == pretrained_model.vocab["<vocab_size>"]:
            self.head = pretrained_model.head
        else:
            self.head = nn.Linear(
                self.backbone.output_dims, self.vocab["<vocab_size>"] + 2
            )

    @staticmethod
    def forward_pass(
        T,
        S,
        prob_log,
        not_edge_prob_log,
        edge_prob_log,
        curr_ph_max_prob_log,
        dp,
        backtrack_s,
        ph_seq_id,
        prob3_pad_len,
    ):
        for t in range(1, T):
            # [t-1,s] -> [t,s]
            prob1 = dp[t - 1, :] + prob_log[t, :] + not_edge_prob_log[t]

            prob2 = np.empty(S, dtype=np.float32)
            prob2[0] = -np.inf
            for i in range(1, S):
                prob2[i] = (
                    dp[t - 1, i - 1]
                    + prob_log[t, i - 1]
                    + edge_prob_log[t]
                    + curr_ph_max_prob_log[i - 1] * (T / S)
                )

            # [t-1,s-2] -> [t,s]
            prob3 = np.empty(S, dtype=np.float32)
            for i in range(prob3_pad_len):
                prob3[i] = -np.inf
            for i in range(prob3_pad_len, S):
                if (
                    i - prob3_pad_len + 1 < S - 1
                    and ph_seq_id[i - prob3_pad_len + 1] != 0
                ):
                    prob3[i] = -np.inf
                else:
                    prob3[i] = (
                        dp[t - 1, i - prob3_pad_len]
                        + prob_log[t, i - prob3_pad_len]
                        + edge_prob_log[t]
                        + curr_ph_max_prob_log[i - prob3_pad_len] * (T / S)
                    )

            stacked_probs = np.empty((3, S), dtype=np.float32)
            for i in range(S):
                stacked_probs[0, i] = prob1[i]
                stacked_probs[1, i] = prob2[i]
                stacked_probs[2, i] = prob3[i]

            for i in range(S):
                max_idx = 0
                max_val = stacked_probs[0, i]
                for j in range(1, 3):
                    if stacked_probs[j, i] > max_val:
                        max_val = stacked_probs[j, i]
                        max_idx = j
                dp[t, i] = max_val
                backtrack_s[t, i] = max_idx

            for i in range(S):
                if backtrack_s[t, i] == 0:
                    curr_ph_max_prob_log[i] = max(
                        curr_ph_max_prob_log[i], prob_log[t, i]
                    )
                elif backtrack_s[t, i] > 0:
                    curr_ph_max_prob_log[i] = prob_log[t, i]

            for i in range(S):
                if ph_seq_id[i] == 0:
                    curr_ph_max_prob_log[i] = 0

        return dp, backtrack_s, curr_ph_max_prob_log

    def _decode(self, ph_seq_id, ph_prob_log, edge_prob):
        # ph_seq_id: (S)
        # ph_prob_log: (T, vocab_size)
        # edge_prob: (T,2)
        T = ph_prob_log.shape[0]
        S = len(ph_seq_id)
        # not_SP_num = (ph_seq_id > 0).sum()
        prob_log = ph_prob_log[:, ph_seq_id]

        edge_prob_log = np.log(edge_prob + 1e-6).astype("float32")
        not_edge_prob_log = np.log(1 - edge_prob + 1e-6).astype("float32")

        # init
        curr_ph_max_prob_log = np.full(S, -np.inf)
        dp = np.full((T, S), -np.inf, dtype="float32")  # (T, S)
        backtrack_s = np.full_like(dp, -1, dtype="int32")
        # 如果mode==forced，只能从SP开始或者从第一个音素开始
        if self.inference_mode == "force":
            dp[0, 0] = prob_log[0, 0]
            curr_ph_max_prob_log[0] = prob_log[0, 0]
            if ph_seq_id[0] == 0 and prob_log.shape[-1] > 1:
                dp[0, 1] = prob_log[0, 1]
                curr_ph_max_prob_log[1] = prob_log[0, 1]
        # 如果mode==match，可以从任意音素开始
        elif self.inference_mode == "match":
            for i, ph_id in enumerate(ph_seq_id):
                dp[0, i] = prob_log[0, i]
                curr_ph_max_prob_log[i] = prob_log[0, i]

        # forward
        prob3_pad_len = 2 if S >= 2 else 1
        dp, backtrack_s, curr_ph_max_prob_log = self.forward_pass(
            T,
            S,
            prob_log,
            not_edge_prob_log,
            edge_prob_log,
            curr_ph_max_prob_log,
            dp,
            backtrack_s,
            ph_seq_id,
            prob3_pad_len,
        )

        # backward
        ph_idx_seq = []
        ph_time_int = []
        frame_confidence = []
        # 如果mode==forced，只能从最后一个音素或者SP结束
        if self.inference_mode == "force":
            if S >= 2 and dp[-1, -2] > dp[-1, -1] and ph_seq_id[-1] == 0:
                s = S - 2
            else:
                s = S - 1
        # 如果mode==match，可以从任意音素结束
        elif self.inference_mode == "match":
            s = np.argmax(dp[-1, :])
        else:
            raise ValueError("inference_mode must be 'force' or 'match'")

        for t in np.arange(T - 1, -1, -1):
            assert backtrack_s[t, s] >= 0 or t == 0
            frame_confidence.append(dp[t, s])
            if backtrack_s[t, s] != 0:
                ph_idx_seq.append(s)
                ph_time_int.append(t)
                s -= backtrack_s[t, s]
        ph_idx_seq.reverse()
        ph_time_int.reverse()
        frame_confidence.reverse()
        frame_confidence = np.exp(
            np.diff(
                np.pad(frame_confidence, (1, 0), "constant", constant_values=0.0), 1
            )
        )

        return (
            np.array(ph_idx_seq),
            np.array(ph_time_int),
            np.array(frame_confidence),
        )

    def _infer_once(
        self,
        melspec,
        wav_length,
        ph_seq,
        word_seq=None,
        ph_idx_to_word_idx=None,
        return_ctc=False,
        return_plot=False,
    ):
        ph_seq_id = np.array([self.vocab[ph] for ph in ph_seq])
        ph_mask = np.zeros(self.vocab["<vocab_size>"])
        ph_mask[ph_seq_id] = 1
        ph_mask[0] = 1
        ph_mask = torch.from_numpy(ph_mask)
        if word_seq is None:
            word_seq = ph_seq
            ph_idx_to_word_idx = np.arange(len(ph_seq))

        # forward
        with torch.no_grad():
            (
                ph_frame_logits,  # (B, T, vocab_size)
                ph_edge_logits,  # (B, T)
                ctc_logits,  # (B, T, vocab_size)
            ) = self.forward(melspec.transpose(1, 2))
        if wav_length is not None:
            num_frames = int(
                (
                    (
                        wav_length
                        * self.melspec_config["scale_factor"]
                        * self.melspec_config["sample_rate"]
                        + 0.5
                    )
                )
                / self.melspec_config["hop_length"]
            )
            ph_frame_logits = ph_frame_logits[:, :num_frames, :]
            ph_edge_logits = ph_edge_logits[:, :num_frames]
            ctc_logits = ctc_logits[:, :num_frames, :]

        ph_mask = (
            ph_mask.to(torch.float32)
            .to(ph_frame_logits.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .logical_not()
            * 1e9
        )
        ph_frame_pred = (
            torch.nn.functional.softmax(
                ph_frame_logits.float() - ph_mask.float(), dim=-1
            )
            .squeeze(0)
            .cpu()
            .numpy()
            .astype("float32")
        )
        ph_prob_log = (
            torch.log_softmax(ph_frame_logits.float() - ph_mask.float(), dim=-1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype("float32")
        )
        ph_edge_pred = (
            (torch.nn.functional.sigmoid(ph_edge_logits.float()) - 0.1) / 0.8
        ).clamp(0.0, 1.0)
        ph_edge_pred = ph_edge_pred.squeeze(0).cpu().numpy().astype("float32")
        ctc_logits = (
            ctc_logits.float().squeeze(0).cpu().numpy().astype("float32")
        )  # (ctc_logits.squeeze(0) - ph_mask)

        T, vocab_size = ph_frame_pred.shape

        # decode
        edge_diff = np.concatenate((np.diff(ph_edge_pred, axis=0), [0]), axis=0)
        edge_prob = (ph_edge_pred + np.concatenate(([0], ph_edge_pred[:-1]))).clip(0, 1)
        (ph_idx_seq, ph_time_int_pred, frame_confidence,) = self._decode(
            ph_seq_id, ph_prob_log, edge_prob,
        )
        total_confidence = np.exp(np.mean(np.log(frame_confidence + 1e-6)) / 3)

        # postprocess
        frame_length = self.melspec_config["hop_length"] / (
            self.melspec_config["sample_rate"] * self.melspec_config["scale_factor"]
        )
        ph_time_fractional = (edge_diff[ph_time_int_pred] / 2).clip(-0.5, 0.5)
        ph_time_pred = frame_length * (
            np.concatenate(
                [ph_time_int_pred.astype("float32") + ph_time_fractional, [T],]
            )
        )
        ph_intervals = np.stack([ph_time_pred[:-1], ph_time_pred[1:]], axis=1)

        ph_seq_pred = []
        ph_intervals_pred = []
        word_seq_pred = []
        word_intervals_pred = []

        word_idx_last = -1
        for i, ph_idx in enumerate(ph_idx_seq):
            # ph_idx只能用于两种情况：ph_seq和ph_idx_to_word_idx
            if ph_seq[ph_idx] == "SP":
                continue
            ph_seq_pred.append(ph_seq[ph_idx])
            ph_intervals_pred.append(ph_intervals[i, :])

            word_idx = ph_idx_to_word_idx[ph_idx]
            if word_idx == word_idx_last:
                word_intervals_pred[-1][1] = ph_intervals[i, 1]
            else:
                word_seq_pred.append(word_seq[word_idx])
                word_intervals_pred.append([ph_intervals[i, 0], ph_intervals[i, 1]])
                word_idx_last = word_idx
        ph_seq_pred = np.array(ph_seq_pred)
        ph_intervals_pred = np.array(ph_intervals_pred).clip(min=0, max=None)
        word_seq_pred = np.array(word_seq_pred)
        word_intervals_pred = np.array(word_intervals_pred).clip(min=0, max=None)

        # ctc decode
        ctc = None
        if return_ctc:
            ctc = np.argmax(ctc_logits, axis=-1)
            ctc_index = np.concatenate([[0], ctc])
            ctc_index = (ctc_index[1:] != ctc_index[:-1]) * ctc != 0
            ctc = ctc[ctc_index]
            ctc = np.array([self.vocab[ph] for ph in ctc if ph != 0])

        fig = None
        ph_intervals_pred_int = (
            (ph_intervals_pred / frame_length).round().astype("int32")
        )
        if return_plot:
            raise NotImplementedError()
            # ph_idx_frame = np.zeros(T).astype("int32")
            # last_ph_idx = 0
            # for ph_idx, ph_time in zip(ph_idx_seq, ph_time_int_pred):
            #     ph_idx_frame[ph_time] += ph_idx - last_ph_idx
            #     last_ph_idx = ph_idx
            # ph_idx_frame = np.cumsum(ph_idx_frame)
            # args = {
            #     "melspec": melspec.cpu().numpy(),
            #     "ph_seq": ph_seq_pred,
            #     "ph_intervals": ph_intervals_pred_int,
            #     "frame_confidence": frame_confidence,
            #     "ph_frame_prob": ph_frame_pred[:, ph_seq_id],
            #     "ph_frame_id_gt": ph_idx_frame,
            #     "edge_prob": edge_prob,
            # }
            # fig = plot_for_valid(**args)

        return (
            ph_seq_pred,
            ph_intervals_pred,
            word_seq_pred,
            word_intervals_pred,
            total_confidence,
            ctc,
            fig,
        )

    def set_inference_mode(self, mode):
        self.inference_mode = mode

    def on_predict_start(self):
        if self.get_melspec is None:
            self.get_melspec = MelSpecExtractor(**self.melspec_config)

    def predict_step(self, batch, batch_idx):
        try:
            wav_path, ph_seq, word_seq, ph_idx_to_word_idx = batch
            waveform = load_wav(
                wav_path, self.device, self.melspec_config["sample_rate"]
            )
            wav_length = waveform.shape[0] / self.melspec_config["sample_rate"]
            melspec = self.get_melspec(waveform).detach().unsqueeze(0)
            melspec = (melspec - melspec.mean()) / melspec.std()
            melspec = repeat(
                melspec, "B C T -> B C (T N)", N=self.melspec_config["scale_factor"]
            )

            (
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
                confidence,
                _,
                _,
            ) = self._infer_once(
                melspec, wav_length, ph_seq, word_seq, ph_idx_to_word_idx, False, False
            )

            return (
                wav_path,
                wav_length,
                confidence,
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
            )
        except Exception as e:
            e.args += (f"{str(wav_path)}",)
            raise e

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        h = self.backbone(*args, **kwargs)
        logits = self.head(h)
        ph_frame_logits = logits[:, :, 2:]
        ph_edge_logits = logits[:, :, 0]
        ctc_logits = torch.cat([logits[:, :, [1]], logits[:, :, 3:]], dim=-1)
        return ph_frame_logits, ph_edge_logits, ctc_logits


#########################################################
# !!!             DO NOT MODIFY ABOVE CODES             #
#########################################################
#                       main utils                      #
# ------------------------------------------------------#
#                       postprocess                     #
#########################################################


def standardization(audio, sr=44100):
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object, the audio to be preprocessed.

    Returns:
        dict: A dictionary containing the preprocessed audio waveform, audio file name, and sample rate, formatted as:
              {
                  "waveform": np.ndarray, the preprocessed audio waveform, dtype is np.float32, shape is (num_samples,)
                  "name": str, the audio file name
                  "sample_rate": int, the audio sample rate
              }

    Raises:
        ValueError: If the audio parameter is neither a str nor an AudioSegment.
    """
    global audio_count
    name = "audio"

    if isinstance(audio, str):
        name = os.path.basename(audio)
        try:
            audio = AudioSegment.from_file(audio)
        except:
            os.remove(audio)
            return None
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")

    # Convert the audio file to WAV format
    audio = audio.set_frame_rate(sr)
    audio = audio.set_sample_width(2)  # Set bit depth to 16bit

    # Calculate the gain to be applied
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    # Normalize volume and limit gain range to between -3 and 3
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    if audio.channels == 1:
        normalized_audio = normalized_audio.set_channels(1)  # Set to mono
        waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
        waveform = np.column_stack((waveform, waveform))
    elif audio.channels == 2:
        normalized_audio = normalized_audio.split_to_mono()
        left_channel = normalized_audio[0]
        right_channel = normalized_audio[1]
        left_channel = np.array(left_channel.get_array_of_samples(), dtype=np.float32)
        right_channel = np.array(right_channel.get_array_of_samples(), dtype=np.float32)
        # 检查左右声道长度是否一致
        waveform = np.column_stack((left_channel, right_channel))

    max_amplitude = np.max(np.abs(waveform))
    waveform /= max_amplitude  # Normalize
    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": sr,
    }


def source_separation(predictor, audio):
    """
    Separate the audio into vocals and non-vocals using the given predictor.

    Args:
        predictor: The separation model predictor.
        audio (str or dict): The audio file path or a dictionary containing audio waveform and sample rate.

    Returns:
        dict: A dictionary containing the separated vocals and updated audio waveform.
    """

    mix, rate = None, None

    if isinstance(audio, str):
        mix, rate = librosa.load(audio, mono=False, sr=44100)
    else:
        # resample to 44100
        rate = audio["sample_rate"]
        mix = librosa.resample(audio["waveform"], orig_sr=rate, target_sr=44100)
    vocals, no_vocals = predictor.predict(mix)
    # convert vocals back to previous sample rate
    # vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    audio["waveform"] = vocals
    audio["company"] = no_vocals
    return audio


def calculate_silence_duration(audio, sr, top_db=30, frame_length=2048, hop_length=512):
    """
    Args:
        audio: 输入的音频信号 (np.array)
        sr: 采样率 (Hz)
        top_db: 静音判定的分贝阈值 (dB)
        frame_length: 分析帧长度
        hop_length: 帧移长度
        
    Returns:
        silence_duration: 静音总时长(秒)
        silence_proportion: 静音占比(0-1)
    """
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # cal frame energy
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, frame_length),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
    )
    frame_energies = 10 * np.log10(np.mean(frames ** 2, axis=1))  # dB计算

    # silent detect
    silence_frames = frame_energies < -top_db
    silence_samples = np.sum(silence_frames) * hop_length

    if (n_frames * hop_length + frame_length) > len(audio):
        last_frame = audio[-frame_length:]
        last_energy = 10 * np.log10(np.mean(last_frame ** 2))
        if last_energy < -top_db:
            silence_samples += len(audio) - (n_frames * hop_length)

    silence_duration = silence_samples / sr
    total_duration = len(audio) / sr
    silence_proportion = silence_duration / total_duration

    return silence_duration, silence_proportion


def dereverb(dereverb_model, audio, is_dereverb=False):
    """
    Perform dereverberation on the given audio.
    """
    results = dereverb_model.process(audio)
    if is_dereverb:
        audio["waveform"] = np.array(results[0], dtype=np.float32)
    else:
        silent_dur0, _ = calculate_silence_duration(results[0], audio["sample_rate"])
        silent_dur1, _ = calculate_silence_duration(results[1], audio["sample_rate"])

        if silent_dur0 > silent_dur1:
            audio["waveform"] = np.array(results[1], dtype=np.float32)
        else:
            audio["waveform"] = np.array(results[0], dtype=np.float32)
    audio["sample_rate"] = results[2]

    return audio


def speaker_diarization(dia_pipeline, audio, device="cuda:0"):
    """
    Perform speaker diarization on the given audio.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.

    Returns:
        pd.DataFrame: A dataframe containing segments with speaker labels.
    """

    waveform = torch.tensor(audio["waveform"]).to(device)
    waveform = torch.unsqueeze(waveform, 0)

    segments = dia_pipeline(
        {"waveform": waveform, "sample_rate": audio["sample_rate"], "channel": 0,}
    )

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True), columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    return diarize_df


def cut_vad_segments(vad_list):
    """
    Merge and trim VAD segments to ensure segment durations are between 3-30 seconds.
    No segments are filtered out, and merging is based solely on time gaps.

    Args:
        vad_list (list): List of VAD segments with start and end times.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = 3  # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = 3  # min segment length in seconds
    MAX_SEGMENT_LENGTH = 30  # max segment length in seconds

    updated_list = []

    for idx, vad in enumerate(vad_list):
        # If the list is empty, add the current segment directly
        if not updated_list:
            updated_list.append(vad.copy())
            continue

        last_start_time = updated_list[-1]["start"]
        last_end_time = updated_list[-1]["end"]

        # If the current segment duration exceeds the maximum limit, split the segment
        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                new_vad = vad.copy()
                new_vad["end"] = current_start + MAX_SEGMENT_LENGTH
                updated_list.append(new_vad)
                current_start += MAX_SEGMENT_LENGTH
                new_vad = vad.copy()
                new_vad["start"] = current_start
                new_vad["end"] = segment_end
            if segment_end - current_start > 0:
                updated_list.append(new_vad)
            continue

        # Check if the gap between current segment and last segment is smaller than MERGE_GAP
        if vad["start"] - last_end_time < MERGE_GAP:
            # If the merged duration doesn't exceed the maximum limit, merge them
            if vad["end"] - last_start_time <= MAX_SEGMENT_LENGTH:
                updated_list[-1]["end"] = vad["end"]
            else:
                # If merging would exceed the maximum limit, don't merge, just add
                updated_list.append(vad.copy())
        else:
            # If gap is larger than MERGE_GAP, add as new segment
            updated_list.append(vad.copy())

    # Final pass to ensure all segments are between MIN_SEGMENT_LENGTH and MAX_SEGMENT_LENGTH
    final_list = []
    for vad in updated_list:
        segment_duration = vad["end"] - vad["start"]
        if segment_duration > MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                new_vad = vad.copy()
                new_vad["end"] = current_start + MAX_SEGMENT_LENGTH
                final_list.append(new_vad)
                current_start += MAX_SEGMENT_LENGTH
                new_vad = vad.copy()
                new_vad["start"] = current_start
                new_vad["end"] = segment_end
            if segment_end - current_start > 0:
                final_list.append(new_vad)
        elif segment_duration >= MIN_SEGMENT_LENGTH:
            final_list.append(vad)

    return final_list


def split_text_by_time(text, start_time, end_time, max_duration):
    """
    Split text based on time proportion
    
    Args:
        text: Text to be split
        start_time: Start time
        end_time: End time
        max_duration: Maximum duration
    
    Returns:
        list: List of split text parts
    """
    total_duration = end_time - start_time
    num_parts = int(np.ceil(total_duration / max_duration))

    if num_parts <= 1:
        return [text]

    # Split text evenly by character count
    text_length = len(text)
    chars_per_part = text_length // num_parts

    parts = []
    for i in range(num_parts):
        start_idx = i * chars_per_part
        end_idx = start_idx + chars_per_part if i < num_parts - 1 else text_length
        parts.append(text[start_idx:end_idx])

    return parts


def cut_vad_segments_with_text(vad_list):
    """
    Merge and trim VAD segments to ensure segment duration is between 3-30 seconds, 
    while merging text content.
    
    Args:
        vad_list (list): List of VAD segments containing start, end, text fields
    
    Returns:
        list: List of merged and trimmed VAD segments
    """
    MERGE_GAP = 3  # Merge gap, segments with gap smaller than this will be merged
    MIN_SEGMENT_LENGTH = 3  # Minimum segment length
    MAX_SEGMENT_LENGTH = 20  # Maximum segment length

    updated_list = []

    for idx, vad in enumerate(vad_list):
        # If list is empty, directly add current segment
        if not updated_list:
            updated_list.append(vad.copy())
            continue

        last_start_time = updated_list[-1]["start"]
        last_end_time = updated_list[-1]["end"]

        # Check if the gap between current segment and last segment is smaller than MERGE_GAP
        if vad["start"] - last_end_time < MERGE_GAP:
            # If merged duration doesn't exceed maximum limit, merge them
            if vad["end"] - last_start_time <= MAX_SEGMENT_LENGTH:
                # Merge time and text
                updated_list[-1]["end"] = vad["end"]
                updated_list[-1]["text"] = updated_list[-1]["text"] + vad["text"]
            else:
                # If merging would exceed maximum limit, don't merge, add directly
                updated_list.append(vad.copy())
        else:
            # If gap is larger than MERGE_GAP, add as new segment
            updated_list.append(vad.copy())

        # If current segment duration exceeds maximum limit, split the segment
        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            text_parts = split_text_by_time(
                vad["text"], vad["start"], vad["end"], MAX_SEGMENT_LENGTH
            )
            part_index = 0

            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                new_vad = {
                    "start": current_start,
                    "end": current_start + MAX_SEGMENT_LENGTH,
                    "text": text_parts[part_index]
                    if part_index < len(text_parts)
                    else "",
                }
                updated_list.append(new_vad)
                current_start += MAX_SEGMENT_LENGTH
                part_index += 1

            if segment_end - current_start > 0:
                new_vad = {
                    "start": current_start,
                    "end": segment_end,
                    "text": text_parts[part_index]
                    if part_index < len(text_parts)
                    else "",
                }
                updated_list.append(new_vad)
            continue

    # Final check to ensure all segments are between MIN_SEGMENT_LENGTH and MAX_SEGMENT_LENGTH
    final_list = []
    for vad in updated_list:
        segment_duration = vad["end"] - vad["start"]
        if segment_duration > MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            text_parts = split_text_by_time(
                vad["text"], vad["start"], vad["end"], MAX_SEGMENT_LENGTH
            )
            part_index = 0

            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                new_vad = {
                    "start": current_start,
                    "end": current_start + MAX_SEGMENT_LENGTH,
                    "text": text_parts[part_index]
                    if part_index < len(text_parts)
                    else "",
                }
                final_list.append(new_vad)
                current_start += MAX_SEGMENT_LENGTH
                part_index += 1

            if segment_end - current_start > 0:
                new_vad = {
                    "start": current_start,
                    "end": segment_end,
                    "text": text_parts[part_index]
                    if part_index < len(text_parts)
                    else "",
                }
                final_list.append(new_vad)
        elif segment_duration >= MIN_SEGMENT_LENGTH:
            final_list.append(vad)

    return final_list


def number_to_chinese(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return text


def add_spaces_around_chinese(text):
    # 定义正则表达式模式，用于匹配中文字符
    pattern = re.compile(r"([\u4e00-\u9fff])")

    # 使用正则表达式，在匹配到的中文字符前后插入空格
    modified_text = pattern.sub(r" \1 ", text)

    # 去除多余的空格，将连续的多个空格替换为一个空格，并去除两端的空格
    modified_text = re.sub(r"\s+", " ", modified_text).strip()

    return modified_text


def chinese_to_pinyin(text):
    text = re.sub(r"[^\w\s]|_", "", text)

    words = jieba.lcut(text, cut_all=False)
    raw_text = []
    g2p_text = []
    for word in words:
        word = word.strip()
        if word == "":
            continue
        pinyins = lazy_pinyin(word)
        if len(pinyins) == 1:
            raw_text.append(word)
            g2p_text.append(pinyins[0])
        else:
            word = add_spaces_around_chinese(word).split(" ")
            assert len(pinyins) == len(word), logger.error(word, pinyins)
            for _pinyins, _word in zip(pinyins, word):
                raw_text.append(_word)
                g2p_text.append(_pinyins)
    return raw_text, g2p_text


def chinese_to_ipa(text):
    try:
        text = number_to_chinese(text)
    except:
        text = ""
    raw_text, g2p_text = chinese_to_pinyin(text)
    return raw_text, g2p_text


def asr(asr_model, vad_segments, audio, batch_size=16, multilingual_flag=False):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments of the given audio.

    Args:
        vad_segments (list): List of VAD segments with start and end times.
        audio (dict): A dictionary containing the audio waveform and sample rate.

    Returns:
        list: A list of ASR results with transcriptions and language details.
    """
    if len(vad_segments) == 0:
        return []

    temp_audio = audio["waveform"]
    start_time = vad_segments[0]["start"]
    end_time = vad_segments[-1]["end"]
    start_frame = int(start_time * audio["sample_rate"])
    end_frame = int(end_time * audio["sample_rate"])
    temp_audio = temp_audio[start_frame:end_frame]  # remove silent start and end

    # update vad_segments start and end time (this is a little trick for batched asr:)
    for idx, segment in enumerate(vad_segments):
        vad_segments[idx]["start"] -= start_time
        vad_segments[idx]["end"] -= start_time

    # resample to 16k
    temp_audio = librosa.resample(
        temp_audio, orig_sr=audio["sample_rate"], target_sr=16000
    )

    language, prob = asr_model.detect_language(temp_audio)
    # if language in supported_languages and prob > 0.8:
    transcribe_result = asr_model.transcribe(
        temp_audio,
        vad_segments,
        batch_size=batch_size,
        language=language,
        print_progress=True,
    )
    result = transcribe_result["segments"]
    for idx, segment in enumerate(result):
        result[idx]["start"] += start_time
        result[idx]["end"] += start_time
        result[idx]["language"] = transcribe_result["language"]
        result[idx]["text"], result[idx]["phoneme"] = chinese_to_ipa(
            result[idx]["text"]
        )
    return result


def write_wav(path, sr, x):
    """Write numpy array to WAV file."""
    sf.write(path, x, sr)


def export_to_wav(audio, asr_result, folder_path, file_name):
    """Export segmented audio to WAV files."""
    sr = audio["sample_rate"]
    company = audio["company"]
    audio = audio["waveform"]
    os.makedirs(folder_path, exist_ok=True)
    save_file_path = os.path.join(folder_path, file_name + "_人声.wav")
    save_company_path = os.path.join(folder_path, file_name + "_伴奏.wav")
    write_wav(save_file_path, sr, audio)
    write_wav(save_company_path, sr, company)


def singing_voice_separate(
    source_separater,
    echo_seprater,
    dereverb_model,
    audio_path: str,
    save_path=None,
    audio_name=None,
    save: bool = False,
):
    """
    Process the audio file, including standardization, source separation, speaker segmentation, VAD, ASR, export to MP3, and MOS prediction.

    Args:
        audio_path (str): Audio file path.
        save_path (str, optional): Save path, defaults to None, which means saving in the "_processed" folder in the audio file's directory.
        audio_name (str, optional): Audio file name, defaults to None, which means using the file name from the audio file path.

    Returns:
        tuple: Contains the save path and the MOS list.
    """
    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
        raise Warning(f"Unsupported file type: {audio_path}")

    # for a single audio from path Ïaaa/bbb/ccc.wav ---> save to aaa/bbb_processed/ccc/ccc_0.wav
    audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
    if save_path is not None:
        save_path = save_path
    else:
        save_path = save_path or os.path.join(
            os.path.dirname(audio_path) + "_processed", audio_name
        )
    os.makedirs(save_path, exist_ok=True)
    audio = standardization(audio_path)
    if audio is None:
        raise IOError("Can't read audio, have deleted the audio")

    audio = source_separation(source_separater, audio)
    audio = dereverb(echo_seprater, audio, False)  # False
    audio = dereverb(dereverb_model, audio, True)
    if save:
        export_to_wav(audio, None, save_path, audio_name)
    return audio, save_path, audio_name


def get_audio_files(folder_path):
    """Get all audio files in a folder."""
    audio_files = []
    for files in Path(folder_path).rglob("**/*"):
        files: Path
        p = files.parent
        processed_parent = Path(str(p) + "_processed")
        if (
            processed_parent.exists()
            or "_processed" in str(files)
            or "伴奏" in str(files)
        ):
            continue

        if str(files).endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
            audio_files.append(os.path.join(str(files)))

    return audio_files


def interp_f0(f0, uv=None):
    def denorm_f0(f0, uv, pitch_padding=None):
        f0 = 2 ** f0
        if uv is not None:
            f0[uv > 0] = 0
        if pitch_padding is not None:
            f0[pitch_padding] = 0
        return f0

    if uv is None:
        uv = f0 == 0
    f0 = f0 = np.log2(f0)
    if sum(uv) == len(f0):
        f0[uv] = -np.inf
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def get_pitch_parselmouth(wav_data, hop_size, audio_sample_rate, interp_uv=True):
    time_step = hop_size / audio_sample_rate
    f0_min = 65.0
    f0_max = 1100.0

    # noinspection PyArgumentList
    f0 = (
        parselmouth.Sound(wav_data, sampling_frequency=audio_sample_rate)
        .to_pitch_ac(
            time_step=time_step,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        .selected_array["frequency"]
    )
    uv = f0 == 0
    if interp_uv:
        f0, uv = interp_f0(f0, uv)
    return time_step, f0, uv


resample_transform_dict = {}


def load_wav(path, device, sample_rate=None):
    waveform, sr = torchaudio.load(str(path))
    if sample_rate != sr and sample_rate is not None:
        global resample_transform_dict
        if sr not in resample_transform_dict:
            resample_transform_dict[sr] = torchaudio.transforms.Resample(
                sr, sample_rate
            )

        waveform = resample_transform_dict[sr](waveform)

    waveform = waveform[0].to(device)
    return waveform


def get_pitch(algorithm, wav_data, hop_size, audio_sample_rate, interp_uv=True):
    if algorithm == "parselmouth":
        return get_pitch_parselmouth(
            wav_data, hop_size, audio_sample_rate, interp_uv=interp_uv
        )
    else:
        raise ValueError(f" [x] Unknown f0 extractor: {algorithm}")


def build_dataset(wavs, tg, dataset, skip_silence_insertion=True, wav_subtype="PCM_16"):
    wavs = Path(wavs)
    tg_dir = Path(tg)
    del tg
    dataset = Path(dataset)
    filelist = list(wavs.glob("*.wav"))

    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "wavs").mkdir(exist_ok=True)
    transcriptions = []
    samplerate = 44100
    min_sil = int(0.1 * samplerate)
    max_sil = int(0.5 * samplerate)
    with tqdm.tqdm(
        total=len(filelist),
        desc="Build Dataset",
        file=tqdm_out,
        ascii=True,
        ncols=os.get_terminal_size().columns - 35,
        miniters=1,
    ) as pbar:
        for wavfile in filelist:
            y, _ = librosa.load(wavfile, sr=samplerate, mono=True)
            tgfile = tg_dir / wavfile.with_suffix(".TextGrid").name

            if not tgfile.exists():
                pbar.update(1)
                continue
            tg = TextGrid()
            tg.read(str(tgfile))

            word_seq = [w.mark.strip() for w in tg[0]]
            if "_人声" in str(wavfile):
                ds_path = str(wavfile).replace("_人声.wav", ".txt")
            else:
                ds_path = str(wavfile).replace("wav", "txt")

            with open(ds_path, "r") as f_word:
                text_seq = f_word.readline().strip().split(" ")
            new_text_seq = []

            for idx, w in enumerate(word_seq):
                if w in ["SP", "AP"]:
                    new_text_seq.append(w.strip())
                else:
                    new_text_seq.append(text_seq.pop(0).strip())

            word_dur = [w.maxTime - w.minTime for w in tg[0]]
            ph_seq = [ph.mark for ph in tg[1]]
            ph_dur = [ph.maxTime - ph.minTime for ph in tg[1]]

            if not skip_silence_insertion:
                if random.random() < 0.5:
                    len_sil = random.randrange(min_sil, max_sil)
                    y = np.concatenate((np.zeros((len_sil,), dtype=np.float32), y))
                    if ph_seq[0] == "SP":
                        ph_dur[0] += len_sil / samplerate
                    else:
                        ph_seq.insert(0, "SP")
                        ph_dur.insert(0, len_sil / samplerate)
                if random.random() < 0.5:
                    len_sil = random.randrange(min_sil, max_sil)
                    y = np.concatenate((y, np.zeros((len_sil,), dtype=np.float32)))
                    if ph_seq[-1] == "SP":
                        ph_dur[-1] += len_sil / samplerate
                    else:
                        ph_seq.append("SP")
                        ph_dur.append(len_sil / samplerate)

            new_text_seq = " ".join(new_text_seq)
            word_seq = " ".join(word_seq)
            word_dur = " ".join([str(round(d, 6)) for d in word_dur])
            ph_seq = " ".join(ph_seq)
            ph_dur = " ".join([str(round(d, 6)) for d in ph_dur])
            sf.write(
                dataset / "wavs" / wavfile.name, y, samplerate, subtype=wav_subtype
            )

            transcriptions.append(
                {
                    "name": wavfile.stem,
                    "text_seq": new_text_seq,
                    "word_seq": word_seq,
                    "word_dur": word_dur,
                    "ph_dur": ph_dur,
                    "ph_seq": ph_seq,
                }
            )
            pbar.update(1)

    with open(dataset / "transcriptions.csv", "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "text_seq", "word_seq", "word_dur", "ph_seq", "ph_dur"],
        )
        writer.writeheader()
        writer.writerows(transcriptions)


def add_ph_num(transcription):
    global dictionary
    vowels = {"SP", "AP"}
    word2ph = {
        item.split("\t")[0]: item.split("\t")[1].strip().split(" ")
        for item in dictionary.split("\n")
    }
    consonants = set()
    rules = [item.split("\t")[1].strip().split(" ") for item in dictionary.split("\n")]
    for phonemes in rules:
        assert (
            len(phonemes) <= 2
        ), "We only support two-phase dictionaries for automatically adding ph_num."
        if len(phonemes) == 1:
            vowels.add(phonemes[0])
        else:
            consonants.add(phonemes[0])
            vowels.add(phonemes[1])

    transcription = Path(transcription)
    items: list[dict] = []
    with open(transcription, "r", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for item in reader:
            items.append(item)

    for item in items:
        item: dict
        ph_seq = item["ph_seq"].split()
        word_seq = item["word_seq"].split()

        for ph in ph_seq:
            assert (
                ph in vowels or ph in consonants
            ), f'Invalid phoneme symbol \'{ph}\' in \'{item["name"]}\'.'
        ph_num = []
        i = 0
        while i < len(ph_seq):
            j = i + 1
            while j < len(ph_seq) and ph_seq[j] in consonants:
                j += 1
            ph_num.append(str(j - i))
            i = j
        item["ph_num"] = " ".join(ph_num)
        item["word2ph"] = " ".join([str(len(word2ph[i])) for i in word_seq])

    with open(transcription, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=items[0].keys())
        writer.writeheader()
        writer.writerows(items)


# TODO: modified to `SOME`, and `RMVPE`
def estimate_midi(
    transcriptions, waveforms, pe: str = "parselmouth", rest_uv_ratio: float = 0.85,
):
    transcriptions = Path(transcriptions)
    waveforms = Path(waveforms)
    with open(transcriptions, "r", encoding="utf8") as f:
        reader = csv.DictReader(f)
        items: List[dict] = []
        for item in reader:
            items.append(item)

    timestep = 512 / 44100
    with tqdm.tqdm(
        total=len(items),
        desc="Estimate MIDI",
        file=tqdm_out,
        ascii=True,
        ncols=os.get_terminal_size().columns - 35,
        miniters=1,
    ) as pbar:
        for item in items:
            item: dict
            ph_dur = [float(d) for d in item["ph_dur"].split()]
            ph_num = [int(n) for n in item["ph_num"].split()]
            assert sum(ph_num) == len(
                ph_dur
            ), f'ph_num does not sum to number of phones in \'{item["name"]}\'.'

            word_dur = []
            i = 0
            for num in ph_num:
                word_dur.append(sum(ph_dur[i : i + num]))
                i += num

            total_secs = sum(ph_dur)
            waveform, _ = librosa.load(
                waveforms / (item["name"] + ".wav"), sr=44100, mono=True
            )
            _, f0, uv = get_pitch(pe, waveform, 512, 44100)
            pitch = librosa.hz_to_midi(f0)
            if pitch.shape[0] < total_secs / timestep:
                pad = math.ceil(total_secs / timestep) - pitch.shape[0]
                pitch = np.pad(
                    pitch, [0, pad], mode="constant", constant_values=[0, pitch[-1]]
                )
                uv = np.pad(uv, [0, pad], mode="constant")

            note_seq = []
            note_dur = []
            start = 0.0
            for dur in word_dur:
                end = start + dur
                start_idx = math.floor(start / timestep)
                end_idx = math.ceil(end / timestep)
                word_pitch = pitch[start_idx:end_idx]
                word_uv = uv[start_idx:end_idx]
                word_valid_pitch = np.extract(~word_uv & (word_pitch >= 0), word_pitch)
                if len(word_valid_pitch) < (1 - rest_uv_ratio) * (end_idx - start_idx):
                    note_seq.append("rest")
                else:
                    counts = np.bincount(np.round(word_valid_pitch).astype(np.int64))
                    midi = counts.argmax()
                    midi = np.mean(
                        word_valid_pitch[
                            (word_valid_pitch >= midi - 0.5)
                            & (word_valid_pitch < midi + 0.5)
                        ]
                    )
                    note_seq.append(
                        librosa.midi_to_note(midi, cents=True, unicode=False)
                    )
                note_dur.append(dur)

                start = end

            item["note_seq"] = " ".join(note_seq)
            item["note_dur"] = " ".join([str(round(d, 6)) for d in note_dur])
            pbar.update(1)

    with open(transcriptions, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "text_seq",
                "word_seq",
                "word2ph",
                "word_dur",
                "ph_seq",
                "ph_dur",
                "ph_num",
                "note_seq",
                "note_dur",
            ],
        )
        writer.writeheader()
        writer.writerows(items)


def try_resolve_note_slur_by_matching(ph_dur, ph_num, note_dur, tol):
    if len(ph_num) > len(note_dur):
        raise ValueError("ph_num should not be longer than note_dur.")
    ph_num_cum = np.cumsum([0] + ph_num)
    word_pos = np.cumsum(
        [sum(ph_dur[l:r]) for l, r in zip(ph_num_cum[:-1], ph_num_cum[1:])]
    )
    note_pos = np.cumsum(note_dur)
    new_note_dur = []

    note_slur = []
    idx_word, idx_note = 0, 0
    slur = False
    while idx_word < len(word_pos) and idx_note < len(note_pos):
        if isclose(word_pos[idx_word], note_pos[idx_note], abs_tol=tol):
            note_slur.append(1 if slur else 0)
            new_note_dur.append(word_pos[idx_word])
            idx_word += 1
            idx_note += 1
            slur = False
        elif note_pos[idx_note] > word_pos[idx_word]:
            raise ValueError("Cannot resolve note_slur by matching.")
        elif note_pos[idx_note] <= word_pos[idx_word]:
            note_slur.append(1 if slur else 0)
            new_note_dur.append(note_pos[idx_note])
            idx_note += 1
            slur = True
    ret_note_dur = np.diff(new_note_dur, prepend=Decimal("0.0")).tolist()
    assert len(ret_note_dur) == len(note_slur)
    return ret_note_dur, note_slur


def try_resolve_slur_by_slicing(ph_dur, ph_num, note_seq, note_dur, tol):
    ph_num_cum = np.cumsum([0] + ph_num)
    word_pos = np.cumsum(
        [sum(ph_dur[l:r]) for l, r in zip(ph_num_cum[:-1], ph_num_cum[1:])]
    )
    note_pos = np.cumsum(note_dur)
    new_note_seq = []
    new_note_dur = []

    note_slur = []
    idx_word, idx_note = 0, 0
    while idx_word < len(word_pos):
        slur = False
        if note_pos[idx_note] > word_pos[idx_word] and not isclose(
            note_pos[idx_note], word_pos[idx_word], abs_tol=tol
        ):
            new_note_seq.append(note_seq[idx_note])
            new_note_dur.append(word_pos[idx_word])
            note_slur.append(1 if slur else 0)
        else:
            while idx_note < len(note_pos) and (
                note_pos[idx_note] < word_pos[idx_word]
                or isclose(note_pos[idx_note], word_pos[idx_word], abs_tol=tol)
            ):
                new_note_seq.append(note_seq[idx_note])
                new_note_dur.append(note_pos[idx_note])
                note_slur.append(1 if slur else 0)
                slur = True
                idx_note += 1
            if new_note_dur[-1] < word_pos[idx_word]:
                if isclose(new_note_dur[-1], word_pos[idx_word], abs_tol=tol):
                    new_note_dur[-1] = word_pos[idx_word]
                else:
                    new_note_seq.append(note_seq[idx_note])
                    new_note_dur.append(word_pos[idx_word])
                    note_slur.append(1 if slur else 0)
        idx_word += 1
    ret_note_dur = np.diff(new_note_dur, prepend=Decimal("0.0")).tolist()
    assert len(new_note_seq) == len(ret_note_dur) == len(note_slur)
    return new_note_seq, ret_note_dur, note_slur


def csv2ds(
    transcription_file,
    wavs_folder,
    tolerance=0.005,
    hop_size=512,
    sample_rate=44100,
    pe="parselmouth",
):
    """Convert a transcription file to DS file"""

    wavs_folder = Path(wavs_folder)
    transcription_file = Path(transcription_file)

    assert wavs_folder.is_dir(), "wavs folder not found."
    out_ds = {}
    out_exists = []
    with open(transcription_file, "r", encoding="utf-8") as f:
        pbar = tqdm.tqdm(
            desc="Convert csv to DS",
            file=tqdm_out,
            ascii=True,
            ncols=os.get_terminal_size().columns - 35,
            miniters=1,
        )
        for trans_line in csv.DictReader(f):
            item_name = trans_line["name"]
            wav_fn = wavs_folder / f"{item_name}.wav"
            ds_fn = wavs_folder / f"{item_name}.ds"
            word_dur = list(map(Decimal, trans_line["word_dur"].strip().split()))
            ph_dur = list(map(Decimal, trans_line["ph_dur"].strip().split()))
            ph_num = list(map(int, trans_line["ph_num"].strip().split()))
            note_seq = trans_line["note_seq"].strip().split()
            note_dur = list(map(Decimal, trans_line["note_dur"].strip().split()))
            note_glide = (
                trans_line["note_glide"].strip().split()
                if "note_glide" in trans_line
                else None
            )

            assert wav_fn.is_file(), f"{item_name}.wav not found."
            assert len(ph_dur) == sum(ph_num), "ph_dur and ph_num mismatch."
            assert len(note_seq) == len(
                note_dur
            ), "note_seq and note_dur should have the same length."
            if note_glide:
                assert len(note_glide) == len(
                    note_seq
                ), "note_glide and note_seq should have the same length."
            assert isclose(
                sum(ph_dur), sum(note_dur), abs_tol=tolerance
            ), f"[{item_name}] ERROR: mismatch total duration: {sum(ph_dur) - sum(note_dur)}"

            # Resolve note_slur
            if "note_slur" in trans_line and trans_line["note_slur"]:
                note_slur = list(map(int, trans_line["note_slur"].strip().split()))
            else:
                try:
                    note_dur, note_slur = try_resolve_note_slur_by_matching(
                        ph_dur, ph_num, note_dur, tolerance
                    )
                except ValueError:
                    # logging.warning(f"note_slur is not resolved by matching for {item_name}")
                    note_seq, note_dur, note_slur = try_resolve_slur_by_slicing(
                        ph_dur, ph_num, note_seq, note_dur, tolerance
                    )
            # Extract f0_seq
            wav, _ = librosa.load(wav_fn, sr=sample_rate, mono=True)
            # length = len(wav) + (win_size - hop_size) // 2 + (win_size - hop_size + 1) // 2
            # length = ceil((length - win_size) / hop_size)
            f0_timestep, f0, _ = get_pitch(pe, wav, hop_size, sample_rate)
            ds_content = [
                {
                    "offset": 0.0,
                    "text": trans_line["text_seq"],
                    "word_seq": trans_line["word_seq"],
                    "word2ph": trans_line["word2ph"],
                    "word_dur": " ".join(str(round(d, 6)) for d in word_dur),
                    "ph_seq": trans_line["ph_seq"],
                    "ph_dur": " ".join(str(round(d, 6)) for d in ph_dur),
                    "ph_num": trans_line["ph_num"],
                    "note_seq": " ".join(note_seq),
                    "note_dur": " ".join(str(round(d, 6)) for d in note_dur),
                    "note_slur": " ".join(map(str, note_slur)),
                    "f0_seq": " ".join(map("{:.1f}".format, f0)),
                    "f0_timestep": str(f0_timestep),
                }
            ]
            if note_glide:
                ds_content[0]["note_glide"] = " ".join(note_glide)
            out_ds[ds_fn] = ds_content
            if ds_fn.exists():
                out_exists.append(ds_fn)
            pbar.update(1)
    for ds_fn, ds_content in out_ds.items():
        with open(ds_fn, "w", encoding="utf-8") as f:
            json.dump(ds_content, f, ensure_ascii=False, indent=4)


def enhance_tg(
    wavs,
    src,
    dst,
    f0_min=40.0,
    f0_max=1300.0,
    br_len=0.1,
    br_db=-60,
    br_centroid=2000,
    time_step=0.005,
    min_space=0.04,
    voicing_thresh_vowel=0.45,
    voicing_thresh_breath=0.6,
    br_win_sz=0.05,
):
    global dictionary
    wavs = Path(wavs)
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    rules = [item.strip().split("\t") for item in dictionary.split("\n")]

    dictionary_entg = {}
    phoneme_set = set()
    for r in rules:
        phonemes = r[1].split()
        dictionary_entg[r[0]] = phonemes
        phoneme_set.update(phonemes)

    filelist = list(wavs.glob("*.wav"))
    with tqdm.tqdm(
        total=len(filelist),
        desc="Enhance Textgrid",
        file=tqdm_out,
        ascii=True,
        ncols=os.get_terminal_size().columns - 35,
        miniters=1,
    ) as pbar:
        for wavfile in filelist:
            tgfile = src / wavfile.with_suffix(".TextGrid").name
            if not tgfile.exists():
                pbar.update(1)
                continue

            textgrid = TextGrid()
            textgrid.read(str(tgfile))
            words = textgrid[0]
            phones = textgrid[1]
            sound = parselmouth.Sound(str(wavfile))
            f0_voicing_breath = sound.to_pitch_ac(
                time_step=time_step,
                voicing_threshold=voicing_thresh_breath,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            ).selected_array["frequency"]
            f0_voicing_vowel = sound.to_pitch_ac(
                time_step=time_step,
                voicing_threshold=voicing_thresh_vowel,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            ).selected_array["frequency"]
            y, sr = librosa.load(wavfile, sr=24000, mono=True)
            hop_size = int(time_step * sr)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=2048, hop_length=hop_size
            ).squeeze(0)

            # Fix long utterances
            i = j = 0
            while i < len(words):
                word = words[i]
                phone = phones[j]
                if word.mark is not None and word.mark != "":
                    i += 1
                    j += len(dictionary_entg[word.mark])
                    continue
                if i == 0:
                    i += 1
                    j += 1
                    continue
                prev_word = words[i - 1]
                prev_phone = phones[j - 1]
                # Extend length of long utterances
                while word.minTime < word.maxTime - time_step:
                    pos = min(
                        f0_voicing_vowel.shape[0] - 1, int(word.minTime / time_step)
                    )
                    if f0_voicing_vowel[pos] < f0_min:
                        break
                    prev_word.maxTime += time_step
                    prev_phone.maxTime += time_step
                    word.minTime += time_step
                    phone.minTime += time_step
                i += 1
                j += 1

            # Detect aspiration
            i = j = 0
            while i < len(words):
                word = words[i]
                phone = phones[j]
                if word.mark is not None and word.mark != "":
                    i += 1
                    j += len(dictionary_entg[word.mark])
                    continue
                if word.maxTime - word.minTime < br_len:
                    i += 1
                    j += 1
                    continue
                ap_ranges = []
                br_start = None
                win_pos = word.minTime
                while win_pos + br_win_sz <= word.maxTime:
                    all_noisy = (
                        f0_voicing_breath[
                            int(win_pos / time_step) : int(
                                (win_pos + br_win_sz) / time_step
                            )
                        ]
                        < f0_min
                    ).all()
                    rms_db = 20 * np.log10(
                        np.clip(
                            sound.get_rms(
                                from_time=win_pos, to_time=win_pos + br_win_sz
                            ),
                            a_min=1e-12,
                            a_max=1,
                        )
                    )
                    # print(win_pos, win_pos + br_win_sz, all_noisy, rms_db)
                    if all_noisy and rms_db >= br_db:
                        if br_start is None:
                            br_start = win_pos
                    else:
                        if br_start is not None:
                            br_end = win_pos + br_win_sz - time_step
                            if br_end - br_start >= br_len:
                                centroid = spectral_centroid[
                                    int(br_start / time_step) : int(br_end / time_step)
                                ].mean()
                                if centroid >= br_centroid:
                                    ap_ranges.append((br_start, br_end))
                            br_start = None
                            win_pos = br_end
                    win_pos += time_step
                if br_start is not None:
                    br_end = win_pos + br_win_sz - time_step
                    if br_end - br_start >= br_len:
                        centroid = spectral_centroid[
                            int(br_start / time_step) : int(br_end / time_step)
                        ].mean()
                        if centroid >= br_centroid:
                            ap_ranges.append((br_start, br_end))
                # print(ap_ranges)
                if len(ap_ranges) == 0:
                    i += 1
                    j += 1
                    continue
                words.removeInterval(word)
                phones.removeInterval(phone)
                if word.minTime < ap_ranges[0][0]:
                    words.add(minTime=word.minTime, maxTime=ap_ranges[0][0], mark=None)
                    phones.add(
                        minTime=phone.minTime, maxTime=ap_ranges[0][0], mark=None
                    )
                    i += 1
                    j += 1
                for k, ap in enumerate(ap_ranges):
                    if k > 0:
                        words.add(minTime=ap_ranges[k - 1][1], maxTime=ap[0], mark=None)
                        phones.add(
                            minTime=ap_ranges[k - 1][1], maxTime=ap[0], mark=None
                        )
                        i += 1
                        j += 1
                    words.add(
                        minTime=ap[0], maxTime=min(word.maxTime, ap[1]), mark="AP"
                    )
                    phones.add(
                        minTime=ap[0], maxTime=min(word.maxTime, ap[1]), mark="AP"
                    )
                    i += 1
                    j += 1
                if ap_ranges[-1][1] < word.maxTime:
                    words.add(minTime=ap_ranges[-1][1], maxTime=word.maxTime, mark=None)
                    phones.add(
                        minTime=ap_ranges[-1][1], maxTime=phone.maxTime, mark=None
                    )
                    i += 1
                    j += 1

            # Remove short spaces
            i = j = 0
            while i < len(words):
                word = words[i]
                phone = phones[j]
                if word.mark is not None and word.mark != "":
                    i += 1
                    j += 1 if word.mark == "AP" else len(dictionary_entg[word.mark])
                    continue
                if word.maxTime - word.minTime >= min_space:
                    word.mark = "SP"
                    phone.mark = "SP"
                    i += 1
                    j += 1
                    continue
                if i == 0:
                    if len(words) >= 2:
                        words[i + 1].minTime = word.minTime
                        phones[j + 1].minTime = phone.minTime
                        words.removeInterval(word)
                        phones.removeInterval(phone)
                    else:
                        break
                elif i == len(words) - 1:
                    if len(words) >= 2:
                        words[i - 1].maxTime = word.maxTime
                        phones[j - 1].maxTime = phone.maxTime
                        words.removeInterval(word)
                        phones.removeInterval(phone)
                    else:
                        break
                else:
                    words[i - 1].maxTime = words[i + 1].minTime = (
                        word.minTime + word.maxTime
                    ) / 2
                    phones[j - 1].maxTime = phones[j + 1].minTime = (
                        phone.minTime + phone.maxTime
                    ) / 2
                    words.removeInterval(word)
                    phones.removeInterval(phone)
            textgrid.write(str(dst / tgfile.name))
            pbar.update(1)


def postprocess(
    song_and_lyrics_path, align_out_path, dataset_path,
):
    enhance_tg(
        song_and_lyrics_path,
        f"{align_out_path}/TextGrid",
        f"{align_out_path}/TextGrid_entg",
    )
    build_dataset(song_and_lyrics_path, f"{align_out_path}/TextGrid_entg", dataset_path)
    add_ph_num(f"{dataset_path}/transcriptions.csv")
    estimate_midi(f"{dataset_path}/transcriptions.csv", f"{dataset_path}/wavs")
    csv2ds(f"{dataset_path}/transcriptions.csv", f"{dataset_path}/wavs")
    shutil.rmtree(f"{dataset_path}/transcriptions.csv", ignore_errors=True)


def get_sorted_files_with_prefix(folder_path, pattern="*.ds"):
    files = glob.glob(os.path.join(folder_path, pattern))

    if not files:
        return None, []

    first_file = os.path.basename(files[0])
    if "_人声" in first_file:
        prefix_match = re.match(r"^(.+?)_\d+\_人声.ds$", first_file)
    else:
        prefix_match = re.match(r"^(.+?)_\d+\.ds$", first_file)
    if not prefix_match:
        return None, []

    prefix = prefix_match.group(1)

    def extract_number(file_path):
        if "_人声" in first_file:
            match = re.search(r"_(\d+)\_人声.ds$", file_path)
        else:
            match = re.search(r"_(\d+)\.ds$", file_path)
        if match:
            return int(match.group(1))
        return 0

    sorted_files = sorted(files, key=extract_number)

    return prefix, sorted_files


#########################################################
#                       Core CLASS                      #
#########################################################


class VocalSeparate:
    def __init__(
        self, separate_model_path: str = "checkpoints/step1/separate_model.pt",
    ):
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"
        device = torch.device(device_name)
        self.device = device

        separate_state_dict = torch.load(separate_model_path, map_location="cpu")
        self.source_separater = Predictor(
            separate_state_dict["src_separate"], device=device_name
        )
        deecho_ckpt = separate_state_dict["deecho"]
        dereverb_ckpt = separate_state_dict["dereverb"]

        self.echo_seprater = Dereverb(
            model_name=deecho_ckpt["name"],
            model_state_dict=deecho_ckpt["state_dict"],
            device=device_name,
        )

        self.dereverb_model = Dereverb(
            model_name=dereverb_ckpt["name"],
            model_state_dict=dereverb_ckpt["state_dict"],
            device=device_name,
        )

    def process(
        self, input_folder_path, out_path,
    ):
        # Input Folder Path
        input_folder_path = Path(input_folder_path)
        if input_folder_path.is_dir():
            audio_paths = get_audio_files(input_folder_path)
        elif input_folder_path.is_file():
            audio_paths = [str(input_folder_path).strip()]
        else:
            raise IOError("Please check your input files, It's must be folder or file.")

        # Song and Lyrics Path, save audio segment and phoneme
        out_path = Path(out_path)
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)

        max_retries = 5
        for path in audio_paths:
            attempt = 0
            while attempt < max_retries:
                try:
                    clear_audio, save_path, save_audio_name = singing_voice_separate(
                        self.source_separater,
                        self.echo_seprater,
                        self.dereverb_model,
                        path,
                        out_path,
                        False,
                    )
                    break
                except Exception as e:
                    attempt += 1
                    if attempt == max_retries:
                        raise RuntimeError(f"操作失败，经过 {max_retries} 次尝试") from e
                    logger.warning(f"尝试 {attempt} 次，失败 错误: {str(e)}")

            clear_audio["waveform"] = clear_audio["waveform"][:, 0]
            export_to_wav(clear_audio, None, save_path, save_audio_name)


class SOFA:
    r""" MFA Align 
    `song_and_lyrics_path`: 放的是wav,lab,txt 其中 lab是拼音、txt是文字
    `align_out_path`: 是SOFA对齐后的结果
    `mode`=txt 时候自动执行G2P并对齐, =lab时 自动对齐
    """

    def __init__(self, model_path):
        global dictionary
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('high')
        self.align_model = LitForcedAlignmentTask.load_from_checkpoint(model_path)
        self.grapheme_to_phoneme = DictionaryG2P(dictionary)
        self.get_AP = LoudnessSpectralcentroidAPDetector()
        self.align_trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=True)

    @torch.no_grad()
    def process(
        self,
        song_and_lyrics_path,
        align_out_path,
        mode: Literal["force", "match"] = "force",
        out_formats="textgrid,trans",
        in_format: Literal["txt", "lab"] = "lab",
        save_confidence=False,
    ):

        song_and_lyrics_path = Path(song_and_lyrics_path)
        out_formats = [i.strip().lower() for i in out_formats.split(",")]
        
        self.align_model.set_inference_mode(mode)

        self.grapheme_to_phoneme.set_in_format(in_format)

        dataset = self.grapheme_to_phoneme.get_dataset(
            song_and_lyrics_path.rglob("*.wav")
        )
        with tqdm_logger():
            predictions = self.align_trainer.predict(
                self.align_model, dataloaders=dataset, return_predictions=True
            )

        predictions = self.get_AP.process(predictions)
        predictions, log = post_processing(predictions)
        exporter = Exporter(predictions, log, align_out_path)
        if save_confidence:
            out_formats.append("confidence")
        exporter.export(out_formats)


class Proofreading:
    def __init__(self, align_model_path,):
        self.sofa = SOFA(align_model_path)

    def process(self, dataset_path, temp_save_path):
        """ Check insert data and align all data."""
        global DS_KEYS
        temp_save_path = Path(temp_save_path)
        temp_save_path.mkdir(parents=True, exist_ok=True)

        dataset_path = Path(dataset_path)
        dataset_path_files = list(dataset_path.rglob("*.ds"))

        for i in tqdm.tqdm(
            dataset_path_files, total=len(dataset_path_files), desc="Check Files..."
        ):
            assert Path(str(i).replace("ds", "wav")).exists()

        for ds_file in dataset_path_files:
            file_name = ds_file.stem

            with open(ds_file, "r") as f_ds:
                ds_data = json.load(f_ds)

                audio = standardization(str(ds_file.with_suffix(".wav")))

                offset = {}
                for ds_idx, ds_dict in enumerate(ds_data):
                    insert_data_keys = ds_dict.keys()
                    offset[ds_idx] = {}
                    offset[ds_idx]["start"] = ds_dict["offset"]
                    offset[ds_idx]["end"] = ds_dict["offset_end"]
                    if len(insert_data_keys) != len(DS_KEYS):
                        # check insert
                        assert (
                            "offset" in insert_data_keys
                            and "offset_end" in insert_data_keys
                            and "text" in insert_data_keys
                        ), f"""
                        Please check your new data dict keys, keys should be consist of `offset`, `offset_end` and `text`.
                        """
                        start = int(ds_dict["offset"] * audio["sample_rate"])
                        end = int(ds_dict["offset_end"] * audio["sample_rate"])
                        sf.write(
                            temp_save_path / f"{file_name}_{ds_idx}.wav",
                            audio["waveform"][start:end],
                            audio["sample_rate"],
                        )

                        clear_text, clear_phoneme = chinese_to_ipa(ds_dict["text"])

                    else:
                        # check modified
                        start = int(ds_dict["offset"] * audio["sample_rate"])
                        end = int(ds_dict["offset_end"] * audio["sample_rate"])
                        sf.write(
                            temp_save_path / f"{file_name}_{ds_idx}.wav",
                            audio["waveform"][start:end],
                            audio["sample_rate"],
                        )
                        words = ds_dict["text"].split()
                        filtered_words = [
                            word for word in words if word not in ["SP", "AP"]
                        ]
                        clear_text, clear_phoneme = chinese_to_ipa(
                            " ".join(filtered_words)
                        )
                    clear_text = " ".join(clear_text)
                    clear_phoneme = " ".join(clear_phoneme)
                    with open(
                        temp_save_path / f"{file_name}_{ds_idx}.lab", "w"
                    ) as f_lab:
                        f_lab.write(clear_phoneme)
                    with open(
                        temp_save_path / f"{file_name}_{ds_idx}.txt", "w"
                    ) as f_txt:
                        f_txt.write(clear_text)

            self.sofa.process(temp_save_path, temp_save_path / "align")
            postprocess(
                temp_save_path, temp_save_path / "align", temp_save_path / "dataset"
            )

            dataset_save_path = temp_save_path / "dataset" / "wavs"

            # Combine ds
            prefix, sorted_ds_files = get_sorted_files_with_prefix(
                dataset_save_path, "*.ds"
            )
            for i in dataset_save_path.glob("*.wav"):
                os.remove(i)

            for ds in sorted_ds_files:
                _idx = int(re.match(r"^.+?_(\d+)+\.ds$", ds).group(1))
                with open(ds, "r") as f_ds:
                    data = json.load(f_ds)[0]
                    data["offset"] = offset[_idx]["start"]
                    data["offset_end"] = offset[_idx]["end"]
                    ds_data[_idx] = data

            with open(
                dataset_path / f"{prefix.split('.')[0]}_proofread.ds",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(ds_data, f, indent=4, ensure_ascii=False)
        shutil.rmtree(temp_save_path, ignore_errors=True)


class LyricsProcessor:
    def __init__(self):
        pass

    @staticmethod
    def end_silence_detection(y, sr, threshold_db=-35, frame_size=2048, hop_length=512):
        """末尾静音检测"""

        # 计算短时能量
        frame_energy = []
        for i in range(0, len(y) - frame_size, hop_length):
            frame = y[i : i + frame_size]
            energy = np.sum(frame ** 2)
            energy_db = 10 * np.log10(energy + 1e-10)
            frame_energy.append(energy_db)

        # 从末尾查找静音开始点
        silence_threshold = threshold_db
        silence_start_frame = None

        for i in range(len(frame_energy) - 1, -1, -1):
            if frame_energy[i] > silence_threshold:
                silence_start_frame = i + 1
                break

        if silence_start_frame is not None and silence_start_frame < len(frame_energy):
            # 转换为时间戳
            silence_start_time = silence_start_frame * hop_length / sr
            assert silence_start_time < len(y) / sr

            return silence_start_time

        return None

    @staticmethod
    def detect_file_encoding(file_path):
        r"""检测文件的编码格式"""
        with open(file_path, "rb") as f:
            raw_data = f.read(1024)  # 读取前1024字节用于检测编码
        result = detect(raw_data)
        return result["encoding"]

    def check_and_clean_lrc(self, input_file, output_file):
        """
        检查并清理 LRC 歌词文件
        
        参数:
            input_file: 输入的 LRC 文件路径
            output_file: 清理后的输出文件路径
        """
        # 定义关键词列表（可能不是真正歌词的行）
        KEYWORDS = [
            "编曲",
            "词",
            "曲",
            "作词",
            "作曲",
            "吉他",
            "贝斯",
            "鼓",
            "键盘",
            "和声",
            "混音",
            "母带",
            "制作",
            "录音",
            "监制",
            "制作人",
            "出品",
        ]

        # 定义英文检测正则表达式
        english_pattern = re.compile(r"[a-zA-Z]")

        try:
            # 自动检测文件编码
            try:
                encoding = self.detect_file_encoding(input_file)
                logging.info(f"检测到文件编码: {encoding}")
            except Exception as e:
                logging.warning(
                    f"无法自动检测文件编码，将尝试UTF-8和GB2312 - {str(e)}", RuntimeWarning
                )
                encoding = "utf-8"

            # 尝试用检测到的编码打开文件
            try:
                with codecs.open(input_file, "r", encoding=encoding) as infile:
                    # 测试读取几行确保编码正确
                    for i, line in enumerate(infile):
                        if i > 5:  # 只测试前几行
                            break
            except UnicodeDecodeError:
                # 如果检测到的编码失败，尝试备选编码
                alt_encoding = "gb2312" if encoding.lower() == "utf-8" else "utf-8"
                logging.warning(f"使用{encoding}编码读取失败，尝试{alt_encoding}编码")
                try:
                    with codecs.open(input_file, "r", encoding=alt_encoding) as infile:
                        for i, line in enumerate(infile):
                            if i > 5:
                                break
                    encoding = alt_encoding
                except UnicodeDecodeError:
                    raise ValueError(f"无法用{encoding}或{alt_encoding}编码读取文件")

            with codecs.open(input_file, "r", encoding=encoding) as infile, codecs.open(
                output_file, "w", encoding="utf-8"
            ) as outfile:
                new_lrc_file = []
                for line_num, line in enumerate(infile, 1):
                    original_line = line.strip()
                    cleaned_line = original_line

                    # 跳过空行
                    if not original_line:
                        continue

                    # 检查时间戳格式（[mm:ss.xx] 或 [mm:ss]）
                    timestamp_match = re.match(
                        r"^(\[\d{2}:\d{2}(?:\.\d{2,3})?\])(.*)$", original_line
                    )

                    if not timestamp_match:
                        logging.warning(f"(行 {line_num}): 不符合时间戳格式 - {original_line}")
                        continue

                    timestamp = timestamp_match.group(1)
                    lyric_part = timestamp_match.group(2).strip()

                    # 检查并清理括号及其内容
                    if "(" in lyric_part or ")" in lyric_part:
                        cleaned_lyric = re.sub(r"\([^)]*\)", "", lyric_part)
                        logging.warning(
                            f"(行 {line_num}): 发现并移除了括号内容 - {lyric_part} -> {cleaned_lyric}"
                        )
                        lyric_part = cleaned_lyric

                    # 检查特殊符号（保留常见中文标点）
                    if "—" in lyric_part or "-" in lyric_part:  # 如果包含中文破折号，跳过清理
                        continue
                    special_chars = re.sub(
                        r'[\w\s\u4e00-\u9fff，。？！、；："“”‘’_—…]', "", lyric_part
                    )
                    if special_chars:
                        cleaned_lyric = re.sub(
                            r'[^\w\s\u4e00-\u9fff，。？！、；："“”‘’_—…]', "", lyric_part
                        )
                        logging.warning(
                            f"(行 {line_num}): 发现并移除了特殊字符 - {lyric_part} -> {cleaned_lyric}"
                        )
                        lyric_part = cleaned_lyric

                    # 检查英文
                    if english_pattern.search(lyric_part):
                        logging.error(
                            f"(行 {line_num}): 本工具只能处理纯中文歌词，发现英文字符 - {lyric_part}"
                        )
                        continue

                    # 检查关键词（可能的歌曲信息而非歌词）
                    FOUND_KEYWORDS = False
                    for keyword in KEYWORDS:
                        if keyword in lyric_part:
                            # 检查是否包含中文/英文冒号或短横线
                            if (
                                ":" in lyric_part
                                or "：" in lyric_part
                                or "-" in lyric_part
                                or "－" in lyric_part
                            ):
                                logging.warning(
                                    f"(行 {line_num}): 可能发现歌曲信息而非歌词 - {lyric_part}"
                                )
                                FOUND_KEYWORDS = True
                                break  # 跳过当前行，继续检查下一行
                    if FOUND_KEYWORDS:
                        continue

                    # 写入清理后的行
                    cleaned_line = f"{timestamp} {lyric_part}\n"
                    new_lrc_file.append(cleaned_line)
                    outfile.write(cleaned_line)
                return new_lrc_file

        except FileNotFoundError:
            logging.error(f"文件 {input_file} 未找到")
            sys.exit(1)
        except UnicodeDecodeError:
            logging.error(f"无法用检测到的编码({encoding})读取文件 {input_file}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"发生错误: {str(e)}")
            sys.exit(1)

    @staticmethod
    def timestamp_to_seconds(timestamp: str) -> float:
        # 匹配时间格式 [MM:SS.MMM]
        match = re.match(r"\[(\d+):(\d+)\.(\d+)\]", timestamp)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            milliseconds = int(match.group(3).ljust(3, "0")[:3])  # 确保毫秒是3位
            return minutes * 60 + seconds + milliseconds / 1000
        return 0.0

    def parse_lrc_to_dict(
        self, lrc_text: list, audio_length_sec: float
    ) -> List[Dict[str, float | str]]:
        """解析 LRC 歌词为字典列表"""
        lyrics = []
        for i, line in enumerate(lrc_text):
            if not line.strip():
                continue
            match = re.match(r"(\[\d+:\d+\.\d+\])(.*)", line)
            if match:
                timestamp = match.group(1)
                text = match.group(2).strip()
                start = self.timestamp_to_seconds(timestamp)
                # 如果不是最后一行，end 是下一行的 start；否则假设 end 是 start + 1
                if i < len(lrc_text) - 1 and re.match(
                    r"(\[\d+:\d+\.\d+\])(.*)", lrc_text[i + 1]
                ):
                    next_timestamp = re.match(
                        r"(\[\d+:\d+\.\d+\])(.*)", lrc_text[i + 1]
                    ).group(1)
                    end = self.timestamp_to_seconds(next_timestamp)
                else:

                    end = (
                        audio_length_sec
                        if start + 20.0 > audio_length_sec
                        else start + 20.0
                    )

                lyrics.append({"start": start, "end": end, "text": text})
        return lyrics


class SongEdit:
    r"""
    A professional toolchain for AI-powered vocal editing and synthesis.
    
    NOTE: 
        1. The model may occasionally fail during vocal separation. 
           In such cases, it will automatically retry (max 5 attempts).
    
    TODO: 
        1. Implement RMVPE for F0 extraction
           - Current synthesis relies on input song's F0 contour using ParselMouth,
             which has limited robustness compared to modern extractors
        2. [Add other planned improvements here]
    """

    def __init__(
        self,
        separate_model_path: str = "checkpoints/step1/separate_model.pt",
        asr_model_path: str = "checkpoints/whisper",
        align_model_path: str = "checkpoints/sofa/align.ckpt",
        spk_dia_model_path: str = "checkpoints/pyannote",
        vad_model_path: str = "checkpoints/silero-vad"
    ):

        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"
        device = torch.device(device_name)
        self.device = device

        logger.info("Load checkpoints...")
        separate_state_dict = torch.load(separate_model_path, map_location="cpu")
        self.source_separater = Predictor(
            separate_state_dict["src_separate"], device=device_name
        )
        deecho_ckpt = separate_state_dict["deecho"]
        dereverb_ckpt = separate_state_dict["dereverb"]

        self.echo_seprater = Dereverb(
            model_name=deecho_ckpt["name"],
            model_state_dict=deecho_ckpt["state_dict"],
            device=device_name,
        )

        self.dereverb_model = Dereverb(
            model_name=dereverb_ckpt["name"],
            model_state_dict=dereverb_ckpt["state_dict"],
            device=device_name,
        )

        self.dia_pipeline = DiaPipeline.from_pretrained(
            checkpoint_path=spk_dia_model_path
        ).to(device)

        self.asr_model = load_asr_model(
            asr_model_path,
            device_name,
            compute_type="float16",
            threads=4,
            asr_options={
                "initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음."
            },
        )
        self.lrc_processor = LyricsProcessor()
        self.vad_model = SileroVAD(vad_model_path, device=device)
        self.align_model = SOFA(align_model_path)
        logger.info("Load checkpoints successful.")

    def process(
        self,
        input_folder_path,
        dataset_path,
        align_out_path,
        song_and_lyrics_path,
        *,
        mode: Literal["force", "match"] = "force",
        out_formats="textgrid,trans",
        in_format="lab",
        save_confidence=False,
    ):
        # Input Folder Path
        input_folder_path = Path(input_folder_path)
        if input_folder_path.is_dir():
            audio_paths = get_audio_files(input_folder_path)
        elif input_folder_path.is_file():
            audio_paths = [str(input_folder_path).strip()]
        else:
            raise IOError("Please check your input files, It's must be folder or file.")

        # Song and Lyrics Path, save audio segment and phoneme
        song_and_lyrics_path = Path(song_and_lyrics_path)
        if not song_and_lyrics_path.exists():
            song_and_lyrics_path.mkdir(parents=True, exist_ok=True)
        else:
            for i in song_and_lyrics_path.rglob("*"):
                shutil.rmtree(i, ignore_errors=True)

        # MFA Align save Path
        align_out_path = Path(align_out_path)
        if not align_out_path.exists():
            align_out_path.mkdir(parents=True, exist_ok=True)
        else:
            for i in align_out_path.rglob("*"):
                shutil.rmtree(i, ignore_errors=True)

        # Dataset Path
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True, exist_ok=True)
        else:
            for i in dataset_path.rglob("*"):
                shutil.rmtree(i, ignore_errors=True)

        max_retries = 5
        for path in audio_paths:

            # [BUG]: 去混响存在概率出现问题 `Audio buffer is not finite everywhere`
            attempt = 0

            while attempt < max_retries:
                try:
                    clear_audio, save_path, save_audio_name = singing_voice_separate(
                        self.source_separater,
                        self.echo_seprater,
                        self.dereverb_model,
                        path,
                        song_and_lyrics_path,
                        False,
                    )
                    break
                except Exception as e:
                    attempt += 1
                    if attempt == max_retries:
                        raise RuntimeError(f"操作失败，经过 {max_retries} 次尝试") from e
                    logger.warning(f"尝试 {attempt} 次，失败 错误: {str(e)}")

            clear_audio["waveform"] = clear_audio["waveform"][:, 0]

            final_audio = {k: v for k, v in clear_audio.items()}

            #############################################
            #    2.1 Coarse VAD & Lyric Recognition     #
            #    2.2 Singing Voice Align                #
            #############################################

            # Step 2.1 Coarse VAD & Lyric Recognition
            speakerdia = speaker_diarization(
                self.dia_pipeline, clear_audio, self.device
            )

            vad_list = self.vad_model.vad(speakerdia, clear_audio)
            segment_list = cut_vad_segments(vad_list)
            asr_result = asr(self.asr_model, segment_list, clear_audio)
            # save phoneme to `lab` file, and cut waveform.
            sr = clear_audio["sample_rate"]
            temp_audio = clear_audio["waveform"]
            temp_company = clear_audio["company"]

            offset = {}
            for idx, segment in enumerate(asr_result):

                # save phoneme
                with open(
                    os.path.join(save_path, save_audio_name + f"_{idx}.lab"), "w"
                ) as f_ph:
                    f_ph.write(" ".join(segment["phoneme"]))

                # save word level lyrics
                with open(
                    os.path.join(save_path, save_audio_name + f"_{idx}.txt"), "w"
                ) as f_word:
                    f_word.write(" ".join(segment["text"]))

                start, end = int(segment["start"] * sr), int(segment["end"] * sr)
                clear_audio["waveform"] = temp_audio[start:end]
                clear_audio["company"] = temp_company[start:end]
                export_to_wav(clear_audio, None, save_path, save_audio_name + f"_{idx}")
                offset[idx] = {}
                offset[idx]["start"] = segment["start"]
                offset[idx]["end"] = segment["end"]

            del clear_audio

            # Step 2.2 Singing Voice Align
            self.align_model.process(
                song_and_lyrics_path,
                align_out_path,
                mode=mode,
                out_formats=out_formats,
                in_format=in_format,
                save_confidence=save_confidence,
            )

            # Step 2.3 Make DS dataset
            postprocess(song_and_lyrics_path, align_out_path, dataset_path)

            dataset_save_path = Path(dataset_path) / "wavs"

            # remove intermediate results
            shutil.rmtree(align_out_path, ignore_errors=True)
            shutil.rmtree(song_and_lyrics_path, ignore_errors=True)

            # Combine ds
            prefix, sorted_ds_files = get_sorted_files_with_prefix(dataset_save_path)
            for i in dataset_save_path.glob("*.wav"):
                os.remove(i)

            combined_ds = []

            for ds in sorted_ds_files:
                with open(ds, "r") as f_ds:
                    data = json.load(f_ds)[0]
                    combined_ds.append(data)

            assert len(combined_ds) == len(offset)
            for idx, ds in enumerate(combined_ds):
                ds["offset"] = offset[idx]["start"]
                ds["offset_end"] = offset[idx]["end"]

            for i in dataset_save_path.glob("*.ds"):
                os.remove(i)

            with open(dataset_save_path / f"{prefix}.ds", "w", encoding="utf-8") as f:
                json.dump(combined_ds, f, indent=4, ensure_ascii=False)

            export_to_wav(final_audio, None, dataset_save_path, save_audio_name)
            save_file_path = os.path.join(
                dataset_save_path, save_audio_name + "_人声.wav"
            )
            save_company_path = os.path.join(
                dataset_save_path, save_audio_name + "_伴奏.wav"
            )
            logger.info(dataset_save_path / f"{prefix}.ds")
            logger.info(save_file_path)
            logger.info(save_company_path)
        
        return (
            str(dataset_save_path / f"{prefix}.ds"),
            str(save_file_path),
            str(save_company_path),
        )

    def process_lrc(
        self,
        input_folder_path,
        input_lrc_path,
        dataset_path,
        align_out_path,
        song_and_lyrics_path,
        *,
        mode: Literal["force", "match"] = "force",
        out_formats="textgrid,trans",
        in_format="lab",
        save_confidence=False,
    ):

        # Input Folder Path
        input_folder_path = Path(input_folder_path)
        if input_folder_path.is_dir():
            audio_paths = get_audio_files(input_folder_path)
        elif input_folder_path.is_file():
            audio_paths = [str(input_folder_path).strip()]
        else:
            raise IOError("Please check your input files, It's must be folder or file.")

        # Lyric Folder/File Path
        input_lrc_path = Path(input_lrc_path)
        if input_lrc_path.is_dir():
            pass
        elif input_lrc_path.is_file():
            assert input_lrc_path.exists()
            input_lrc_path = input_lrc_path.parent
        else:
            raise IOError("Please check your lyric files, It's must be folder or file.")

        logger.info("Check that each audio file has a corresponding .lrc lyrics file.")
        for a in audio_paths:
            assert (input_lrc_path / f"{Path(a).stem}.lrc").exists(), logger.error(
                "Missing lyrics file for: {input_lrc_path/f'{Path(a).stem}.lrc'}"
            )

        # Song and Lyrics Path, save audio segment and phoneme
        song_and_lyrics_path = Path(song_and_lyrics_path)
        if not song_and_lyrics_path.exists():
            song_and_lyrics_path.mkdir(parents=True, exist_ok=True)
        else:
            for i in song_and_lyrics_path.rglob("*"):
                shutil.rmtree(i, ignore_errors=True)

        # MFA Align save Path
        align_out_path = Path(align_out_path)
        if not align_out_path.exists():
            align_out_path.mkdir(parents=True, exist_ok=True)
        else:
            for i in align_out_path.rglob("*"):
                shutil.rmtree(i, ignore_errors=True)

        # Dataset Path
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True, exist_ok=True)
        else:
            for i in dataset_path.rglob("*"):
                shutil.rmtree(i, ignore_errors=True)

        max_retries = 5
        for path in audio_paths:

            # 检查 LRC 歌词
            logger.info("检查LRC格式...")
            old_lrc_path = input_lrc_path / f"{Path(path).stem}.lrc"
            new_lrc_path = input_lrc_path / f"{Path(path).stem}_correct.lrc"
            new_lrc_file = self.lrc_processor.check_and_clean_lrc(
                old_lrc_path, new_lrc_path
            )

            # [BUG]: 去混响存在概率出现问题 `Audio buffer is not finite everywhere`
            attempt = 0

            while attempt < max_retries:
                try:
                    clear_audio, save_path, save_audio_name = singing_voice_separate(
                        self.source_separater,
                        self.echo_seprater,
                        self.dereverb_model,
                        path,
                        song_and_lyrics_path,
                        False,
                    )
                    break
                except Exception as e:
                    attempt += 1
                    if attempt == max_retries:
                        raise RuntimeError(f"操作失败，经过 {max_retries} 次尝试") from e
                    logger.warning(f"尝试 {attempt} 次，失败 错误: {str(e)}")

            clear_audio["waveform"] = clear_audio["waveform"][:, 0]

            final_audio = {k: v for k, v in clear_audio.items()}

            #############################################
            #    2.1 Coarse VAD & Extract LRC           #
            #    2.2 Singing Voice Align                #
            #############################################

            end_timestamps = self.lrc_processor.end_silence_detection(
                clear_audio["waveform"], clear_audio["sample_rate"]
            )
            lyrics = self.lrc_processor.parse_lrc_to_dict(new_lrc_file, end_timestamps)

            sr = clear_audio["sample_rate"]
            temp_audio = clear_audio["waveform"]
            temp_company = clear_audio["company"]

            filtered_timestamps = []
            for idx, timestamp in enumerate(lyrics):
                _, silence_proportion = calculate_silence_duration(
                    clear_audio["waveform"][
                        int(timestamp["start"] * sr) : int(timestamp["end"] * sr)
                    ],
                    sr,
                )
                if silence_proportion > 0.7:
                    logger.warning(
                        f"删除静音片段: {timestamp['start']:.3f}-{timestamp['end']:.3f}s: '{timestamp['text']}'"
                    )
                else:
                    filtered_timestamps.append(timestamp.copy())

            # merge timestamps
            merged_timestamps = cut_vad_segments_with_text(filtered_timestamps)

            offset = {}
            for idx, segment in enumerate(merged_timestamps):

                segment["text"], segment["phoneme"] = chinese_to_ipa(segment["text"])
                # save phoneme
                with open(
                    os.path.join(save_path, save_audio_name + f"_{idx}.lab"), "w"
                ) as f_ph:
                    f_ph.write(" ".join(segment["phoneme"]))

                # save word level lyrics
                with open(
                    os.path.join(save_path, save_audio_name + f"_{idx}.txt"), "w"
                ) as f_word:
                    f_word.write(" ".join(segment["text"]))

                start, end = int(segment["start"] * sr), int(segment["end"] * sr)
                clear_audio["waveform"] = temp_audio[start:end]
                clear_audio["company"] = temp_company[start:end]
                export_to_wav(clear_audio, None, save_path, save_audio_name + f"_{idx}")
                offset[idx] = {}
                offset[idx]["start"] = segment["start"]
                offset[idx]["end"] = segment["end"]
                
            del clear_audio
            # Step 2.2 Singing Voice Align
            self.align_model.process(
                song_and_lyrics_path,
                align_out_path,
                mode=mode,
                out_formats=out_formats,
                in_format=in_format,
                save_confidence=save_confidence,
            )

            # Step 2.3 Make DS dataset
            postprocess(song_and_lyrics_path, align_out_path, dataset_path)

            dataset_save_path = Path(dataset_path) / "wavs"

            # remove intermediate results
            shutil.rmtree(align_out_path, ignore_errors=True)
            shutil.rmtree(song_and_lyrics_path, ignore_errors=True)

            # Combine ds
            prefix, sorted_ds_files = get_sorted_files_with_prefix(dataset_save_path)
            for i in dataset_save_path.glob("*.wav"):
                os.remove(i)

            combined_ds = []

            for ds in sorted_ds_files:
                with open(ds, "r") as f_ds:
                    data = json.load(f_ds)[0]
                    combined_ds.append(data)

            assert len(combined_ds) == len(offset)
            for idx, ds in enumerate(combined_ds):
                ds["offset"] = offset[idx]["start"]
                ds["offset_end"] = offset[idx]["end"]

            for i in dataset_save_path.glob("*.ds"):
                os.remove(i)

            with open(dataset_save_path / f"{prefix}.ds", "w", encoding="utf-8") as f:
                json.dump(combined_ds, f, indent=4, ensure_ascii=False)

            export_to_wav(final_audio, None, dataset_save_path, save_audio_name)
            save_file_path = os.path.join(
                dataset_save_path, save_audio_name + "_人声.wav"
            )
            save_company_path = os.path.join(
                dataset_save_path, save_audio_name + "_伴奏.wav"
            )
            logger.info(dataset_save_path / f"{prefix}.ds")
            logger.info(save_file_path)
            logger.info(save_company_path)
        return (
            str(dataset_save_path / f"{prefix}.ds"),
            str(save_file_path),
            str(save_company_path),
        )


__all__ = [
    "SongEdit",
    "VocalSeparate",
    "SOFA",
    "Proofreading",
    "postprocess",
]
