#!/usr/bin/env python3
# cython: language_level=3
# Copyright (c) 2025 IACAS. by YangChen

""" 
Step. 2. 歌声合成
"""

import math
import os
import re
import warnings

import cn2an
import jieba
import numpy as np
import torch
from pypinyin import lazy_pinyin

warnings.simplefilter("ignore")
import json
from collections import OrderedDict, deque
from dataclasses import dataclass
from functools import partial
from inspect import isfunction
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import einops
import librosa
import numpy as np
import soundfile as sf
import sox
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from librosa.util import pad_center
from munch import Munch
from pedalboard import (
    Compressor,
    HighpassFilter,
    Limiter,
    LowpassFilter,
    Pedalboard,
    Reverb,
)
from pydub import AudioSegment
from scipy.signal import get_window
from torch import Tensor, nn
from torch.nn.utils import spectral_norm, weight_norm
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WhisperModel

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

SEED_VC_CONFIG = {
    "preprocess_params": {
        "sr": 44100,
        "spect_params": {
            "n_fft": 2048,
            "win_length": 2048,
            "hop_length": 512,
            "n_mels": 128,
            "fmin": 0,
            "fmax": "None",
        },
    },
    "model_params": {
        "dit_type": "DiT",
        "reg_loss_type": "l1",
        "vocoder": {"type": "bigvgan", "name": "nvidia/bigvgan_v2_44khz_128band_512x"},
        "speech_tokenizer": {"type": "whisper", "name": "openai/whisper-small"},
        "style_encoder": {"dim": 192, "campplus_path": "campplus_cn_common.bin"},
        "DAC": {
            "encoder_dim": 64,
            "encoder_rates": [2, 5, 5, 6],
            "decoder_dim": 1536,
            "decoder_rates": [6, 5, 5, 2],
            "sr": 24000,
        },
        "length_regulator": {
            "channels": 768,
            "is_discrete": False,
            "in_channels": 768,
            "content_codebook_size": 2048,
            "sampling_ratios": [1, 1, 1, 1],
            "vector_quantize": False,
            "n_codebooks": 1,
            "quantizer_dropout": 0.0,
            "f0_condition": True,
            "n_f0_bins": 256,
        },
        "DiT": {
            "hidden_dim": 768,
            "num_heads": 12,
            "depth": 17,
            "class_dropout_prob": 0.1,
            "block_size": 8192,
            "in_channels": 128,
            "style_condition": True,
            "final_layer_type": "mlp",
            "target": "mel",
            "content_dim": 768,
            "content_codebook_size": 1024,
            "content_type": "discrete",
            "f0_condition": True,
            "n_f0_bins": 256,
            "content_codebooks": 1,
            "is_causal": False,
            "long_skip_connection": False,
            "zero_prompt_speech_token": False,
            "time_as_token": False,
            "style_as_token": False,
            "uvit_skip_connection": True,
            "add_resblock_in_transformer": False,
        },
        "wavenet": {
            "hidden_dim": 768,
            "num_layers": 8,
            "kernel_size": 5,
            "dilation_rate": 1,
            "p_dropout": 0.2,
            "style_condition": True,
        },
    },
}

DIFFSINGER_CONFIG = {
    "K_step": 400,
    "K_step_infer": 400,
    "accumulate_grad_batches": 1,
    "audio_num_mel_bins": 128,
    "audio_sample_rate": 44100,
    "augmentation_args": {
        "fixed_pitch_shifting": {
            "enabled": False,
            "scale": 0.75,
            "targets": [-5.0, 5.0],
        },
        "random_pitch_shifting": {"enabled": False, "range": [-5.0, 5.0], "scale": 1.0},
        "random_time_stretching": {
            "domain": "log",
            "enabled": False,
            "range": [0.5, 2.0],
            "scale": 1.0,
        },
    },
    "base_config": [],
    "breathiness_smooth_width": 0.12,
    "clip_grad_norm": 1,
    "dataloader_prefetch_factor": 2,
    "diff_accelerator": "ddim",
    "diff_decoder_type": "wavenet",
    "diff_loss_type": "l2",
    "dilation_cycle_length": 4,
    "dropout": 0.1,
    "ds_workers": 4,
    "enc_ffn_kernel_size": 9,
    "enc_layers": 4,
    "energy_smooth_width": 0.12,
    "f0_embed_type": "continuous",
    "ffn_act": "gelu",
    "ffn_padding": "SAME",
    "fft_size": 2048,
    "finetune_ckpt_path": None,
    "finetune_enabled": False,
    "finetune_ignored_params": [
        "model.fs2.encoder.embed_tokens",
        "model.fs2.txt_embed",
        "model.fs2.spk_embed",
    ],
    "finetune_strict_shapes": True,
    "fmax": 16000,
    "fmin": 40,
    "freezing_enabled": False,
    "frozen_params": [],
    "hidden_size": 256,
    "hop_size": 512,
    "interp_uv": True,
    "lambda_aux_mel_loss": 0.2,
    "log_interval": 100,
    "lr_scheduler_args": {
        "gamma": 0.5,
        "scheduler_cls": "torch.optim.lr_scheduler.StepLR",
        "step_size": 50000,
    },
    "max_batch_frames": 80000,
    "max_batch_size": 24,
    "max_beta": 0.02,
    "max_updates": 320000,
    "max_val_batch_frames": 60000,
    "max_val_batch_size": 1,
    "mel_vmax": 1.5,
    "mel_vmin": -6.0,
    "num_ckpt_keep": 5,
    "num_heads": 2,
    "num_pad_tokens": 1,
    "num_sanity_val_steps": 1,
    "num_spk": 1,
    "num_valid_plots": 5,
    "pe": "parselmouth",
    "pe_ckpt": "",
    "permanent_ckpt_interval": 40000,
    "permanent_ckpt_start": 200000,
    "pl_trainer_accelerator": "auto",
    "pl_trainer_devices": "auto",
    "pl_trainer_num_nodes": 1,
    "pl_trainer_precision": "16-mixed",
    "pl_trainer_strategy": "auto",
    "pndm_speedup": 10,
    "raw_data_dir": ["data/opencpop/raw"],
    "rel_pos": True,
    "residual_channels": 512,
    "residual_layers": 20,
    "sampler_frame_count_grid": 6,
    "save_codes": ["configs", "modules", "training", "utils"],
    "schedule_type": "linear",
    "seed": 1234,
    "shallow_diffusion_args": {
        "aux_decoder_arch": "convnext",
        "aux_decoder_args": {
            "dropout_rate": 0.1,
            "kernel_size": 7,
            "num_channels": 512,
            "num_layers": 6,
        },
        "aux_decoder_grad": 0.1,
        "train_aux_decoder": True,
        "train_diffusion": True,
        "val_gt_start": False,
    },
    "sort_by_len": True,
    "speakers": ["opencpop"],
    "spec_max": [0],
    "spec_min": [-5],
    "spk_ids": [],
    "task_cls": "training.acoustic_task.AcousticTask",
    "timesteps": 1000,
    "train_set_name": "train",
    "use_breathiness_embed": False,
    "use_energy_embed": False,
    "use_key_shift_embed": False,
    "use_pos_embed": True,
    "use_shallow_diffusion": False,
    "use_speed_embed": False,
    "use_spk_id": False,
    "val_check_interval": 2000,
    "val_with_vocoder": True,
    "valid_set_name": "valid",
    "vocoder": "NsfHifiGAN",
    "win_size": 2048,
}

BIGVGAN_CONFIG = {
    "resblock": "1",
    "num_gpus": 0,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "adam_b1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.9999996,
    "seed": 1234,
    "upsample_rates": [8, 4, 2, 2, 2, 2],
    "upsample_kernel_sizes": [16, 8, 4, 4, 4, 4],
    "upsample_initial_channel": 1536,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "use_tanh_at_final": False,
    "use_bias_at_final": False,
    "activation": "snakebeta",
    "snake_logscale": True,
    "use_cqtd_instead_of_mrd": True,
    "cqtd_filters": 128,
    "cqtd_max_filters": 1024,
    "cqtd_filters_scale": 1,
    "cqtd_dilations": [1, 2, 4],
    "cqtd_hop_lengths": [512, 256, 256],
    "cqtd_n_octaves": [9, 9, 9],
    "cqtd_bins_per_octaves": [24, 36, 48],
    "mpd_reshapes": [2, 3, 5, 7, 11],
    "use_spectral_norm": False,
    "discriminator_channel_mult": 1,
    "use_multiscale_melloss": True,
    "lambda_melloss": 15,
    "clip_grad_norm": 500,
    "segment_size": 65536,
    "num_mels": 128,
    "num_freq": 2049,
    "n_fft": 2048,
    "hop_size": 512,
    "win_size": 2048,
    "sampling_rate": 44100,
    "fmin": 0,
    "fmax": None,
    "fmax_for_loss": None,
    "normalize_volume": True,
    "num_workers": 4,
}

NSFHIFIGAN_CONFIG = {
    "resblock": "1",
    "num_gpus": 4,
    "batch_size": 10,
    "learning_rate": 0.0002,
    "adam_b1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.999,
    "seed": 1234,
    "upsample_rates": [8, 8, 2, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4, 4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "discriminator_periods": [3, 5, 7, 11, 17, 23, 37],
    "segment_size": 16384,
    "num_mels": 128,
    "num_freq": 1025,
    "n_fft": 2048,
    "hop_size": 512,
    "win_size": 2048,
    "sampling_rate": 44100,
    "fmin": 40,
    "fmax": 16000,
    "fmax_for_loss": None,
    "num_workers": 16,
    "dist_config": {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "world_size": 1,
    },
}

SAMPLERATE = SEED_VC_CONFIG["preprocess_params"]["sr"]
HOP_LENGTH = 512
MAX_CONTEXT_WINDOW = SAMPLERATE // HOP_LENGTH * 30
OVERLAP_FRAME_LEN = 16
OVERLAP_WAVE_LEN = OVERLAP_FRAME_LEN * HOP_LENGTH

MEL_FN_ARGS = {
    "n_fft": SEED_VC_CONFIG["preprocess_params"]["spect_params"]["n_fft"],
    "win_size": SEED_VC_CONFIG["preprocess_params"]["spect_params"]["win_length"],
    "hop_size": SEED_VC_CONFIG["preprocess_params"]["spect_params"]["hop_length"],
    "num_mels": SEED_VC_CONFIG["preprocess_params"]["spect_params"]["n_mels"],
    "sampling_rate": SAMPLERATE,
    "fmin": SEED_VC_CONFIG["preprocess_params"]["spect_params"].get("fmin", 0),
    "fmax": None
    if SEED_VC_CONFIG["preprocess_params"]["spect_params"].get("fmax", "None") == "None"
    else 8000,
    "center": False,
}

CONV_NORMALIZATIONS = frozenset(
    [
        "none",
        "weight_norm",
        "spectral_norm",
        "time_layer_norm",
        "layer_norm",
        "time_group_norm",
    ]
)


# f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


#########################################
#               Function                #
#########################################
def f0_to_coarse(f0, f0_bin):
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.0
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    # torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
    return f0_coarse


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(sampling_rate)}_{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(sampling_rate) + "_" + str(y.device)] = torch.hann_window(
            win_size
        ).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(sampling_rate) + "_" + str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(
        mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)], spec
    )
    spec = spectral_normalize_torch(spec)

    return spec



def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = (
            chunk2[:overlap] * fade_in[: len(chunk2)]
            + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
        )
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError("Unexpected module ({}).".format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


def masked_statistics_pooling(
    x, x_lens, dim=-1, keepdim=False, unbiased=True, eps=1e-2
):
    stats = []
    for i, x_len in enumerate(x_lens):
        x_i = x[i, :, :x_len]
        mean = x_i.mean(dim=dim)
        std = x_i.std(dim=dim, unbiased=unbiased)
        stats.append(torch.cat([mean, std], dim=-1))
    stats = torch.stack(stats, dim=0)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


#########################################
#               DiT Model               #
#########################################


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs
) -> nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """

    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, "b ... t -> b t ...")
        x = super().forward(x)
        x = einops.rearrange(x, "b t ... -> b ... t")
        return


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = self.apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

    @staticmethod
    def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
        assert norm in CONV_NORMALIZATIONS
        if norm == "weight_norm":
            return weight_norm(module)
        elif norm == "spectral_norm":
            return spectral_norm(module)
        else:
            # We already check was in CONV_NORMALIZATION, so any other choice
            # doesn't need reparametrization.
            return module


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`.
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


class SConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        pad_mode: str = "reflect",
        **kwargs,
    ):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "SConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (
            kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )
        if self.causal:
            # Left padding for causal
            x = self.pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = self.pad1d(
                x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )
        return self.conv(x)

    @staticmethod
    def pad1d(
        x: torch.Tensor, paddings: Tuple[int, int], mode: str = "zero", value: float = 0.0
    ):
        """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happen.
        """
        length = x.shape[-1]
        padding_left, padding_right = paddings
        assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
        if mode == "reflect":
            max_pad = max(padding_left, padding_right)
            extra_pad = 0
            if length <= max_pad:
                extra_pad = max_pad - length + 1
                x = F.pad(x, (0, extra_pad))
            padded = F.pad(x, paddings, mode, value)
            end = padded.shape[-1] - extra_pad
            return padded[..., :end]
        else:
            return F.pad(x, paddings, mode, value)

class WN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
        causal=False,
    ):
        super(WN, self).__init__()
        conv1d_type = SConv1d
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = conv1d_type(
                gin_channels, 2 * hidden_channels * n_layers, 1, norm="weight_norm"
            )

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = conv1d_type(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
                norm="weight_norm",
                causal=causal,
            )
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = conv1d_type(
                hidden_channels, res_skip_channels, 1, norm="weight_norm", causal=causal
            )
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = self.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    @staticmethod
    def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
        n_channels_int = n_channels[0]
        in_act = input_a + input_b
        t_act = torch.tanh(in_act[:, :n_channels_int, :])
        s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
        acts = t_act * s_act
        return acts

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        if embedding is None:
            return self.norm(input)
        weight, bias = torch.split(
            self.project_layer(embedding), split_size_or_sections=self.d_model, dim=-1,
        )
        return weight * self.norm(input) + bias


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    has_cross_attention: bool = False
    context_dim: int = 0
    uvit_skip_connection: bool = False
    time_as_token: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        # self.head_dim = self.dim // self.n_head


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = AdaptiveLayerNorm(
            config.dim, RMSNorm(config.dim, eps=config.norm_eps)
        )

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length, use_kv_cache=False):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.norm.project_layer.weight.dtype
        device = self.norm.project_layer.weight.device

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size, self.config.head_dim, self.config.rope_base, dtype
        ).to(device)
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        ).to(device)
        self.use_kv_cache = use_kv_cache
        self.uvit_skip_connection = self.config.uvit_skip_connection
        if self.uvit_skip_connection:
            self.layers_emit_skip = [
                i for i in range(self.config.n_layer) if i < self.config.n_layer // 2
            ]
            self.layers_receive_skip = [
                i for i in range(self.config.n_layer) if i > self.config.n_layer // 2
            ]
        else:
            self.layers_emit_skip = []
            self.layers_receive_skip = []

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        input_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_input_pos: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        if mask is None:  # in case of non-causal model
            if not self.training and self.use_kv_cache:
                mask = self.causal_mask[None, None, input_pos]
            else:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        if context is not None:
            context_freqs_cis = self.freqs_cis[context_input_pos]
        else:
            context_freqs_cis = None
        skip_in_x_list = []
        for i, layer in enumerate(self.layers):
            if self.uvit_skip_connection and i in self.layers_receive_skip:
                skip_in_x = skip_in_x_list.pop(-1)
            else:
                skip_in_x = None
            x = layer(
                x,
                c,
                input_pos,
                freqs_cis,
                mask,
                context,
                context_freqs_cis,
                cross_attention_mask,
                skip_in_x,
            )
            if self.uvit_skip_connection and i in self.layers_emit_skip:
                skip_in_x_list.append(x)
        x = self.norm(x, c)
        return x

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = AdaptiveLayerNorm(
            config.dim, RMSNorm(config.dim, eps=config.norm_eps)
        )
        self.attention_norm = AdaptiveLayerNorm(
            config.dim, RMSNorm(config.dim, eps=config.norm_eps)
        )

        if config.has_cross_attention:
            self.has_cross_attention = True
            self.cross_attention = Attention(config, is_cross_attention=True)
            self.cross_attention_norm = AdaptiveLayerNorm(
                config.dim, RMSNorm(config.dim, eps=config.norm_eps)
            )
        else:
            self.has_cross_attention = False

        if config.uvit_skip_connection:
            self.skip_in_linear = nn.Linear(config.dim * 2, config.dim)
            self.uvit_skip_connection = True
        else:
            self.uvit_skip_connection = False

        self.time_as_token = config.time_as_token

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        input_pos: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        context: Optional[Tensor] = None,
        context_freqs_cis: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
        skip_in_x: Optional[Tensor] = None,
    ) -> Tensor:
        c = None if self.time_as_token else c
        if self.uvit_skip_connection and skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask, input_pos)
        if self.has_cross_attention:
            h = h + self.cross_attention(
                self.cross_attention_norm(h, c),
                freqs_cis,
                cross_attention_mask,
                input_pos,
                context,
                context_freqs_cis,
            )
        out = h + self.feed_forward(self.ffn_norm(h, c))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, is_cross_attention: bool = False):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        if is_cross_attention:
            self.wq = nn.Linear(config.dim, config.n_head * config.head_dim, bias=False)
            self.wkv = nn.Linear(
                config.context_dim,
                2 * config.n_local_heads * config.head_dim,
                bias=False,
            )
        else:
            self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_freqs_cis: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        if context is None:
            q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
            context_seqlen = seqlen
        else:
            q = self.wq(x)
            k, v = self.wkv(context).split([kv_size, kv_size], dim=-1)
            context_seqlen = context.shape[1]

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(
            k, context_freqs_cis if context_freqs_cis is not None else freqs_cis
        )

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(bsz, seqlen, self.head_dim * self.n_head)
        )

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = 10000
        self.scale = 1000

        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        self.register_buffer("freqs", freqs)

    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        args = self.scale * t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class StyleEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, input_size, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(int(use_cfg_embedding), hidden_size)
        self.style_in = weight_norm(nn.Linear(input_size, hidden_size, bias=True))
        self.input_size = input_size
        self.dropout_prob = dropout_prob

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        else:
            labels = self.style_in(labels)
        embeddings = labels
        return embeddings


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(torch.nn.Module):
    def __init__(self, args):
        super(DiT, self).__init__()
        self.time_as_token = (
            args.DiT.time_as_token if hasattr(args.DiT, "time_as_token") else False
        )
        self.style_as_token = (
            args.DiT.style_as_token if hasattr(args.DiT, "style_as_token") else False
        )
        self.uvit_skip_connection = (
            args.DiT.uvit_skip_connection
            if hasattr(args.DiT, "uvit_skip_connection")
            else False
        )
        model_args = ModelArgs(
            block_size=16384,  # args.DiT.block_size,
            n_layer=args.DiT.depth,
            n_head=args.DiT.num_heads,
            dim=args.DiT.hidden_dim,
            head_dim=args.DiT.hidden_dim // args.DiT.num_heads,
            vocab_size=1024,
            uvit_skip_connection=self.uvit_skip_connection,
            time_as_token=self.time_as_token,
        )
        self.transformer = Transformer(model_args)
        self.in_channels = args.DiT.in_channels
        self.out_channels = args.DiT.in_channels
        self.num_heads = args.DiT.num_heads

        self.x_embedder = weight_norm(
            nn.Linear(args.DiT.in_channels, args.DiT.hidden_dim, bias=True)
        )

        self.content_type = args.DiT.content_type  # 'discrete' or 'continuous'
        self.content_codebook_size = (
            args.DiT.content_codebook_size
        )  # for discrete content
        self.content_dim = args.DiT.content_dim  # for continuous content
        self.cond_embedder = nn.Embedding(
            args.DiT.content_codebook_size, args.DiT.hidden_dim
        )  # discrete content
        self.cond_projection = nn.Linear(
            args.DiT.content_dim, args.DiT.hidden_dim, bias=True
        )  # continuous content

        self.is_causal = args.DiT.is_causal

        self.t_embedder = TimestepEmbedder(args.DiT.hidden_dim)

        input_pos = torch.arange(16384)
        self.register_buffer("input_pos", input_pos)

        self.final_layer_type = args.DiT.final_layer_type  # mlp or wavenet
        if self.final_layer_type == "wavenet":
            self.t_embedder2 = TimestepEmbedder(args.wavenet.hidden_dim)
            self.conv1 = nn.Linear(args.DiT.hidden_dim, args.wavenet.hidden_dim)
            self.conv2 = nn.Conv1d(args.wavenet.hidden_dim, args.DiT.in_channels, 1)
            self.wavenet = WN(
                hidden_channels=args.wavenet.hidden_dim,
                kernel_size=args.wavenet.kernel_size,
                dilation_rate=args.wavenet.dilation_rate,
                n_layers=args.wavenet.num_layers,
                gin_channels=args.wavenet.hidden_dim,
                p_dropout=args.wavenet.p_dropout,
                causal=False,
            )
            self.final_layer = FinalLayer(
                args.wavenet.hidden_dim, 1, args.wavenet.hidden_dim
            )
            self.res_projection = nn.Linear(
                args.DiT.hidden_dim, args.wavenet.hidden_dim
            )  # residual connection from tranformer output to final output
            self.wavenet_style_condition = args.wavenet.style_condition
            assert args.DiT.style_condition == args.wavenet.style_condition
        else:
            self.final_mlp = nn.Sequential(
                nn.Linear(args.DiT.hidden_dim, args.DiT.hidden_dim),
                nn.SiLU(),
                nn.Linear(args.DiT.hidden_dim, args.DiT.in_channels),
            )
        self.transformer_style_condition = args.DiT.style_condition

        self.class_dropout_prob = args.DiT.class_dropout_prob
        self.content_mask_embedder = nn.Embedding(1, args.DiT.hidden_dim)

        self.long_skip_connection = args.DiT.long_skip_connection
        self.skip_linear = nn.Linear(
            args.DiT.hidden_dim + args.DiT.in_channels, args.DiT.hidden_dim
        )

        self.cond_x_merge_linear = nn.Linear(
            args.DiT.hidden_dim
            + args.DiT.in_channels * 2
            + args.style_encoder.dim
            * self.transformer_style_condition
            * (not self.style_as_token),
            args.DiT.hidden_dim,
        )
        if self.style_as_token:
            self.style_in = nn.Linear(args.style_encoder.dim, args.DiT.hidden_dim)

    def setup_caches(self, max_batch_size, max_seq_length):
        self.transformer.setup_caches(
            max_batch_size, max_seq_length, use_kv_cache=False
        )

    def forward(self, x, prompt_x, x_lens, t, style, cond, mask_content=False):
        class_dropout = False
        if self.training and torch.rand(1) < self.class_dropout_prob:
            class_dropout = True
        if not self.training and mask_content:
            class_dropout = True
        # cond_in_module = self.cond_embedder if self.content_type == 'discrete' else self.cond_projection
        cond_in_module = self.cond_projection

        B, _, T = x.size()

        t1 = self.t_embedder(t)  # (N, D)

        cond = cond_in_module(cond)

        x = x.transpose(1, 2)
        prompt_x = prompt_x.transpose(1, 2)

        x_in = torch.cat([x, prompt_x, cond], dim=-1)
        if self.transformer_style_condition and not self.style_as_token:
            x_in = torch.cat([x_in, style[:, None, :].repeat(1, T, 1)], dim=-1)
        if class_dropout:
            x_in[..., self.in_channels :] = x_in[..., self.in_channels :] * 0
        x_in = self.cond_x_merge_linear(x_in)  # (N, T, D)

        if self.style_as_token:
            style = self.style_in(style)
            style = torch.zeros_like(style) if class_dropout else style
            x_in = torch.cat([style.unsqueeze(1), x_in], dim=1)
        if self.time_as_token:
            x_in = torch.cat([t1.unsqueeze(1), x_in], dim=1)
        x_mask = (
            sequence_mask(x_lens + self.style_as_token + self.time_as_token)
            .to(x.device)
            .unsqueeze(1)
        )
        input_pos = self.input_pos[: x_in.size(1)]  # (T,)
        x_mask_expanded = (
            x_mask[:, None, :].repeat(1, 1, x_in.size(1), 1)
            if not self.is_causal
            else None
        )
        x_res = self.transformer(x_in, t1.unsqueeze(1), input_pos, x_mask_expanded)
        x_res = x_res[:, 1:] if self.time_as_token else x_res
        x_res = x_res[:, 1:] if self.style_as_token else x_res
        if self.long_skip_connection:
            x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))
        if self.final_layer_type == "wavenet":
            x = self.conv1(x_res)
            x = x.transpose(1, 2)
            t2 = self.t_embedder2(t)
            x = self.wavenet(x, x_mask, g=t2.unsqueeze(2)).transpose(
                1, 2
            ) + self.res_projection(
                x_res
            )  # long residual connection
            x = self.final_layer(x, t1).transpose(1, 2)
            x = self.conv2(x)
        else:
            x = self.final_mlp(x_res)
            x = x.transpose(1, 2)
        return x


#########################################
#           Flow Matching               #
#########################################


class BASECFM(torch.nn.Module):
    def __init__(
        self, args,
    ):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = None

        self.in_channels = args.DiT.in_channels

        self.criterion = (
            torch.nn.MSELoss() if args.reg_loss_type == "l2" else torch.nn.L1Loss()
        )

        if hasattr(args.DiT, "zero_prompt_speech_token"):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False

    @torch.inference_mode()
    def inference(
        self,
        mu,
        x_lens,
        prompt,
        style,
        f0,
        n_timesteps,
        temperature=1.0,
        inference_cfg_rate=0.5,
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        # t_span = t_span + (-1) * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(
            z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate
        )

    def solve_euler(
        self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5
    ):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0
        for step in tqdm(range(1, len(t_span)), desc="Stage 2 Infer"):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                # Stack original and CFG (null) inputs for batched processing
                stacked_prompt_x = torch.cat(
                    [prompt_x, torch.zeros_like(prompt_x)], dim=0
                )
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)

                # Perform a single forward pass for both original and CFG inputs
                stacked_dphi_dt = self.estimator(
                    stacked_x,
                    stacked_prompt_x,
                    x_lens,
                    stacked_t,
                    stacked_style,
                    stacked_mu,
                )

                # Split the output back into the original and CFG components
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)

                # Apply CFG formula
                dphi_dt = (
                    1.0 + inference_cfg_rate
                ) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        return sol[-1]

    def forward(self, x1, x_lens, prompt_lens, mu, style):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, : prompt_lens[bib]] = x1[bib, :, : prompt_lens[bib]]
            # range covered by prompt are set to 0
            y[bib, :, : prompt_lens[bib]] = 0
            if self.zero_prompt_speech_token:
                mu[bib, :, : prompt_lens[bib]] = 0

        estimator_out = self.estimator(
            y, prompt, x_lens, t.squeeze(1).squeeze(1), style, mu, prompt_lens
        )
        loss = 0
        for bib in range(b):
            loss += self.criterion(
                estimator_out[bib, :, prompt_lens[bib] : x_lens[bib]],
                u[bib, :, prompt_lens[bib] : x_lens[bib]],
            )
        loss /= b

        return loss, estimator_out + (1 - self.sigma_min) * z


class CFM(BASECFM):
    def __init__(self, args):
        super().__init__(args)
        if args.dit_type == "DiT":
            self.estimator = DiT(args)
        else:
            raise NotImplementedError(f"Unknown diffusion type {args.dit_type}")


#########################################
#                   LR                  #
#########################################


# def WNConv1d(*args, **kwargs):
#     return weight_norm(nn.Conv1d(*args, **kwargs))


# class VectorQuantize(nn.Module):
#     """
#     Implementation of VQ similar to Karpathy's repo:
#     https://github.com/karpathy/deep-vector-quantization
#     Additionally uses following tricks from Improved VQGAN
#     (https://arxiv.org/pdf/2110.04627.pdf):
#         1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
#             for improved codebook usage
#         2. l2-normalized codes: Converts euclidean distance to cosine similarity which
#             improves training stability
#     """

#     def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
#         super().__init__()
#         self.codebook_size = codebook_size
#         self.codebook_dim = codebook_dim

#         self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
#         self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
#         self.codebook = nn.Embedding(codebook_size, codebook_dim)

#     def forward(self, z):
#         """Quantized the input tensor using a fixed codebook and returns
#         the corresponding codebook vectors

#         Parameters
#         ----------
#         z : Tensor[B x D x T]

#         Returns
#         -------
#         Tensor[B x D x T]
#             Quantized continuous representation of input
#         Tensor[1]
#             Commitment loss to train encoder to predict vectors closer to codebook
#             entries
#         Tensor[1]
#             Codebook loss to update the codebook
#         Tensor[B x T]
#             Codebook indices (quantized discrete representation of input)
#         Tensor[B x D x T]
#             Projected latents (continuous representation of input before quantization)
#         """

#         # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
#         z_e = self.in_proj(z)  # z_e : (B x D x T)
#         z_q, indices = self.decode_latents(z_e)

#         commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
#         codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

#         z_q = (
#             z_e + (z_q - z_e).detach()
#         )  # noop in forward pass, straight-through gradient estimator in backward pass

#         z_q = self.out_proj(z_q)

#         return z_q, commitment_loss, codebook_loss, indices, z_e

#     def embed_code(self, embed_id):
#         return F.embedding(embed_id, self.codebook.weight)

#     def decode_code(self, embed_id):
#         return self.embed_code(embed_id).transpose(1, 2)

#     def decode_latents(self, latents):
#         encodings = einops.rearrange(latents, "b d t -> (b t) d")
#         codebook = self.codebook.weight  # codebook: (N x D)

#         # L2 normalize encodings and codebook (ViT-VQGAN)
#         encodings = F.normalize(encodings)
#         codebook = F.normalize(codebook)

#         # Compute euclidean distance with codebook
#         dist = (
#             encodings.pow(2).sum(1, keepdim=True)
#             - 2 * encodings @ codebook.t()
#             + codebook.pow(2).sum(1, keepdim=True).t()
#         )
#         indices = einops.rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
#         z_q = self.decode_code(indices)
#         return z_q, indices


class InterpolateRegulator(nn.Module):
    def __init__(
        self,
        channels: int,
        sampling_ratios: list,
        is_discrete: bool = False,
        in_channels: int = None,  # only applies to continuous input
        vector_quantize: bool = False,  # whether to use vector quantization, only applies to continuous input
        codebook_size: int = 1024,  # for discrete only
        out_channels: int = None,
        groups: int = 1,
        n_codebooks: int = 1,  # number of codebooks
        quantizer_dropout: float = 0.0,  # dropout for quantizer
        f0_condition: bool = False,
        n_f0_bins: int = 512,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            self.interpolate = True
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        else:
            self.interpolate = False
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)
        self.embedding = nn.Embedding(codebook_size, channels)
        self.is_discrete = is_discrete

        self.mask_token = nn.Parameter(torch.zeros(1, channels))

        self.n_codebooks = n_codebooks
        if n_codebooks > 1:
            self.extra_codebooks = nn.ModuleList(
                [nn.Embedding(codebook_size, channels) for _ in range(n_codebooks - 1)]
            )
            self.extra_codebook_mask_tokens = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, channels)) for _ in range(n_codebooks - 1)]
            )
        self.quantizer_dropout = quantizer_dropout

        if f0_condition:
            self.f0_embedding = nn.Embedding(n_f0_bins, channels)
            self.f0_condition = f0_condition
            self.n_f0_bins = n_f0_bins
            self.f0_bins = torch.arange(2, 1024, 1024 // n_f0_bins)
            self.f0_mask = nn.Parameter(torch.zeros(1, channels))
        else:
            self.f0_condition = False

        if not is_discrete:
            self.content_in_proj = nn.Linear(in_channels, channels)
            # if vector_quantize:
            #     self.vq = VectorQuantize(channels, codebook_size, 8)

    def forward(self, x, ylens=None, n_quantizers=None, f0=None):
        # apply token drop
        if self.training:
            n_quantizers = torch.ones((x.shape[0],)) * self.n_codebooks
            dropout = torch.randint(1, self.n_codebooks + 1, (x.shape[0],))
            n_dropout = int(x.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(x.device)
            # decide whether to drop for each sample in batch
        else:
            n_quantizers = torch.ones((x.shape[0],), device=x.device) * (
                self.n_codebooks if n_quantizers is None else n_quantizers
            )
        if self.is_discrete:
            if self.n_codebooks > 1:
                assert len(x.size()) == 3
                x_emb = self.embedding(x[:, 0])
                for i, emb in enumerate(self.extra_codebooks):
                    x_emb = x_emb + (n_quantizers > i + 1)[..., None, None] * emb(
                        x[:, i + 1]
                    )
                    # add mask token if not using this codebook
                    # x_emb = x_emb + (n_quantizers <= i+1)[..., None, None] * self.extra_codebook_mask_tokens[i]
                x = x_emb
            elif self.n_codebooks == 1:
                if len(x.size()) == 2:
                    x = self.embedding(x)
                else:
                    x = self.embedding(x[:, 0])
        else:
            x = self.content_in_proj(x)
        # x in (B, T, D)
        mask = sequence_mask(ylens).unsqueeze(-1)
        if self.interpolate:
            x = F.interpolate(
                x.transpose(1, 2).contiguous(), size=ylens.max(), mode="nearest"
            )
        else:
            x = x.transpose(1, 2).contiguous()
            mask = mask[:, : x.size(2), :]
            ylens = ylens.clamp(max=x.size(2)).long()
        if self.f0_condition:
            if f0 is None:
                x = x + self.f0_mask.unsqueeze(-1)
            else:
                # quantized_f0 = torch.bucketize(f0, self.f0_bins.to(f0.device))  # (N, T)
                quantized_f0 = f0_to_coarse(f0, self.n_f0_bins)
                quantized_f0 = quantized_f0.clamp(0, self.n_f0_bins - 1).long()
                f0_emb = self.f0_embedding(quantized_f0)
                f0_emb = F.interpolate(
                    f0_emb.transpose(1, 2).contiguous(),
                    size=ylens.max(),
                    mode="nearest",
                )
                x = x + f0_emb
        out = self.model(x).transpose(1, 2).contiguous()
        if hasattr(self, "vq"):
            out_q, commitment_loss, codebook_loss, codes, out, = self.vq(
                out.transpose(1, 2)
            )
            out_q = out_q.transpose(1, 2)
            return out_q * mask, ylens, codes, commitment_loss, codebook_loss
        olens = ylens
        return out * mask, olens, None, None, None


#########################################
#               campplus                #
#########################################


class CampplusStatsPool(nn.Module):
    def forward(self, x, x_lens=None):
        if x_lens is not None:
            return masked_statistics_pooling(x, x_lens)
        return statistics_pooling(x)


class CampplusTDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super(CampplusTDNNLayer, self).__init__()
        if padding < 0:
            assert (
                kernel_size % 2 == 1
            ), "Expect equal paddings, but got even kernel size ({})".format(
                kernel_size
            )
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    def __init__(
        self,
        bn_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        reduction=2,
    ):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., : x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super(CAMDenseTDNNLayer, self).__init__()
        assert (
            kernel_size % 2 == 1
        ), "Expect equal paddings, but got even kernel size ({})".format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module("tdnnd%d" % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class CampplusTransitLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"
    ):
        super(CampplusTransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class CampplusDenseLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"
    ):
        super(CampplusDenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CampplusBasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(CampplusBasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CampplusFCM(nn.Module):
    def __init__(
        self, block=CampplusBasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80
    ):
        super(CampplusFCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)

        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        embedding_size=512,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
    ):
        super(CAMPPlus, self).__init__()

        self.head = CampplusFCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        CampplusTDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                CampplusTransitLayer(
                    channels, channels // 2, bias=False, config_str=config_str
                ),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        # self.xvector.add_module('stats', CampplusStatsPool())
        # self.xvector.add_module(
        #     'dense',
        #     CampplusDenseLayer(channels * 2, embedding_size, config_str='batchnorm_'))
        self.stats = CampplusStatsPool()
        self.dense = CampplusDenseLayer(
            channels * 2, embedding_size, config_str="batchnorm_"
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict that remaps keys from a previous version of the model where
        stats and dense layers were part of xvector.
        """
        new_state_dict = {}

        # Remap keys for compatibility
        for key in state_dict.keys():
            new_key = key
            if key.startswith("xvector.stats"):
                new_key = key.replace("xvector.stats", "stats")
            elif key.startswith("xvector.dense"):
                new_key = key.replace("xvector.dense", "dense")
            new_state_dict[new_key] = state_dict[key]

        # Call the original load_state_dict with the modified state_dict
        super(CAMPPlus, self).load_state_dict(new_state_dict, strict)

    def forward(self, x, x_lens=None):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        x = self.stats(x, x_lens)
        x = self.dense(x)
        return x


#########################################
#               BigVGAN                 #
#########################################


def kaiser_sinc_filter1d(
    cutoff, half_width, kernel_size
):  # return filter [1,1,kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        """
        Normalize filter to have sum = 1, otherwise we will have a small leakage of the constant component in the input signal.
        """
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        """
        kernel_size should be even number for stylegan3 setup, in this implementation, odd number is also possible.
        """
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    # Input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)

        return out


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C
        )
        x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        xx = self.lowpass(x)

        return xx


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(torch.sin(x * alpha), 2)

        return x


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: list = [1, 3, 5]):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility

        # Activation functions
        self.activations = nn.ModuleList(
            [
                Activation1d(activation=SnakeBeta(channels, alpha_logscale=True))
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.convs2:
            torch.nn.utils.remove_weight_norm(l)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class BigVGAN(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        h = AttrDict(h)

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            nn.Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            nn.ConvTranspose1d(
                                h.upsample_initial_channel // (2 ** i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(AMPBlock1(ch, k, d))

        # Post-conv
        activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            nn.Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    @torch.no_grad()
    def forward(self, x):
        # Pre-conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x

    def remove_weight_norm(self):
        try:
            for l in self.ups:
                for l_i in l:
                    torch.nn.utils.remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            torch.nn.utils.remove_weight_norm(self.conv_pre)
            torch.nn.utils.remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass


#########################################
#               RMVPE                   #
#########################################


class STFT(torch.nn.Module):
    def __init__(
        self, filter_length=1024, hop_length=512, win_length=None, window="hann"
    ):
        """
        This module implements an STFT using 1D convolution and 1D transpose convolutions.
        This is a bit tricky so there are some cases that probably won't work as working
        out the same sizes before and after in all overlap add setups is tough. Right now,
        this code should work with hop lengths that are half the filter length (50% overlap
        between frames).

        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris)
                (default: {'hann'})
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        forward_basis = torch.FloatTensor(fourier_basis)
        inverse_basis = torch.FloatTensor(np.linalg.pinv(fourier_basis))

        assert filter_length >= self.win_length
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window
        inverse_basis = (inverse_basis.T * fft_window).T

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())
        self.register_buffer("fft_window", fft_window.float())

    def transform(self, input_data, return_phase=False):
        """Take input data (audio) to STFT domain.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)
        """
        input_data = F.pad(
            input_data, (self.pad_amount, self.pad_amount), mode="reflect",
        )
        forward_transform = input_data.unfold(
            1, self.filter_length, self.hop_length
        ).permute(0, 2, 1)
        forward_transform = torch.matmul(self.forward_basis, forward_transform)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        if return_phase:
            phase = torch.atan2(imag_part.data, real_part.data)
            return magnitude, phase
        else:
            return magnitude

    def inverse(self, magnitude, phase):
        """Call the inverse STFT (iSTFT), given magnitude and phase tensors produced
        by the ```transform``` function.

        Arguments:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)

        Returns:
            inverse_transform {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        cat = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        fold = torch.nn.Fold(
            output_size=(1, (cat.size(-1) - 1) * self.hop_length + self.filter_length),
            kernel_size=(1, self.filter_length),
            stride=(1, self.hop_length),
        )
        inverse_transform = torch.matmul(self.inverse_basis, cat)
        inverse_transform = fold(inverse_transform)[
            :, 0, 0, self.pad_amount : -self.pad_amount
        ]
        window_square_sum = (
            self.fft_window.pow(2).repeat(cat.size(-1), 1).T.unsqueeze(0)
        )
        window_square_sum = fold(window_square_sum)[
            :, 0, 0, self.pad_amount : -self.pad_amount
        ]
        inverse_transform /= window_square_sum
        return inverse_transform

    def forward(self, input_data):
        """Take input data (audio) to STFT domain and then back to audio.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            reconstruction {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        self.magnitude, self.phase = self.transform(input_data, return_phase=True)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class RMVPEBiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(RMVPEBiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class RMVPEConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(RMVPEConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        # self.shortcut:Optional[nn.Module] = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        else:
            return self.conv(x) + self.shortcut(x)


class RMVPEResEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01
    ):
        super(RMVPEResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(RMVPEConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(RMVPEConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i, conv in enumerate(self.conv):
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class RMVPEEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super(RMVPEEncoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                RMVPEResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor):
        concat_tensors: List[torch.Tensor] = []
        x = self.bn(x)
        for i, layer in enumerate(self.layers):
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class RMVPEIntermediate(nn.Module):  #
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(RMVPEIntermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(
            RMVPEResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for i in range(self.n_inters - 1):
            self.layers.append(
                RMVPEResEncoderBlock(
                    out_channels, out_channels, None, n_blocks, momentum
                )
            )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class RMVPEResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(RMVPEResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(RMVPEConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(RMVPEConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i, conv2 in enumerate(self.conv2):
            x = conv2(x)
        return x


class RMVPEDecoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(RMVPEDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                RMVPEResDecoderBlock(
                    in_channels, out_channels, stride, n_blocks, momentum
                )
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: List[torch.Tensor]):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class RMVPEDeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(RMVPEDeepUnet, self).__init__()
        self.encoder = RMVPEEncoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = RMVPEIntermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = RMVPEDecoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class RMVPEE2E(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(RMVPEE2E, self).__init__()
        self.unet = RMVPEDeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                RMVPEBiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * nn.N_MELS, nn.N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
            )

    def forward(self, mel):
        # print(mel.shape)
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        # print(x.shape)
        return x


class RMVPEMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        is_half,
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
        mel_basis = librosa_mel_fn(
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
        self.is_half = is_half

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
        if "privateuseone" in str(audio.device):
            if not hasattr(self, "stft"):
                self.stft = STFT(
                    filter_length=n_fft_new,
                    hop_length=hop_length_new,
                    win_length=win_length_new,
                    window="hann",
                ).to(audio.device)
            magnitude = self.stft.transform(audio)
        else:
            fft = torch.stft(
                audio,
                n_fft=n_fft_new,
                hop_length=hop_length_new,
                win_length=win_length_new,
                window=self.hann_window[keyshift_key],
                center=center,
                return_complex=True,
            )
            magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half == True:
            mel_output = mel_output.half()
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class RMVPE:
    def __init__(self, model_ckpt, device=None, use_jit=False):
        self.resample_kernel = {}
        self.resample_kernel = {}
        if device is None:
            # device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.mel_extractor = RMVPEMelSpectrogram(
            False, 128, 16000, 1024, 160, None, 30, 8000
        ).to(device)
        self.model = RMVPEE2E(4, 1, (2, 2))
        self.model.load_state_dict(model_ckpt)
        self.model.eval().to(device)
        self.model.float()
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            if "privateuseone" in str(self.device):
                onnx_input_name = self.model.get_inputs()[0].name
                onnx_outputs_names = self.model.get_outputs()[0].name
                hidden = self.model.run(
                    [onnx_outputs_names],
                    input_feed={onnx_input_name: mel.cpu().numpy()},
                )[0]
            else:
                mel = mel.float()
                hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        # f0 = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        # torch.cuda.synchronize()
        # t0 = ttime()
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        mel = self.mel_extractor(
            audio.float().to(self.device).unsqueeze(0), center=True
        )
        # print(123123123,mel.device.type)
        # torch.cuda.synchronize()
        # t1 = ttime()
        hidden = self.mel2hidden(mel)
        # torch.cuda.synchronize()
        # t2 = ttime()
        # print(234234,hidden.device.type)
        if "privateuseone" not in str(self.device):
            hidden = hidden.squeeze(0).cpu().numpy()
        else:
            hidden = hidden[0]
        f0 = self.decode(hidden, thred=thred)
        # torch.cuda.synchronize()
        # t3 = ttime()
        # print("hmvpe:%s\t%s\t%s\t%s"%(t1-t0,t2-t1,t3-t2,t3-t0))
        return f0

    def infer_from_audio_batch(self, audio, thred=0.03):
        # torch.cuda.synchronize()
        # t0 = ttime()
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        mel = self.mel_extractor(audio.float().to(self.device), center=True)
        # print(123123123,mel.device.type)
        # torch.cuda.synchronize()
        # t1 = ttime()
        hidden = self.mel2hidden(mel)
        # torch.cuda.synchronize()
        # t2 = ttime()
        # print(234234,hidden.device.type)
        if "privateuseone" not in str(self.device):
            hidden = hidden.cpu().numpy()
        else:
            pass

        f0s = []
        for bib in range(hidden.shape[0]):
            f0s.append(self.decode(hidden[bib], thred=thred))
        f0s = np.stack(f0s)
        f0s = torch.from_numpy(f0s).to(self.device)
        # torch.cuda.synchronize()
        # t3 = ttime()
        # print("hmvpe:%s\t%s\t%s\t%s"%(t1-t0,t2-t1,t3-t2,t3-t0))
        return f0s

    def to_local_average_cents(self, salience, thred=0.05):
        # t0 = ttime()
        center = np.argmax(salience, axis=1)  # 帧长#index
        salience = np.pad(salience, ((0, 0), (4, 4)))  # 帧长,368
        # t1 = ttime()
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        # t2 = ttime()
        todo_salience = np.array(todo_salience)  # 帧长，9
        todo_cents_mapping = np.array(todo_cents_mapping)  # 帧长，9
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)  # 帧长
        devided = product_sum / weight_sum  # 帧长
        # t3 = ttime()
        maxx = np.max(salience, axis=1)  # 帧长
        devided[maxx <= thred] = 0
        # t4 = ttime()
        # print("decode:%s\t%s\t%s\t%s" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        return devided


#########################################
#               nsf hifigan             #
#########################################
LRELU_SLOPE = 0.1


class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-waveform (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_threshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values, upp):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        rad_values = (f0_values / self.sampling_rate).fmod(1.0)  # %1意味着n_har的乘积无法后处理优化
        rand_ini = torch.rand(1, self.dim, device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] += rand_ini
        is_half = rad_values.dtype is not torch.float32
        tmp_over_one = torch.cumsum(
            rad_values.double(), 1
        )  # % 1  #####%1意味着后面的cumsum无法再优化
        if is_half:
            tmp_over_one = tmp_over_one.half()
        else:
            tmp_over_one = tmp_over_one.float()
        tmp_over_one *= upp
        tmp_over_one = F.interpolate(
            tmp_over_one.transpose(2, 1),
            scale_factor=upp,
            mode="linear",
            align_corners=True,
        ).transpose(2, 1)
        rad_values = F.interpolate(
            rad_values.transpose(2, 1), scale_factor=upp, mode="nearest"
        ).transpose(2, 1)
        tmp_over_one = tmp_over_one.fmod(1.0)
        diff = F.conv2d(
            tmp_over_one.unsqueeze(1),
            torch.FloatTensor([[[[-1.0], [1.0]]]]).to(tmp_over_one.device),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
        ).squeeze(
            1
        )  # Equivalent to torch.diff, but able to export ONNX
        cumsum_shift = (diff < 0).double()
        cumsum_shift = torch.cat(
            (
                torch.zeros((1, 1, self.dim), dtype=torch.double).to(f0_values.device),
                cumsum_shift,
            ),
            dim=1,
        )
        sines = torch.sin(
            torch.cumsum(rad_values.double() + cumsum_shift, dim=1) * 2 * np.pi
        )
        if is_half:
            sines = sines.half()
        else:
            sines = sines.float()
        return sines

    @torch.no_grad()
    def forward(self, f0, upp):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0 = f0.unsqueeze(-1)
        fn = torch.multiply(
            f0, torch.arange(1, self.dim + 1, device=f0.device).reshape((1, 1, -1))
        )
        sine_waves = self._f02sine(fn, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).float()
        uv = F.interpolate(
            uv.transpose(2, 1), scale_factor=upp, mode="nearest"
        ).transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.convs2:
            torch.nn.utils.remove_weight_norm(l)


class NsfHifiGAN(nn.Module):
    def __init__(self, h):
        super(NsfHifiGAN, self).__init__()
        h = AttrDict(h)
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=h.sampling_rate, harmonic_num=8)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(
            nn.Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = ResBlock1

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            c_cur = h.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        h.upsample_initial_channel // (2 ** i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(h.upsample_rates):  #
                stride_f0 = int(np.prod(h.upsample_rates[i + 1 :]))
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        ch = h.upsample_initial_channel
        for i in range(len(self.ups)):
            ch //= 2
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = int(np.prod(h.upsample_rates))

    @torch.no_grad()
    def forward(self, x, f0):
        x = x.transpose(2, 1)  # [B, T, bins]
        x = 2.30259 * x

        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x).view(-1)
        return x

    def remove_weight_norm(self):

        for l in self.ups:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.resblocks:
            torch.nn.utils.remove_weight_norm(l)
        torch.nn.utils.remove_weight_norm(self.conv_pre)
        torch.nn.utils.remove_weight_norm(self.conv_post)


#########################################
#               DiffSinger              #
#########################################


PAD = "<PAD>"
PAD_INDEX = 0
VOCAB_LIST = [
    "AP",
    "E",
    "En",
    "SP",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
]


class TokenTextEncoder:
    """Encoder based on a user-supplied vocabulary (file or list)."""

    def __init__(self, hparams, vocab_list):
        """Initialize from a file or list, one token per line.

        Handling of reserved tokens works as follows:
        - When initializing from a list, we add reserved tokens to the vocab.

        Args:
            vocab_list: If not None, a list of elements of the vocabulary.
        """
        self.num_reserved_ids = hparams.get("num_pad_tokens", 3)
        assert self.num_reserved_ids > 0, "num_pad_tokens must be positive"
        self.vocab_list = sorted(vocab_list)  # phoneme list

    def encode(self, sentence):
        """Converts a space-separated string of phones to a list of ids."""
        phones = sentence.strip().split() if isinstance(sentence, str) else sentence
        return [
            self.vocab_list.index(ph) + self.num_reserved_ids
            if ph != PAD
            else PAD_INDEX
            for ph in phones
        ]

    def decode(self, ids, strip_padding=False):
        if strip_padding:
            ids = np.trim_zeros(ids)
        ids = list(ids)
        return " ".join(
            [
                self.vocab_list[_id - self.num_reserved_ids]
                if _id >= self.num_reserved_ids
                else PAD
                for _id in ids
            ]
        )

    def pad(self):
        pass

    @property
    def vocab_size(self):
        return len(self.vocab_list) + self.num_reserved_ids

    def __len__(self):
        return self.vocab_size

    def store_to_file(self, filename):
        """Write vocab file to disk.

        Vocab files have one token per line. The file ends in a newline. Reserved
        tokens are written to the vocab file as well.

        Args:
        filename: Full path of the file to store the vocab to.
        """
        with open(filename, "w", encoding="utf8") as f:
            [print(PAD, file=f) for _ in range(self.num_reserved_ids)]
            [print(tok, file=f) for tok in self.vocab_list]


class LengthRegulator(torch.nn.Module):
    def forward(self, dur, dur_padding=None, alpha=None):
        """
        Example (no batch dim version):
            1. dur = [2,2,3]
            2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
            3. token_mask = [[1,1,0,0,0,0,0],
                             [0,0,1,1,0,0,0],
                             [0,0,0,0,1,1,1]]
            4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                         [0,0,2,2,0,0,0],
                                         [0,0,0,0,3,3,3]]
            5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

        :param dur: Batch of durations of each frame (B, T_txt)         # torch.Size([1, 24])
        :param dur_padding: Batch of padding of each frame (B, T_txt)   
        :param alpha: duration rescale coefficient
        :return:
            mel2ph (B, T_speech)
        """
        assert alpha is None or alpha > 0
        if alpha is not None:
            dur = torch.round(dur.float() * alpha).long()
        if dur_padding is not None:
            dur = dur * (1 - dur_padding.long())

        token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
        dur_cumsum = torch.cumsum(dur, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)
        pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (
            pos_idx < dur_cumsum[:, :, None]
        )
        mel2ph = (token_idx * token_mask.long()).sum(1)
        return mel2ph


class NormalInitEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_embeddings, embedding_dim, *args, padding_idx=padding_idx, **kwargs
        )
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(self.weight[padding_idx], 0)


class XavierUniformInitLinear(torch.nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, *args, bias: bool = True, **kwargs
    ):
        super().__init__(in_features, out_features, *args, bias=bias, **kwargs)
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.constant_(self.bias, 0.0)


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = self.get_embedding(init_size, embedding_dim, padding_idx,)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, x, incremental_state=None, timestep=None, positions=None):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = x.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = self.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = (
            self.make_positions(x, self.padding_idx) if positions is None else positions
        )
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )

    @staticmethod
    def make_positions(tensor, padding_idx):
        """Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

    @staticmethod
    def max_positions():
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class BatchNorm1dTBC(nn.Module):
    def __init__(self, c):
        super(BatchNorm1dTBC, self).__init__()
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        """

        :param x: [T, B, C]
        :return: [T, B, C]
        """
        x = x.permute(1, 2, 0)  # [B, C, T]
        x = self.bn(x)  # [B, C, T]
        x = x.permute(2, 0, 1)  # [T, B, C]
        return x


class TransformerFFNLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        filter_size,
        padding="SAME",
        kernel_size=1,
        dropout=0.0,
        act="gelu",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        if padding == "SAME":
            self.ffn_1 = nn.Conv1d(
                hidden_size, filter_size, kernel_size, padding=kernel_size // 2
            )
        elif padding == "LEFT":
            self.ffn_1 = nn.Sequential(
                nn.ConstantPad1d((kernel_size - 1, 0), 0.0),
                nn.Conv1d(hidden_size, filter_size, kernel_size),
            )
        if self.act == "relu":
            self.act_fn = nn.ReLU()
        elif self.act == "gelu":
            self.act_fn = nn.GELU()
        elif self.act == "swish":
            self.act_fn = nn.SiLU()
        self.ffn_2 = XavierUniformInitLinear(filter_size, hidden_size)

    def forward(self, x):
        # x: T x B x C
        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size ** -0.5

        x = self.act_fn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class EncSALayer(nn.Module):
    def __init__(
        self,
        c,
        num_heads,
        dropout,
        attention_dropout=0.1,
        relu_dropout=0.1,
        kernel_size=9,
        padding="SAME",
        norm="ln",
        act="gelu",
    ):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        if num_heads > 0:
            if norm == "ln":
                self.layer_norm1 = nn.LayerNorm(c)
            elif norm == "bn":
                self.layer_norm1 = BatchNorm1dTBC(c)
            self.self_attn = nn.MultiheadAttention(
                self.c, num_heads, dropout=attention_dropout, bias=False,
            )
        if norm == "ln":
            self.layer_norm2 = nn.LayerNorm(c)
        elif norm == "bn":
            self.layer_norm2 = BatchNorm1dTBC(c)
        self.ffn = TransformerFFNLayer(
            c,
            4 * c,
            kernel_size=kernel_size,
            dropout=relu_dropout,
            padding=padding,
            act=act,
        )

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get("layer_norm_training", None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        if self.num_heads > 0:
            residual = x
            x = self.layer_norm1(x)
            x, _, = self.self_attn(
                query=x, key=x, value=x, key_padding_mask=encoder_padding_mask
            )
            x = F.dropout(x, self.dropout, training=self.training)
            x = residual + x
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout,
        kernel_size=None,
        padding="SAME",
        act="gelu",
        num_heads=2,
        norm="ln",
    ):
        super().__init__()
        self.op = EncSALayer(
            hidden_size,
            num_heads,
            dropout=dropout,
            attention_dropout=0.0,
            relu_dropout=dropout,
            kernel_size=kernel_size,
            padding=padding,
            norm=norm,
            act=act,
        )

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = (
            torch.stack(
                [torch.sin(position * div_term), torch.cos(position * div_term)], dim=2
            )
            .view(-1, self.d_model)
            .unsqueeze(0)
        )
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        return self.dropout(x) + self.dropout(pos_emb)


class FastSpeech2Encoder(nn.Module):
    DEFAULT_MAX_TARGET_POSITIONS = 2000
    """
    Args:
    
    
    """

    def __init__(
        self,
        embed_tokens,
        hidden_size,
        num_layers,
        ffn_kernel_size=9,
        ffn_padding="SAME",
        ffn_act="gelu",
        dropout=None,
        num_heads=2,
        use_last_norm=True,
        norm="ln",
        use_pos_embed=True,
        rel_pos=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self.hidden_size,
                    self.dropout,
                    kernel_size=ffn_kernel_size,
                    padding=ffn_padding,
                    act=ffn_act,
                    num_heads=num_heads,
                )
                for _ in range(self.num_layers)
            ]
        )
        if self.use_last_norm:
            if norm == "ln":
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == "bn":
                self.layer_norm = BatchNorm1dTBC(embed_dim)
        else:
            self.layer_norm = None

        self.embed_tokens = embed_tokens  # redundant, but have to persist for compatibility with old checkpoints
        self.embed_scale = math.sqrt(hidden_size)
        self.padding_idx = 0
        self.rel_pos = rel_pos
        if self.rel_pos:
            self.embed_positions = RelPositionalEncoding(hidden_size, dropout_rate=0.0)
        else:
            self.embed_positions = SinusoidalPositionalEmbedding(
                hidden_size, self.padding_idx, init_size=2000,
            )

    def forward_embedding(self, main_embed, extra_embed=None, padding_mask=None):
        # embed tokens and positions
        x = self.embed_scale * main_embed
        if extra_embed is not None:
            x = x + extra_embed
        if self.use_pos_embed:
            if self.rel_pos:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(~padding_mask)
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(
        self,
        main_embed,
        extra_embed,
        padding_mask,
        attn_mask=None,
        return_hiddens=False,
    ):
        x = self.forward_embedding(
            main_embed, extra_embed, padding_mask=padding_mask
        )  # [B, T, H]
        nonpadding_mask_TB = (
            1 - padding_mask.transpose(0, 1).float()[:, :, None]
        )  # [T, B, 1]

        # NOTICE:
        # The following codes are commented out because
        # `self.use_pos_embed` is always False in the older versions,
        # and this argument did not compat with `hparams['use_pos_embed']`,
        # which defaults to True. The new version fixed this inconsistency,
        # resulting in temporary removal of pos_embed_alpha, which has actually
        # never been used before.

        # if self.use_pos_embed:
        #     positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
        #     x = x + positions
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = (
                layer(x, encoder_padding_mask=padding_mask, attn_mask=attn_mask)
                * nonpadding_mask_TB
            )
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]
        return x


class FastSpeech2Acoustic(nn.Module):
    def __init__(self, hparams, vocab_size):
        super().__init__()
        self.txt_embed = NormalInitEmbedding(
            vocab_size, hparams["hidden_size"], PAD_INDEX
        )
        self.dur_embed = XavierUniformInitLinear(1, hparams["hidden_size"])
        self.encoder = FastSpeech2Encoder(
            self.txt_embed,
            hidden_size=hparams["hidden_size"],  # 256
            num_layers=hparams["enc_layers"],  # 4
            ffn_kernel_size=hparams["enc_ffn_kernel_size"],  # 9
            ffn_padding=hparams["ffn_padding"],  # SAME
            ffn_act=hparams["ffn_act"],  # gelu
            dropout=hparams["dropout"],
            num_heads=hparams["num_heads"],
            use_pos_embed=hparams["use_pos_embed"],
            rel_pos=hparams["rel_pos"],
        )

        # f0 embedding
        self.f0_embed_type = hparams.get("f0_embed_type", "discrete")
        if self.f0_embed_type == "discrete":
            self.pitch_embed = NormalInitEmbedding(
                300, hparams["hidden_size"], PAD_INDEX
            )
        elif self.f0_embed_type == "continuous":
            self.pitch_embed = XavierUniformInitLinear(1, hparams["hidden_size"])
        else:
            raise ValueError("f0_embed_type must be 'discrete' or 'continuous'.")

        # 采用的variance parameter
        self.variance_embed_list = []
        self.use_energy_embed = hparams.get("use_energy_embed", False)
        self.use_breathiness_embed = hparams.get("use_breathiness_embed", False)
        if self.use_energy_embed:
            self.variance_embed_list.append("energy")
        if self.use_breathiness_embed:
            self.variance_embed_list.append("breathiness")

        # 如果使用variance 模型
        # ['energy', 'breathiness']
        self.use_variance_embeds = len(self.variance_embed_list) > 0

        if self.use_variance_embeds:
            self.variance_embeds = nn.ModuleDict(
                {
                    v_name: XavierUniformInitLinear(1, hparams["hidden_size"])
                    for v_name in self.variance_embed_list
                }
            )

        # 移音
        self.use_key_shift_embed = hparams.get("use_key_shift_embed", False)
        if self.use_key_shift_embed:
            self.key_shift_embed = XavierUniformInitLinear(1, hparams["hidden_size"])
        # 速度
        self.use_speed_embed = hparams.get("use_speed_embed", False)
        if self.use_speed_embed:
            self.speed_embed = XavierUniformInitLinear(1, hparams["hidden_size"])
        # spk id embedding
        self.use_spk_id = hparams["use_spk_id"]
        if self.use_spk_id:
            self.spk_embed = NormalInitEmbedding(
                hparams["num_spk"], hparams["hidden_size"]
            )

    def forward_variance_embedding(
        self, condition, key_shift=None, speed=None, **variances
    ):
        if self.use_variance_embeds:
            variance_embeds = torch.stack(
                [
                    self.variance_embeds[v_name](variances[v_name][:, :, None])
                    for v_name in self.variance_embed_list
                ],
                dim=-1,
            ).sum(-1)
            condition += variance_embeds

        if self.use_key_shift_embed:
            key_shift_embed = self.key_shift_embed(key_shift[:, :, None])
            condition += key_shift_embed

        if self.use_speed_embed:
            speed_embed = self.speed_embed(speed[:, :, None])
            condition += speed_embed

        return condition

    @staticmethod
    def mel2ph_to_dur(mel2ph: torch.Tensor, T_txt, max_dur=None):
        B, _ = mel2ph.shape
        dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(
            1, mel2ph, torch.ones_like(mel2ph)
        )
        dur = dur[:, 1:]
        if max_dur is not None:
            dur = dur.clamp(max=max_dur)
        return dur

    def forward(
        self,
        txt_tokens,
        mel2ph,
        f0,
        key_shift=None,
        speed=None,
        spk_embed_id=None,
        **kwargs,
    ):
        # 文本编码
        txt_embed = self.txt_embed(txt_tokens)

        # 时长编码
        dur = self.mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float()
        dur_embed = self.dur_embed(dur[:, :, None])

        # FastSpeech2Encoder
        encoder_out = self.encoder(txt_embed, dur_embed, txt_tokens == 0)

        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])

        # mel2ph 1D tensor
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])

        condition = torch.gather(encoder_out, 1, mel2ph_)

        # 多说话人
        if self.use_spk_id:
            spk_mix_embed = kwargs.get("spk_mix_embed")
            if spk_mix_embed is not None:
                spk_embed = spk_mix_embed
            else:
                spk_embed = self.spk_embed(spk_embed_id)[:, None, :]
            condition += spk_embed

        if self.f0_embed_type == "discrete":
            pitch = f0_to_coarse(f0)
            pitch_embed = self.pitch_embed(pitch)
        else:
            f0_mel = (1 + f0 / 700).log()
            pitch_embed = self.pitch_embed(f0_mel[:, :, None])
        condition += pitch_embed

        condition = self.forward_variance_embedding(
            condition, key_shift=key_shift, speed=speed, **kwargs
        )

        return condition


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: Optional[float] = None,
        drop_out: float = 0.0,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.dropout = nn.Dropout(drop_out) if drop_out > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor,) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.dropout(x)

        x = residual + self.drop_path(x)
        return x


class ConvNeXtDecoder(nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        /,
        *,
        num_channels=512,
        num_layers=6,
        kernel_size=7,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.inconv = nn.Conv1d(
            in_dims, num_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )
        self.conv = nn.ModuleList(
            ConvNeXtBlock(
                dim=num_channels,
                intermediate_dim=num_channels * 4,
                layer_scale_init_value=1e-6,
                drop_out=dropout_rate,
            )
            for _ in range(num_layers)
        )
        self.outconv = nn.Conv1d(
            num_channels,
            out_dims,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

    # noinspection PyUnusedLocal
    def forward(self, x, infer=False):
        x = x.transpose(1, 2)
        x = self.inconv(x)
        for conv in self.conv:
            x = conv(x)
        x = self.outconv(x)
        x = x.transpose(1, 2)
        return x


def filter_kwargs(dict_to_filter, kwarg_obj):
    import inspect

    sig = inspect.signature(kwarg_obj)
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    ]
    filtered_dict = {
        filter_key: dict_to_filter[filter_key]
        for filter_key in filter_keys
        if filter_key in dict_to_filter
    }
    return filtered_dict


class AuxDecoderAdaptor(nn.Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_feats: int,
        spec_min: list,
        spec_max: list,
        aux_decoder_arch: str,
        aux_decoder_args: dict,
    ):
        super().__init__()
        kwargs = filter_kwargs(aux_decoder_args, ConvNeXtDecoder)

        self.decoder = ConvNeXtDecoder(
            in_dims=in_dims, out_dims=out_dims * num_feats, **kwargs
        )
        self.out_dims = out_dims
        self.n_feats = num_feats
        if spec_min is not None and spec_max is not None:
            # spec: [B, T, M] or [B, F, T, M]
            # spec_min and spec_max: [1, 1, M] or [1, 1, F, M] => transpose(-3, -2) => [1, 1, M] or [1, F, 1, M]
            spec_min = torch.FloatTensor(spec_min)[None, None, :].transpose(-3, -2)
            spec_max = torch.FloatTensor(spec_max)[None, None, :].transpose(-3, -2)
            self.register_buffer("spec_min", spec_min, persistent=False)
            self.register_buffer("spec_max", spec_max, persistent=False)

    def norm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.0
        b = (self.spec_max + self.spec_min) / 2.0
        return (x - b) / k

    def denorm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.0
        b = (self.spec_max + self.spec_min) / 2.0
        return x * k + b

    def forward(self, condition, infer=False):
        x = self.decoder(condition, infer=infer)  # [B, T, F x C]

        if self.n_feats > 1:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=2, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, x.shape[1], self.n_feats, self.out_dims)  # [B, T, F, C]
            x = x.transpose(1, 2)  # [B, F, T, C]
        if infer:
            x = self.denorm_spec(x)

        return x  # [B, T, C] or [B, F, T, C]


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            encoder_hidden, 2 * residual_channels, 1
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(
            y, [self.residual_channels, self.residual_channels], dim=1
        )
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(
            y, [self.residual_channels, self.residual_channels], dim=1
        )
        return (x + residual) / math.sqrt(2.0), skip



#################################################
#                   DDPM                        #
#################################################


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def linear_beta_schedule(timesteps, max_beta=0.01):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}


class KaiMingConv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class WaveNet(nn.Module):
    def __init__(
        self, hparams, in_dims, n_feats, *, n_layers=20, n_chans=256, n_dilates=4
    ):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = KaiMingConv1d(in_dims * n_feats, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4), nn.Mish(), nn.Linear(n_chans * 4, n_chans)
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    encoder_hidden=hparams["hidden_size"],
                    residual_channels=n_chans,
                    dilation=2 ** (i % n_dilates),
                )
                for i in range(n_layers)
            ]
        )
        self.skip_projection = KaiMingConv1d(n_chans, n_chans, 1)
        self.output_projection = KaiMingConv1d(n_chans, in_dims * n_feats, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """
        if self.n_feats == 1:
            x = spec.squeeze(1)  # [B, M, T]
        else:
            x = spec.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]
        x = self.input_projection(x)  # [B, C, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, M, T]
        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        hparams,
        out_dims,
        num_feats=1,
        timesteps=1000,
        k_step=1000,
        denoiser_type=None,
        denoiser_args=None,
        betas=None,
        spec_min=None,
        spec_max=None,
    ):
        super().__init__()
        self.denoise_fn: nn.Module = WaveNet(
            hparams, out_dims, num_feats, **denoiser_args
        )
        self.out_dims = out_dims
        self.num_feats = num_feats
        self.hparams = hparams
        if betas is not None:
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            betas = beta_schedule[hparams["schedule_type"]](timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.use_shallow_diffusion = hparams.get("use_shallow_diffusion", False)
        if self.use_shallow_diffusion:
            assert k_step <= timesteps, "K_step should not be larger than timesteps."
        self.timesteps = timesteps
        self.k_step = k_step if self.use_shallow_diffusion else timesteps
        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        # spec: [B, T, M] or [B, F, T, M]
        # spec_min and spec_max: [1, 1, M] or [1, 1, F, M] => transpose(-3, -2) => [1, 1, M] or [1, F, 1, M]
        spec_min = torch.FloatTensor(spec_min)[None, None, :out_dims].transpose(-3, -2)
        spec_max = torch.FloatTensor(spec_max)[None, None, :out_dims].transpose(-3, -2)
        self.register_buffer("spec_min", spec_min)
        self.register_buffer("spec_max", spec_max)

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, device = x.shape[0], x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_ddim(self, x, t, interval, cond):
        a_t = extract(self.alphas_cumprod, t, x.shape)
        a_prev = extract(
            self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape
        )

        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_prev = a_prev.sqrt() * (
            x / a_t.sqrt()
            + (((1 - a_prev) / a_prev).sqrt() - ((1 - a_t) / a_t).sqrt()) * noise_pred
        )
        return x_prev

    @torch.no_grad()
    def p_sample_plms(
        self, x, t, interval, cond, clip_denoised=True, repeat_noise=False
    ):
        """
        Use the PLMS method from
        [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(
                self.alphas_cumprod,
                torch.max(t - interval, torch.zeros_like(t)),
                x.shape,
            )
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            x_delta = (a_prev - a_t) * (
                (1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x
                - 1
                / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt()))
                * noise_t
            )
            x_pred = x + x_delta

            return x_pred

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, max(t - interval, 0), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (
                23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]
            ) / 12
        else:
            noise_pred_prime = (
                55 * noise_pred
                - 59 * noise_list[-1]
                + 37 * noise_list[-2]
                - 9 * noise_list[-3]
            ) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)

        return x_prev

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond)

        return x_recon, noise

    def inference(self, cond, b=1, x_start=None, device=None):
        depth = self.hparams.get("K_step_infer", self.k_step)
        noise = torch.randn(
            b, self.num_feats, self.out_dims, cond.shape[2], device=device
        )
        if self.use_shallow_diffusion:
            t_max = min(depth, self.k_step)
        else:
            t_max = self.k_step

        if t_max >= self.timesteps:
            x = noise
        elif t_max > 0:
            assert x_start is not None, "Missing shallow diffusion source."
            x = self.q_sample(
                x_start,
                torch.full((b,), t_max - 1, device=device, dtype=torch.long),
                noise,
            )
        else:
            assert x_start is not None, "Missing shallow diffusion source."
            x = x_start

        if (
            self.hparams.get("pndm_speedup")
            and self.hparams["pndm_speedup"] > 1
            and t_max > 0
        ):
            algorithm = self.hparams.get("diff_accelerator", "ddim")

            if algorithm == "ddim":
                iteration_interval = self.hparams["pndm_speedup"]
                for i in tqdm(
                    reversed(range(0, t_max, iteration_interval)),
                    desc="sample time step",
                    total=t_max // iteration_interval,
                    leave=False,
                ):
                    x = self.p_sample_ddim(
                        x,
                        torch.full((b,), i, device=device, dtype=torch.long),
                        iteration_interval,
                        cond=cond,
                    )
            else:
                raise NotImplementedError(algorithm)
        else:
            for i in tqdm(
                reversed(range(0, t_max)),
                desc="sample time step",
                total=t_max,
                leave=False,
            ):
                x = self.p_sample(
                    x, torch.full((b,), i, device=device, dtype=torch.long), cond
                )
        x = x.transpose(2, 3).squeeze(1)  # [B, F, M, T] => [B, T, M] or [B, F, T, M]
        return x

    def forward(self, condition, gt_spec=None, src_spec=None, infer=True):
        """
            conditioning diffusion, use fastspeech2 encoder output as the condition
        """
        cond = condition.transpose(1, 2)
        b, device = condition.shape[0], condition.device

        if not infer:
            # gt_spec: [B, T, M] or [B, F, T, M]
            spec = self.norm_spec(gt_spec).transpose(
                -2, -1
            )  # [B, M, T] or [B, F, M, T]
            if self.num_feats == 1:
                spec = spec[:, None, :, :]  # [B, F=1, M, T]
            t = torch.randint(0, self.k_step, (b,), device=device).long()
            return self.p_losses(spec, t, cond=cond)
        else:

            # src_spec: [B, T, M] or [B, F, T, M]
            if src_spec is not None:
                spec = self.norm_spec(src_spec).transpose(-2, -1)
                if self.num_feats == 1:
                    spec = spec[:, None, :, :]
            else:
                spec = None
            x = self.inference(cond, b=b, x_start=spec, device=device)
            return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


class ShallowDiffusionOutput:
    def __init__(self, *, aux_out=None, diff_out=None):
        self.aux_out = aux_out
        self.diff_out = diff_out


class DiffSingerAcoustic(nn.Module):
    def __init__(self, hparams, vocab_size, out_dims):
        super().__init__()

        self.fs2 = FastSpeech2Acoustic(hparams, vocab_size=vocab_size)

        self.use_shallow_diffusion = hparams.get("use_shallow_diffusion", False)
        self.shallow_args = hparams.get("shallow_diffusion_args", {})

        if self.use_shallow_diffusion:
            self.train_aux_decoder = self.shallow_args["train_aux_decoder"]
            self.train_diffusion = self.shallow_args["train_diffusion"]
            self.aux_decoder_grad = self.shallow_args["aux_decoder_grad"]
            self.aux_decoder = AuxDecoderAdaptor(
                # 256
                in_dims=hparams["hidden_size"],
                out_dims=out_dims,
                num_feats=1,
                spec_min=hparams["spec_min"],
                spec_max=hparams["spec_max"],
                aux_decoder_arch=self.shallow_args["aux_decoder_arch"],
                aux_decoder_args=self.shallow_args["aux_decoder_args"],
            )

        self.diffusion = GaussianDiffusion(
            hparams,
            out_dims=out_dims,
            num_feats=1,
            timesteps=hparams["timesteps"],
            k_step=hparams["K_step"],
            denoiser_type=hparams["diff_decoder_type"],
            denoiser_args={
                "n_layers": hparams["residual_layers"],
                "n_chans": hparams["residual_channels"],
                "n_dilates": hparams["dilation_cycle_length"],
            },
            spec_min=hparams["spec_min"],
            spec_max=hparams["spec_max"],
        )

    def forward(
        self,
        txt_tokens,
        mel2ph,
        f0,
        key_shift=None,
        speed=None,
        spk_embed_id=None,
        gt_mel=None,
        infer=True,
        **kwargs,
    ) -> ShallowDiffusionOutput:
        condition = self.fs2(
            txt_tokens,
            mel2ph,
            f0,
            key_shift=key_shift,
            speed=speed,
            spk_embed_id=spk_embed_id,
            **kwargs,
        )
        if infer:
            if self.use_shallow_diffusion:
                aux_mel_pred = self.aux_decoder(condition, infer=True)
                aux_mel_pred *= (mel2ph > 0).float()[:, :, None]
                if gt_mel is not None and self.shallow_args["val_gt_start"]:
                    src_mel = gt_mel
                else:
                    src_mel = aux_mel_pred
            else:
                aux_mel_pred = src_mel = None
            mel_pred = self.diffusion(condition, src_spec=src_mel, infer=True)
            mel_pred *= (mel2ph > 0).float()[:, :, None]
            return ShallowDiffusionOutput(aux_out=aux_mel_pred, diff_out=mel_pred)
        else:
            if self.use_shallow_diffusion:
                if self.train_aux_decoder:
                    aux_cond = condition * self.aux_decoder_grad + condition.detach() * (
                        1 - self.aux_decoder_grad
                    )
                    aux_out = self.aux_decoder(aux_cond, infer=False)
                else:
                    aux_out = None

                if self.train_diffusion:
                    x_recon, noise = self.diffusion(
                        condition, gt_spec=gt_mel, infer=False
                    )
                    diff_out = (x_recon, noise)
                else:
                    diff_out = None
                return ShallowDiffusionOutput(aux_out=aux_out, diff_out=diff_out)

            else:
                aux_out = None
                x_recon, noise = self.diffusion(condition, gt_spec=gt_mel, infer=False)
                return ShallowDiffusionOutput(
                    aux_out=aux_out, diff_out=(x_recon, noise)
                )


def resample_align_curve(
    points: np.ndarray,
    original_timestep: float,
    target_timestep: float,
    align_length: int,
):
    t_max = (len(points) - 1) * original_timestep
    curve_interp = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(len(points)),
        points,
    ).astype(points.dtype)
    delta_l = align_length - len(curve_interp)
    if delta_l < 0:
        curve_interp = curve_interp[:align_length]
    elif delta_l > 0:
        curve_interp = np.concatenate(
            (curve_interp, np.full(delta_l, fill_value=curve_interp[-1])), axis=0
        )
    return curve_interp


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx : a.shape[0]] = (1 - k) * a[idx:] + k * b[:fade_len]
    np.copyto(dst=result[a.shape[0] :], src=b[fade_len:])
    return result


def trans_f0_seq(feature_pit, transform):
    feature_pit = feature_pit * 2 ** (transform / 12)
    return round(feature_pit, 1)


def trans_key(raw_data, key):
    warning_tag = False
    for i in raw_data:
        note_seq_list = i["note_seq"].split(" ")
        new_note_seq_list = []
        for note_seq in note_seq_list:
            if note_seq != "rest":
                new_note_seq = librosa.midi_to_note(
                    librosa.note_to_midi(note_seq) + key, unicode=False
                )
                # new_note_seq = move_key(note_seq, key)
                new_note_seq_list.append(new_note_seq)
            else:
                new_note_seq_list.append(note_seq)
        i["note_seq"] = " ".join(new_note_seq_list)
        if i.get("f0_seq"):
            f0_seq_list = i["f0_seq"].split(" ")
            f0_seq_list = [float(x) for x in f0_seq_list]
            new_f0_seq_list = []
            for f0_seq in f0_seq_list:
                new_f0_seq = trans_f0_seq(f0_seq, key)
                new_f0_seq_list.append(str(new_f0_seq))
            i["f0_seq"] = " ".join(new_f0_seq_list)
        else:
            warning_tag = True
    if warning_tag:
        print(
            "Warning: parts of f0_seq do not exist, please freeze the pitch line in the editor.\r\n"
        )
    return raw_data


class DiffSingerAcousticInfer(nn.Module):
    spk_map = {"opencpop": 0}

    def __init__(
        self, hparams, model_ckpt_path, device=None,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hparams = hparams
        self.device = device
        self.timestep = hparams["hop_size"] / hparams["audio_sample_rate"]

        self.variance_checklist = []

        self.variances_to_embed = set()

        ########## pass ##########
        if hparams.get("use_energy_embed", False):
            self.variances_to_embed.add("energy")
        if hparams.get("use_breathiness_embed", False):
            self.variances_to_embed.add("breathiness")

        ##########################
        global VOCAB_LIST, NSFHIFIGAN_CONFIG
        # 构建phoneme set
        self.ph_encoder = TokenTextEncoder(hparams, vocab_list=VOCAB_LIST)
        # 多说话人
        if hparams["use_spk_id"]:
            assert (
                isinstance(self.spk_map, dict) and len(self.spk_map) > 0
            ), "Invalid or empty speaker map!"
            assert len(self.spk_map) == len(
                set(self.spk_map.values())
            ), "Duplicate speaker id in speaker map!"

        # 构建模型

        model_ckpt = torch.load(model_ckpt_path, map_location="cpu")

        self.model = DiffSingerAcoustic(
            hparams,
            vocab_size=len(self.ph_encoder),
            out_dims=hparams["audio_num_mel_bins"],
        )
        self.model.load_state_dict(model_ckpt["diff"])
        self.model.eval().to(self.device)

        self.lr = LengthRegulator().to(self.device)

        self.vocoder = NsfHifiGAN(NSFHIFIGAN_CONFIG)
        self.vocoder.load_state_dict(model_ckpt["nsf"])
        self.vocoder.eval().to(self.device)

    def preprocess_input(self, param, idx=0):
        """
        :param param: one segment in the .ds file
        :param idx: index of the segment
        :return: batch of the model inputs
        """
        ###################  又重新创建了一遍 token
        batch = {}
        summary = OrderedDict()
        txt_tokens = torch.LongTensor([self.ph_encoder.encode(param["ph_seq"])]).to(
            self.device
        )  # => [B, T_txt]
        # NOTE[tokens]:
        batch["tokens"] = txt_tokens

        ph_dur = torch.from_numpy(np.array(param["ph_dur"].split(), np.float32)).to(
            self.device
        )
        ph_acc = torch.round(torch.cumsum(ph_dur, dim=0) / self.timestep + 0.5).long()
        durations = torch.diff(
            ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device)
        )[
            None
        ]  # => [B=1, T_txt]
        mel2ph = self.lr(durations, txt_tokens == 0)  # => [B=1, T]
        # NOTE[mel2ph]:
        batch["mel2ph"] = mel2ph
        length = mel2ph.size(1)  # => T

        summary["tokens"] = txt_tokens.size(1)
        summary["frames"] = length
        summary["seconds"] = "%.2f" % (length * self.timestep)
        # NOTE[f0]:
        batch["f0"] = torch.from_numpy(
            resample_align_curve(
                np.array(param["f0_seq"].split(), np.float32),
                original_timestep=float(param["f0_timestep"]),
                target_timestep=self.timestep,
                align_length=length,
            )
        ).to(self.device)[None]

        return batch

    @torch.no_grad()
    def forward_model(self, sample):
        txt_tokens = sample["tokens"]
        variances = {v_name: sample.get(v_name) for v_name in self.variances_to_embed}
        if self.hparams["use_spk_id"]:
            spk_mix_id = sample["spk_mix_id"]
            spk_mix_value = sample["spk_mix_value"]
            # perform mixing on spk embed
            spk_mix_embed = torch.sum(
                self.model.fs2.spk_embed(spk_mix_id)
                * spk_mix_value.unsqueeze(3),  # => [B, T, N, H]
                dim=2,
                keepdim=False,
            )  # => [B, T, H]
        else:
            spk_mix_embed = None
        mel_pred: ShallowDiffusionOutput = self.model(
            txt_tokens,
            mel2ph=sample["mel2ph"],
            f0=sample["f0"],
            **variances,
            key_shift=sample.get("key_shift"),
            speed=sample.get("speed"),
            spk_mix_embed=spk_mix_embed,
            infer=True,
        )
        return mel_pred.diff_out

    @torch.no_grad()
    def run_vocoder(self, spec, **kwargs):
        y = self.vocoder(spec, **kwargs)
        return y[None]

    def run_inference(
        self,
        params,
        out_dir,  # 输出文件名
        num_runs: int = 1,
        spk_mix: Dict[str, float] = None,
        seed: int = -1,
        save_mel: bool = False,
        save_audio: bool = True,
    ):
        # print("=============  Reading ds to batch data =================")
        batches = [
            self.preprocess_input(param, idx=i) for i, param in enumerate(params)
        ]

        # print(f"batches:\t{batches[0]}")
        Path(out_dir).parent.mkdir(parents=True, exist_ok=True)

        for i in range(num_runs):
            result = np.zeros(0)
            current_length = 0

            for param, batch in tqdm(
                zip(params, batches), desc="Stage 1 Infer", total=len(params)
            ):
                if "seed" in param:
                    torch.manual_seed(param["seed"] & 0xFFFF_FFFF)
                    torch.cuda.manual_seed_all(param["seed"] & 0xFFFF_FFFF)
                elif seed >= 0:
                    torch.manual_seed(seed & 0xFFFF_FFFF)
                    torch.cuda.manual_seed_all(seed & 0xFFFF_FFFF)

                mel_pred = self.forward_model(batch)
                if save_mel:
                    result.append(
                        {
                            "offset": param.get("offset", 0.0),
                            "mel": mel_pred.cpu(),
                            "f0": batch["f0"].cpu(),
                        }
                    )
                else:
                    waveform_pred = (
                        self.run_vocoder(mel_pred, f0=batch["f0"])[0].cpu().numpy()
                    )
                    silent_length = (
                        round(
                            param.get("offset", 0) * self.hparams["audio_sample_rate"]
                        )
                        - current_length
                    )
                    if silent_length >= 0:
                        result = np.append(result, np.zeros(silent_length))
                        result = np.append(result, waveform_pred)
                    else:
                        result = cross_fade(
                            result, waveform_pred, current_length + silent_length
                        )
                    current_length = (
                        current_length + silent_length + waveform_pred.shape[0]
                    )

            if save_audio:
                sf.write(out_dir, result, self.hparams["audio_sample_rate"], "PCM_16")
        return result


#########################################
#               Remix                   #
#########################################


class Remixer:
    def __init__(self, sample_rate=44100, keep_accompaniment_stereo=True):
        self.sample_rate = sample_rate
        self.keep_accompaniment_stereo = keep_accompaniment_stereo

    def analyze_vocal_frequency(self, vocal_signal):
        # Compute Short-Time Fourier Transform (STFT)
        S = np.abs(librosa.stft(vocal_signal, n_fft=2048, hop_length=512))

        # Compute frequency axis
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)

        # Compute the mean of the spectrum across time
        S_mean = np.mean(S, axis=1)

        # Normalize the spectrum
        S_mean_norm = S_mean / np.sum(S_mean)

        # Find the frequency range where cumulative energy reaches certain thresholds
        cumulative_energy = np.cumsum(S_mean_norm)

        # Set thresholds, 5% to 95% of the energy
        lower_idx = np.where(cumulative_energy >= 0.05)[0][0]
        upper_idx = np.where(cumulative_energy <= 0.95)[0][-1]

        vocal_freq_low = freqs[lower_idx]
        vocal_freq_high = freqs[upper_idx]

        return vocal_freq_low, vocal_freq_high

    def match_audio_length(self, vocal_signal, accompaniment_signal):
        """
        Match the lengths of vocal and accompaniment signals by padding the shorter one with silence (zeros).
        The resulting signals will both have length equal to the longest input signal.
        
        Args:
            vocal_signal: numpy array of vocal audio (1D or 2D)
            accompaniment_signal: numpy array of accompaniment audio (1D or 2D, <= 2 channels)
            
        Returns:
            Tuple of (vocal_signal, accompaniment_signal) with matched lengths
            
        Raises:
            AssertionError: If accompaniment has invalid shape
        """
        # Validate input shapes
        assert (
            len(accompaniment_signal.shape) == 2 and accompaniment_signal.shape[1] <= 2
        ), "Accompaniment signal must be 2D with <= 2 channels"

        vocal_length = vocal_signal.shape[0]
        accompaniment_length = accompaniment_signal.shape[0]
        max_length = max(vocal_length, accompaniment_length)

        # Pad vocal if it's shorter
        if vocal_length < max_length:
            pad_length = max_length - vocal_length
            if len(vocal_signal.shape) == 1:  # Mono
                vocal_signal = np.pad(vocal_signal, (0, pad_length), mode="constant")
            else:  # Stereo
                vocal_signal = np.pad(
                    vocal_signal, ((0, pad_length), (0, 0)), mode="constant"
                )

        # Pad accompaniment if it's shorter
        if accompaniment_length < max_length:
            pad_length = max_length - accompaniment_length
            if accompaniment_signal.shape[1] == 1:  # Mono
                accompaniment_signal = np.pad(
                    accompaniment_signal, ((0, pad_length), (0, 0)), mode="constant"
                )
            else:  # Stereo
                accompaniment_signal = np.pad(
                    accompaniment_signal, ((0, pad_length), (0, 0)), mode="constant"
                )

        return vocal_signal, accompaniment_signal

    def mix(self, vocal_signal, accompaniment_signal):
        # Ensure both audio signals have the same sample rate
        sr = self.sample_rate
        name = vocal_signal["name"]
        sample_rate = vocal_signal["sample_rate"]
        vocal_signal = vocal_signal["waveform"]
        accompaniment_signal = accompaniment_signal["waveform"]

        # Convert vocal to mono (always)
        if vocal_signal.ndim > 1:
            assert vocal_signal.shape[1] == 2
            vocal_signal = np.mean(vocal_signal, axis=1)  # Convert vocal to mono

        # Convert accompaniment based on user preference
        if accompaniment_signal.ndim > 1:
            assert accompaniment_signal.shape[1] == 2
            if not self.keep_accompaniment_stereo:
                accompaniment_signal = np.mean(
                    accompaniment_signal, axis=1
                )  # Convert to mono if requested

        # Match accompaniment length to the vocal length
        vocal_signal, accompaniment_signal = self.match_audio_length(
            vocal_signal, accompaniment_signal
        )

        # Analyze vocal frequency range
        vocal_freq_low, vocal_freq_high = self.analyze_vocal_frequency(vocal_signal)
        print(
            f"Vocal frequency range: {vocal_freq_low:.2f} Hz - {vocal_freq_high:.2f} Hz"
        )

        # Process vocal
        vocal_board = Pedalboard(
            [
                HighpassFilter(cutoff_frequency_hz=40.0),
                LowpassFilter(cutoff_frequency_hz=12000.0),
                Reverb(room_size=0.1, wet_level=0, dry_level=0.5),
            ]
        )
        processed_vocal = vocal_board(vocal_signal, sr)

        # Process accompaniment (sidechain EQ)
        tfm = sox.Transformer()
        center_freq = (vocal_freq_low + vocal_freq_high) / 2
        q_width = center_freq / (vocal_freq_high - vocal_freq_low)
        tfm.equalizer(frequency=center_freq, width_q=q_width, gain_db=-3.0)

        # Handle stereo accompaniment processing
        if accompaniment_signal.ndim == 1:  # Mono
            processed_accompaniment = tfm.build_array(
                input_array=accompaniment_signal, sample_rate_in=sr
            )
        else:  # Stereo
            # Process each channel separately
            left_channel = tfm.build_array(
                input_array=accompaniment_signal[:, 0], sample_rate_in=sr
            )
            right_channel = tfm.build_array(
                input_array=accompaniment_signal[:, 1], sample_rate_in=sr
            )
            processed_accompaniment = np.stack([left_channel, right_channel], axis=-1)

        # Mix the audio
        min_length = min(len(processed_vocal), len(processed_accompaniment))

        if processed_accompaniment.ndim == 1:  # Mono accompaniment
            mixed_audio = (
                processed_vocal[:min_length] + processed_accompaniment[:min_length]
            )
        else:  # Stereo accompaniment
            # Expand vocal to stereo by duplicating channel
            vocal_stereo = np.stack(
                [processed_vocal[:min_length], processed_vocal[:min_length]], axis=-1
            )
            mixed_audio = vocal_stereo + processed_accompaniment[:min_length]

        # Add a limiter
        master_board = Pedalboard(
            [Compressor(threshold_db=-20, ratio=2.0), Limiter(threshold_db=-5.0)]
        )

        # Apply the limiter (handles both mono and stereo)
        final_audio = master_board(mixed_audio, sr)

        # Normalize the volume
        peak = np.max(np.abs(final_audio))
        if peak > 0:
            final_audio = final_audio / peak * 0.99  # Prevent clipping

        # Ensure output matches input accompaniment format
        if self.keep_accompaniment_stereo and final_audio.ndim == 1:
            final_audio = np.stack([final_audio, final_audio], axis=-1)

        return {
            "waveform": final_audio,
            "name": name,
            "sample_rate": sample_rate,
        }


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


#########################################
#               infer                   #
#########################################


class ReplaceLyrics:
    def __init__(self):
        self._init_()

    def _init_(self):
        """Initialize internal phoneme mapping and classification of vowels/consonants."""
        global dictionary
        # Create mapping from pinyin to phonemes using the provided dictionary
        self._pinyin_to_phoneme_map = {
            item.split("\t")[0]: item.split("\t")[1].strip().split(" ")
            for item in dictionary.split("\n")
        }

        # Initialize sets for vowels and consonants
        self._vowels = {"SP", "AP"}  # Special tokens for silence/pause
        self._consonants = set()  # Will store consonant phonemes

        # Extract phoneme rules from dictionary
        self._rules = [
            item.split("\t")[1].strip().split(" ") for item in dictionary.split("\n")
        ]

        # Classify each phoneme as vowel or consonant
        for phonemes in self._rules:
            # Ensure dictionary only contains 1 or 2 phonemes per entry
            assert (
                len(phonemes) <= 2
            ), "We only support two-phase dictionaries for automatically adding ph_num."

            if len(phonemes) == 1:
                self._vowels.add(phonemes[0])  # Single phoneme is a vowel
            else:
                self._consonants.add(phonemes[0])  # First phoneme is consonant
                self._vowels.add(phonemes[1])  # Second phoneme is vowel

    @property
    def pinyin_to_phoneme_map(self):
        """Getter for the pinyin-to-phoneme mapping dictionary."""
        return self._pinyin_to_phoneme_map

    @property
    def consonants(self):
        """Getter for the set of consonant phonemes."""
        return self._consonants

    def _convert_pinyin_to_phonemes(self, text):
        """
        Convert a space-separated pinyin string to a sequence of phonemes.
        
        Args:
            text: Input text containing space-separated pinyin syllables
            
        Returns:
            List of phonemes corresponding to the input pinyin
        """
        pinyin_list = text.split()
        phoneme_seq = []

        for pinyin in pinyin_list:
            if pinyin != ",":  # Skip comma separators
                # Get phonemes for this pinyin or use pinyin itself if not found
                phonemes = self.pinyin_to_phoneme_map.get(pinyin, [pinyin])
                phoneme_seq.extend(phonemes)

        return phoneme_seq

    @staticmethod
    def _assign_phoneme_duration(
        original_phoneme_durations: list[float],
        original_word_to_phoneme: list[int],
        original_word_durations: list[float],
        new_word_to_phoneme: list[int],
    ) -> list[float]:
        """
        Reassign phoneme durations based on new word-to-phoneme mappings while preserving total duration.
        
        Args:
            original_phoneme_durations: List of original phoneme durations
            original_word_to_phoneme: List mapping words to number of phonemes in original
            original_word_durations: List of original word durations
            new_word_to_phoneme: New word-to-phoneme mapping to adjust durations for
            
        Returns:
            List of adjusted phoneme durations matching the new_word_to_phoneme mapping
            
        Raises:
            ValueError: If input lengths don't match or invalid phoneme counts are encountered
        """
        # Validate input lengths
        if not (
            len(original_word_to_phoneme)
            == len(new_word_to_phoneme)
            == len(original_word_durations)
        ):
            raise ValueError("Input lists must have the same length")

        # Create a copy to avoid modifying the original list
        remaining_phoneme_durations = original_phoneme_durations.copy()
        new_phoneme_durations = []

        for original_phoneme_count, new_phoneme_count, word_duration in zip(
            original_word_to_phoneme, new_word_to_phoneme, original_word_durations
        ):
            # Case 1: Original had 2 phonemes, new has 1 - combine durations
            if original_phoneme_count == 2 and new_phoneme_count == 1:
                combined_duration = remaining_phoneme_durations.pop(
                    0
                ) + remaining_phoneme_durations.pop(0)
                new_phoneme_durations.append(combined_duration)

            # Case 2: Original had 1 phoneme, new has 2 - split duration randomly
            elif original_phoneme_count == 1 and new_phoneme_count == 2:
                original_duration = remaining_phoneme_durations.pop(0)
                # Split duration with 10-30% to first phoneme, rest to second
                first_part = np.random.uniform(
                    0.1 * original_duration, 0.3 * original_duration
                )
                second_part = original_duration - first_part
                new_phoneme_durations.extend([first_part, second_part])

            # Case 3: Phoneme counts match or other cases - preserve original durations
            else:
                if original_phoneme_count == 2:
                    # Add phonemes in reverse order since we're popping from end
                    new_phoneme_durations.extend(
                        [
                            remaining_phoneme_durations.pop(0),
                            remaining_phoneme_durations.pop(0),
                        ]
                    )
                elif original_phoneme_count == 1:
                    new_phoneme_durations.append(remaining_phoneme_durations.pop(0))
                else:
                    raise ValueError(f"Invalid phoneme count: {original_phoneme_count}")
        return new_phoneme_durations

    @staticmethod
    def _find_and_replace_lyrics(
        original_entries: List[Dict], modified_text: str, segment_token="*"
    ) -> List[Tuple[int, Dict]]:
        """
        Find and replace lyrics segments while preserving SP/AP markers.
        
        Args:
            original_entries: List of original lyric entries with text, durations, etc.
            modified_text: New lyrics text with segments marked by segment_token
            segment_token: Character used to mark replacement segments (default "*")
            
        Returns:
            List of tuples containing (index, modified_entry) for changed lyrics
            
        Raises:
            ValueError: If segment markers are unbalanced or text doesn't match original
        """
        # Preprocess original text (split into words list, excluding SP/AP)
        original_words = []
        entry_word_ranges = []  # Stores start/end indices of words for each entry

        for entry in original_entries:
            entry_words = [w for w in entry["text"].split() if w not in ["SP", "AP"]]
            start = len(original_words)
            original_words.extend(entry_words)
            end = len(original_words)
            entry_word_ranges.append((start, end))

        # Split modified text into alternating preserved/replaced segments
        modified_parts = modified_text.split(segment_token)
        if len(modified_parts) % 2 == 0:
            raise ValueError("Unmatched * in modified text")

        # Build complete modified words list
        modified_words = []
        is_replacement = False  # Tracks whether current segment is replacement

        for idx, part in enumerate(modified_parts):
            part = part.strip()

            words = part.split()
            if is_replacement:
                # Replacement segment - use new words directly
                modified_words.extend(words)
            else:
                # Preserved segment - must match original text
                expected_words = original_words[
                    len(modified_words) : len(modified_words) + len(words)
                ]
                if words != expected_words:
                    raise ValueError(
                        f"Original text mismatch at position {len(modified_words)}: "
                        f"expected '{' '.join(expected_words)}', got '{' '.join(words)}'"
                    )
                modified_words.extend(words)

            is_replacement = not is_replacement

        # Verify total word count matches original
        if len(modified_words) != len(original_words):
            raise ValueError(
                f"Total word count mismatch. Original: {len(original_words)}, Modified: {len(modified_words)}"
            )

        def restore_sp_ap(original_text: str, new_words: List[str]) -> str:
            """
            Restore SP/AP markers in new text using original text as template.
            
            Args:
                original_text: Original text with SP/AP markers
                new_words: New words without markers
                
            Returns:
                New text with SP/AP markers inserted at original positions
                
            Raises:
                ValueError: If word counts don't match between original and new
            """
            original_words = original_text.split()
            new_word_iter = iter(new_words)

            result = []
            for word in original_words:
                if word in ["SP", "AP"]:
                    result.append(word)  # Preserve special tokens
                else:
                    try:
                        result.append(next(new_word_iter))  # Insert next new word
                    except StopIteration:
                        raise ValueError("Not enough words in new text")

            # Verify all new words were used
            try:
                next(new_word_iter)
                raise ValueError("Too many words in new text")
            except StopIteration:
                pass

            return " ".join(result)

        # Collect all entries that were modified
        results = []
        for entry_idx, (start, end) in enumerate(entry_word_ranges):
            original_entry_words = original_words[start:end]
            modified_entry_words = modified_words[start:end]

            if original_entry_words != modified_entry_words:
                # Rebuild text with original SP/AP markers
                new_text = restore_sp_ap(
                    original_entries[entry_idx]["text"], modified_entry_words
                )
                new_entry = original_entries[entry_idx].copy()
                new_entry["text"] = new_text
                results.append((entry_idx, new_entry))

        return results

    def process(
        self,
        ds: Union[str, list],
        new_lyric: Union[str, list],
        out_ds=Union[str, list],
        segment="*",
    ):
        """Main processing method to replace lyrics in dataset files.
        
        Args:
            ds: Input dataset file path(s)
            new_lyric: New lyric file path(s)
            out_ds: Output file path(s)
            segment: Segment marker character (default "*")
        """
        # Convert single paths to lists for uniform processing
        if isinstance(new_lyric, str):
            new_lyric = [new_lyric]
        if isinstance(ds, str):
            ds = [ds]
        if isinstance(out_ds, str):
            out_ds = [out_ds]

        # Process each lyric file
        for idx, lrc in tqdm(enumerate(new_lyric), total=len(new_lyric)):
            with open(lrc, "r") as f_lrc:
                modified_lyrics = f_lrc.readlines()[0]

            with open(ds[idx], "r") as f_ds:
                original_json = json.load(f_ds)

            # Find and replace marked segments
            replacements = self._find_and_replace_lyrics(
                original_json, modified_lyrics, segment
            )

            # Process each replacement
            for replace_idx, new_entry in replacements:
                # Extract original timing information
                original_phoneme_dur = list(map(float, new_entry["ph_dur"].split()))
                original_word2ph = list(map(int, new_entry["word2ph"].split()))
                original_word_dur = list(map(float, new_entry["word_dur"].split()))

                # Convert new text to IPA phonemes
                _, adjusted_word_seq = self.chinese_to_ipa(new_entry["text"])
                new_word2ph = [
                    len(self.pinyin_to_phoneme_map[i]) for i in adjusted_word_seq
                ]

                # Get phoneme sequence for new text
                adjusted_phoneme_seq = self._convert_pinyin_to_phonemes(
                    " ".join(adjusted_word_seq)
                )

                # Adjust phoneme durations for new phoneme sequence
                new_phoneme_dur = self._assign_phoneme_duration(
                    original_phoneme_dur,
                    original_word2ph,
                    original_word_dur,
                    new_word2ph,
                )

                # Calculate phoneme group counts
                ph_num = []
                i = 0
                while i < len(adjusted_phoneme_seq):
                    j = i + 1
                    # Group consecutive consonants with following vowel
                    while (
                        j < len(adjusted_phoneme_seq)
                        and adjusted_phoneme_seq[j] in self.consonants
                    ):
                        j += 1
                    ph_num.append(str(j - i))
                    i = j

                # Update entry with new phonetic information
                new_entry["word_seq"] = " ".join(adjusted_word_seq)
                new_entry["word2ph"] = " ".join(map(str, new_word2ph))
                new_entry["ph_seq"] = " ".join(adjusted_phoneme_seq)
                new_entry["ph_dur"] = " ".join(map(str, new_phoneme_dur))
                new_entry["ph_num"] = " ".join(ph_num)
                new_entry["is_replace"] = True
                original_json[replace_idx] = new_entry

            # Save modified JSON
            with open(out_ds[idx], "w") as file:
                json.dump(original_json, file, ensure_ascii=False, indent=4)

    def _number_to_chinese(self, text):
        """Convert numbers in text to Chinese character representation."""
        numbers = re.findall(r"\d+(?:\.?\d+)?", text)
        for number in numbers:
            text = text.replace(number, cn2an.an2cn(number), 1)
        return text

    def _add_spaces_around_chinese(self, text):
        """Add spaces around Chinese characters for proper tokenization."""
        pattern = re.compile(r"([\u4e00-\u9fff])")
        modified_text = pattern.sub(r" \1 ", text)
        modified_text = re.sub(r"\s+", " ", modified_text).strip()
        return modified_text

    def _chinese_to_pinyin(self, text):
        """Convert Chinese text to pinyin using jieba for word segmentation."""
        text = re.sub(r"[^\w\s]|_", "", text)  # Remove punctuation
        words = jieba.lcut(text, cut_all=False)  # Chinese word segmentation
        raw_text = []
        g2p_text = []

        for word in words:
            word = word.strip()
            if word == "":
                continue
            pinyins = lazy_pinyin(word)  # Convert to pinyin
            if len(pinyins) == 1:
                raw_text.append(word)
                g2p_text.append(pinyins[0])
            else:
                # Handle multi-character words
                word = self._add_spaces_around_chinese(word).split(" ")
                assert len(pinyins) == len(word), print(word, pinyins)
                for _pinyins, _word in zip(pinyins, word):
                    raw_text.append(_word)
                    g2p_text.append(_pinyins)
        return raw_text, g2p_text
    
    def chinese_to_ipa(self, text):
        """Convert Chinese text to IPA phoneme representation through pinyin."""
        text = self._number_to_chinese(text)  # Convert numbers first
        raw_text, g2p_text = self._chinese_to_pinyin(text)
        return raw_text, g2p_text


class SingingVoiceSynthesis:
    """ 
    Singing Voice Synthesis (SVS) and Singing Voice Conversion (SVC) system based on DiffSinger and Seed-VC.
    
    This class combines two state-of-the-art models for high-quality singing voice synthesis and conversion.
    
    # Introduction
        - DiffSinger: A diffusion-based singing voice synthesis model
        - Seed-VC: A singing voice conversion model with content/style disentanglement
        - Supports both standalone SVS and SVC, or combined SVS+SVC pipeline
    
    # Basic Usage Examples
    
    >>> # Initialize the model with checkpoints
    >>> model = SingingVoiceSynthesis(
        "checkpoints/svs/model_v1.pt",  # DiffSinger checkpoint
        "checkpoints/svs/model_v2.pt",  # Seed-VC checkpoint
        "checkpoints/whisper-small/")  # Whisper feature extractor
    
    >>> # Full SVS + SVC pipeline (text-to-singing with voice conversion)
    >>> model('sample.ds', 'out.wav', 'reference.wav')
    
    >>> # Standalone SVS (text-to-singing only)
    >>> model('sample.ds', 'out.wav', mode='svs')
    
    >>> # Standalone SVC (voice conversion only)
    >>> model('input.wav', 'out.wav', 'reference.wav', mode='svc')
    
    # Advanced Parameters
    >>> model('input.ds', 'out.wav', 'ref.wav',
    ...       diffusion_steps=50,  # More steps = better quality but slower
    ...       length_adjust=1.1,   # Stretch duration by 10%
    ...       cfg_rate=0.7,        # Classifier-free guidance strength
    ...       pitch_shift_svs=2,   # Shift SVS output pitch by +2 semitones
    ...       pitch_shift_svc=-1,  # Shift SVC output pitch by -1 semitone
    ...       mode='svs_svc')      # Pipeline mode
    """

    def __init__(
        self,
        v1_model_path: str = None,  # diffsinger
        v2_model_path: str = None,  # seed-vc
        whisper_model_path: str = None,
        vocoder_type: Literal["bigvgan", "hifigan", "vocos"] = "bigvgan",
        fp16: bool = True,
    ):
        print(f"Load checkpoints...")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.device = device
        self.fp16 = fp16
        
        # Load DiffSinger
        if v1_model_path is not None:
            self.diffsinger = DiffSingerAcousticInfer(
                DIFFSINGER_CONFIG, v1_model_path, device
            )
        
        # Load Seed-VC
        if v2_model_path is not None:
            model_v2_ckpt = torch.load(v2_model_path, map_location="cpu")
            f0_extractor = RMVPE(model_v2_ckpt["rmvpe"], device=device)
            self.f0_fn = f0_extractor.infer_from_audio

            # Load Seed-VC
            seed_vc_params = Munch(
                (k, recursive_munch(v)) for k, v in SEED_VC_CONFIG.items()
            )
            seed_vc_params = seed_vc_params.model_params
            seed_vc_params.dit_type = "DiT"

            lr_params = seed_vc_params.length_regulator
            self.seed_vc = Munch(
                cfm=CFM(seed_vc_params),
                length_regulator=InterpolateRegulator(
                    channels=lr_params.channels,
                    sampling_ratios=lr_params.sampling_ratios,
                    is_discrete=lr_params.is_discrete,
                    in_channels=lr_params.in_channels
                    if hasattr(lr_params, "in_channels")
                    else None,
                    codebook_size=lr_params.content_codebook_size,
                    f0_condition=lr_params.f0_condition
                    if hasattr(lr_params, "f0_condition")
                    else False,
                    n_f0_bins=lr_params.n_f0_bins
                    if hasattr(lr_params, "n_f0_bins")
                    else 512,
                ),
            )
            self.seed_vc = self._load_dit_checkpoint(self.seed_vc, model_v2_ckpt["dit"])
            for key in self.seed_vc:
                self.seed_vc[key].eval()
                self.seed_vc[key].to(device)
            self.seed_vc.cfm.estimator.setup_caches(
                max_batch_size=1, max_seq_length=8192
            )

            # Load Spk Embedding Model
            self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
            self.campplus_model.load_state_dict(model_v2_ckpt["camp"])
            self.campplus_model.eval()
            self.campplus_model.to(device)

            if vocoder_type == "bigvgan":
                bigvgan_model = BigVGAN(BIGVGAN_CONFIG)
                bigvgan_model.load_state_dict(model_v2_ckpt["vocoder"])
                # remove weight norm in the model and set to eval mode
                bigvgan_model.remove_weight_norm()
                bigvgan_model = bigvgan_model.eval().to(device)
                self.vocoder_fn = bigvgan_model
            else:
                raise NotImplementedError()

        # Seed VC Content Encoder
        if whisper_model_path is not None:
            self.whisper_model = WhisperModel.from_pretrained(
                whisper_model_path, torch_dtype=torch.float16
            ).to(device)
            del self.whisper_model.decoder

            self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(
                whisper_model_path
            )

        # melspec function
        self.mel_fn = lambda x: mel_spectrogram(x, **MEL_FN_ARGS)
        self.remixer = Remixer(SAMPLERATE)
        print(f"Load checkpoints successful.")

    def semantic_fn(self, waves_16k):
        ori_inputs = self.whisper_feature_extractor(
            [waves_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        ori_input_features = self.whisper_model._mask_input_features(
            ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
        ).to(self.device)
        with torch.no_grad():
            ori_outputs = self.whisper_model.encoder(
                ori_input_features.to(self.whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        S_ori = ori_outputs.last_hidden_state.to(torch.float32)
        S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
        return S_ori

    def _ds_forward(self, ds_file, out_path, key: int = 0, save_audio: bool = True):
        with open(ds_file, "r", encoding="utf-8") as f:
            params = json.load(f)

        if not isinstance(params, list):
            params = [params]

        if len(params) == 0:
            print("The input file is empty.")
            exit()

        params = trans_key(params, key)

        return self.diffsinger.run_inference(
            params, out_dir=out_path, save_audio=save_audio
        )

    def _sv_forward(
        self,
        ds_out,
        ref_wav_path,
        out_path,
        diffusion_steps: int = 25,
        length_adjust: float = 1.0,
        cfg_rate: float = 0.7,
        auto_adjust_f0: bool = False,
        pitch_shift: int = 0,
    ):
        ref_audio = librosa.load(ref_wav_path, sr=SAMPLERATE)[0]
        ref_audio = (
            torch.tensor(ref_audio[: SAMPLERATE * 25])
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        converted_waves_16k = torchaudio.functional.resample(ds_out, SAMPLERATE, 16000)
        if converted_waves_16k.size(-1) <= 16000 * 30:
            S_alt = self.semantic_fn(converted_waves_16k)
        else:
            overlapping_time = 5  # 5 seconds
            S_alt_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < converted_waves_16k.size(-1):
                if buffer is None:  # first chunk
                    chunk = converted_waves_16k[
                        :, traversed_time : traversed_time + 16000 * 30
                    ]
                else:
                    chunk = torch.cat(
                        [
                            buffer,
                            converted_waves_16k[
                                :,
                                traversed_time : traversed_time
                                + 16000 * (30 - overlapping_time),
                            ],
                        ],
                        dim=-1,
                    )
                S_alt = self.semantic_fn(chunk)
                if traversed_time == 0:
                    S_alt_list.append(S_alt)
                else:
                    S_alt_list.append(S_alt[:, 50 * overlapping_time :])
                buffer = chunk[:, -16000 * overlapping_time :]
                traversed_time += (
                    30 * 16000
                    if traversed_time == 0
                    else chunk.size(-1) - 16000 * overlapping_time
                )

            S_alt = torch.cat(S_alt_list, dim=1)

        ori_waves_16k = torchaudio.functional.resample(ref_audio, SAMPLERATE, 16000)
        S_ori = self.semantic_fn(ori_waves_16k)

        mel = self.mel_fn(ds_out.to(self.device).float())
        mel2 = self.mel_fn(ref_audio.to(self.device).float())

        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(
            mel.device
        )
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))

        F0_ori = self.f0_fn(ori_waves_16k[0], thred=0.03)
        F0_alt = self.f0_fn(converted_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(self.device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(self.device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)

        # shift alt log f0 level to ori log f0 level
        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_adjust_f0:
            shifted_log_f0_alt[F0_alt > 1] = (
                log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            )
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(
                shifted_f0_alt[F0_alt > 1], pitch_shift
            )

        # Length regulation
        cond, _, codes, commitment_loss, codebook_loss = self.seed_vc.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
        )
        (
            prompt_condition,
            _,
            codes,
            commitment_loss,
            codebook_loss,
        ) = self.seed_vc.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
        )
        max_source_window = MAX_CONTEXT_WINDOW - mel2.size(2)

        # split source condition (cond) into chunks
        processed_frames = 0
        generated_wave_chunks = []

        # generate chunk by chunk and stream the output
        while processed_frames < cond.size(1):
            chunk_cond = cond[
                :, processed_frames : processed_frames + max_source_window
            ]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16 if self.fp16 else torch.float32,
            ):
                # Voice Conversion
                vc_target = self.seed_vc.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                    mel2,
                    style2,
                    None,
                    diffusion_steps,
                    inference_cfg_rate=cfg_rate,
                )
                vc_target = vc_target[:, :, mel2.size(-1) :]

            vc_wave = self.vocoder_fn(vc_target.float()).squeeze()
            vc_wave = vc_wave[None, :]
            if processed_frames == 0:
                if is_last_chunk:
                    output_wave = vc_wave[0].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    break
                output_wave = vc_wave[0, :-OVERLAP_WAVE_LEN].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -OVERLAP_WAVE_LEN:]
                processed_frames += vc_target.size(2) - OVERLAP_FRAME_LEN
            elif is_last_chunk:
                output_wave = crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0].cpu().numpy(),
                    OVERLAP_WAVE_LEN,
                )
                generated_wave_chunks.append(output_wave)
                processed_frames += vc_target.size(2) - OVERLAP_FRAME_LEN

                break
            else:
                output_wave = crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0, :-OVERLAP_WAVE_LEN].cpu().numpy(),
                    OVERLAP_WAVE_LEN,
                )
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -OVERLAP_WAVE_LEN:]
                processed_frames += vc_target.size(2) - OVERLAP_FRAME_LEN
        vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
        ref_wav_path = Path(ref_wav_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        torchaudio.save(
            out_path, vc_wave.cpu(), SAMPLERATE,
        )
        return vc_wave

    def _load_dit_checkpoint(
        self, model, state, ignore_modules=[], is_distributed=False, load_ema=False,
    ):
        params = state["net"]
        if load_ema and "ema" in state:
            for key in model:
                i = 0
                for param_name in params[key]:
                    if "input_pos" in param_name:
                        continue
                    assert (
                        params[key][param_name].shape == state["ema"][key][0][i].shape
                    )
                    params[key][param_name] = state["ema"][key][0][i].clone()
                    i += 1
        for key in model:
            if key in params and key not in ignore_modules:
                if not is_distributed:
                    # strip prefix of DDP (module.), create a new OrderedDict that does not contain the prefix
                    for k in list(params[key].keys()):
                        if k.startswith("module."):
                            params[key][k[len("module.") :]] = params[key][k]
                            del params[key][k]
                model_state_dict = model[key].state_dict()
                # 过滤出形状匹配的键值对
                filtered_state_dict = {
                    k: v
                    for k, v in params[key].items()
                    if k in model_state_dict and v.shape == model_state_dict[k].shape
                }
                # skipped_keys = set(params[key].keys()) - set(filtered_state_dict.keys())
                model[key].load_state_dict(filtered_state_dict, strict=False)
        _ = [model[key].eval() for key in model]

        return model

    def __call__(
        self,
        ds_file_path,
        out_path,
        ref_wav_path: Optional[str] = None,
        *,
        diffusion_steps: int = 25,
        length_adjust: float = 1.0,
        cfg_rate: float = 0.7,
        auto_adjust_f0: bool = False,
        pitch_shift_svs: int = 0,
        pitch_shift_svc: int = 0,
        mode: Literal["svs", "svc", "svs_svc"] = "svs_svc",
    ):
        """ 
        Args:
            ds_file_path: input DS_FILE path.
            out_path: output audio path.
            ref_wav_path: reference audio path.
            diffusion_steps: SVC denoising diffusion steps.
            length_adjust: stretch audio length.
            cfg_rate: Classifier-Free Guidance rate.
            auto_adjust_f0: Whether the pitch needs to be adjusted.
            pitch_shift: 
            mode: default to "svs_svc", svs + svc.
        """
        return self._forward(
            ds_file_path,
            out_path,
            ref_wav_path,
            diffusion_steps,
            length_adjust,
            cfg_rate,
            auto_adjust_f0,
            pitch_shift_svs,
            pitch_shift_svc,
            mode,
        )

    def _forward(
        self,
        ds_file_path,
        out_path,
        ref_wav_path: Optional[str] = None,
        diffusion_steps: int = 25,
        length_adjust: float = 1.0,
        cfg_rate: float = 0.7,
        auto_adjust_f0: bool = False,
        pitch_shift_svs: int = 0,
        pitch_shift_svc: int = 0,
        mode: Literal["svs", "svc", "svs_svc"] = "svs_svc",
    ):
        if mode == "svs_svc":
            assert ds_file_path is not None
            assert ref_wav_path is not None
            ds_wav = self._ds_forward(
                ds_file_path, out_path, pitch_shift_svs, save_audio=False
            )
            ds_wav = torch.tensor(ds_wav).unsqueeze(0).float().to(self.device)

            return self._sv_forward(
                ds_wav,
                ref_wav_path,
                out_path,
                diffusion_steps,
                length_adjust,
                cfg_rate,
                auto_adjust_f0,
                pitch_shift_svc,
            )
        elif mode == "svs":
            assert Path(ds_file_path).suffix == ".ds"
            return self._ds_forward(
                ds_file_path, out_path, pitch_shift_svs, save_audio=True
            )

        elif mode == "svc":
            assert ref_wav_path is not None
            assert librosa.load(ds_file_path), "You must be input an audio path."
            ds_audio = librosa.load(ds_file_path, sr=SAMPLERATE)[0]
            ds_audio = torch.tensor(ds_audio).unsqueeze(0).float().to(self.device)
            return self._sv_forward(
                ds_audio,
                ref_wav_path,
                out_path,
                diffusion_steps,
                length_adjust,
                cfg_rate,
                auto_adjust_f0,
                pitch_shift_svc,
            )

    def combine(
        self,
        gen_vocal,
        company,
        out_path,
        ori_vocal: Optional[str] = None,
        *,
        vocal_company_volume_rate: float = 0.5,
        time_stamps_list: Optional[list] = None,
    ):
        """
        Mix generated vocal with accompaniment, with optional original vocal blending.
        
        Args:
            gen_vocal (np.ndarray/str): Generated vocal audio (array or filepath)
            company (np.ndarray/str): Accompaniment audio (array or filepath)
            out_path (str): Output file path
            ori_vocal (str, optional): Original vocal audio path for blending
            vocal_company_volume_rate (float): Volume balance (0.0-1.0)
                                         0.0 = only accompaniment
                                         0.5 = equal balance (default)
                                         1.0 = only vocal
            time_stamps_list (list): List of (start,end) seconds tuples where
                                    original vocal should replace generated vocal
        
        Example:
            >>> # Basic mix at default volume balance
            >>> model.combine('gen_vocal.wav', 'accompaniment.wav', 'output.wav')
            
            >>> # Custom volume balance with original vocal blending
            >>> model.combine('gen.wav', 'accomp.wav', 'out.wav',
            ...               ori_vocal='original.wav',
            ...               vocal_company_volume_rate=0.7,
            ...               time_stamps_list=[(10,15), (30,35)])
        """
        # Standardize and balance volumes
        gen_vocal = standardization(gen_vocal, SAMPLERATE)
        gen_vocal['waveform'] = gen_vocal['waveform'] * vocal_company_volume_rate * 2
        
        company = standardization(company, SAMPLERATE)
        company['waveform'] = company['waveform'] * (1 - vocal_company_volume_rate) * 2

        # Blend with original vocal at specified timestamps
        if time_stamps_list is not None and ori_vocal is not None:
            ori_vocal = standardization(ori_vocal, SAMPLERATE)
            ori_vocal['waveform'] = company['waveform'] * vocal_company_volume_rate * 2
            
            for start, end in time_stamps_list:
                start_sample = int(start * SAMPLERATE)
                end_sample = int(end * SAMPLERATE)
                gen_vocal['waveform'][start_sample:end_sample, ] = ori_vocal['waveform'][start_sample:end_sample, ]

        # Mix and save final output
        mixed = self.remixer.mix(gen_vocal, company)["waveform"]
        sf.write(out_path, mixed, SAMPLERATE)


__all__ = ["SingingVoiceSynthesis", "ReplaceLyrics", "Remixer", "RMVPE"]
