""" 仅保留四个接口
__all__ = ['SingingVoiceSynthesis',
            'ReplaceLyrics',
           'Remixer',
           'RMVPE']
"""
# NOTE: 如果合成效果不好，说明DS文件标注存在问题。
# 好的合成结果参考 '/gpfs/home/shangzengqiang/yangchen/SongEdit/data/out/00.wav'

from songedit.svc import *

# 修改
model = ReplaceLyrics()

model.process(
    "/gpfs/home/shangzengqiang/yangchen/SongEdit/data/new_lrc/陈鸿宇-火烧云_proofread.ds",
    "/gpfs/home/shangzengqiang/yangchen/SongEdit/data/new_lrc/陈鸿宇-火烧云_multi_ds.txt",
    "/gpfs/home/shangzengqiang/yangchen/SongEdit/data/new_lrc/out.ds"
)


model = SingingVoiceSynthesis(
        "checkpoints/step2/model_v1.pt",
        "checkpoints/step2/model_v2.pt",
        "checkpoints/step2/whisper-small/")

# SVS + SVC
model("/gpfs/home/shangzengqiang/yangchen/SongEdit/data/new_lrc/out.ds",
      "/gpfs/home/shangzengqiang/yangchen/SongEdit/data/out/陈鸿宇-火烧云_ref_dingzhen.wav",
      "/gpfs/home/shangzengqiang/yangchen/SongEdit/data/reference/ref_dingzhen.wav",
      pitch_shift_svs=12,    # only for 歌声合成  如果音域不一致的情况，需要升降调以满足对应的歌手
      pitch_shift_svc=-12    # only for 歌声变声  如果音域不一致的情况，需要升降调以满足对应的歌手
      )

model = SingingVoiceSynthesis()     # 无需load checkpoints
model.combine(
    "/gpfs/home/shangzengqiang/yangchen/SongEdit/data/out/陈鸿宇-火烧云_ref_dingzhen.wav",
    "/gpfs/home/shangzengqiang/yangchen/SongEdit/data/dataset/wavs/陈鸿宇-火烧云_伴奏.wav",
    "/gpfs/home/shangzengqiang/yangchen/SongEdit/data/out/陈鸿宇-火烧云-丁真.wav"
    )

# # SVC
# model("/gpfs/home/shangzengqiang/yangchen/SongEdit/data/out/陈鸿宇-火烧云_proofread.wav",
#         "/gpfs/home/shangzengqiang/yangchen/SongEdit/data/out/陈鸿宇-火烧云_ref_dingzhen.wav",
#       "/gpfs/home/shangzengqiang/yangchen/SongEdit/data/reference/ref_dingzhen.wav",
#       mode="svc")

# # SVS
# model("/gpfs/home/shangzengqiang/yangchen/SongEdit/data/new_lrc/out.ds",
#       "/gpfs/home/shangzengqiang/yangchen/SongEdit/data/out/陈鸿宇-火烧云_diff.wav",
#       pitch_shift_svs=12,
#       mode="svs")

