# 🥁 Drum Practice Generator

**本地运行的 AI 电子鼓练习曲生成器。** 拖入一首歌 → AI 提取鼓轨 → 按你设定的比例降低/去除鼓声 → 可选变速、加节拍器 → 输出可直接用来练鼓的伴奏文件。

* **完全本地运行**，不上传任何歌曲，不需要云端 API，不需要购买商业软件。
* 使用 **NVIDIA GPU (CUDA)** 加速 AI 音源分离；无 GPU 时自动回退 CPU。
* 默认模型 **Demucs v4 (htdemucs)**，MIT 许可，由原作者维护。

---

## 目录

1. [系统要求](#1-系统要求)
2. [安装](#2-安装)
3. [启动](#3-启动)
4. [模型下载](#4-模型下载)
5. [GPU 配置](#5-gpu-配置)
6. [使用方法](#6-使用方法)
7. [工作原理](#7-工作原理)
8. [常见错误](#8-常见错误)
9. [性能优化](#9-性能优化)
10. [已知限制](#10-已知限制)
11. [项目结构](#11-项目结构)
12. [License](#12-license)

---

## 1. 系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 11 / 10 64-bit |
| Python | **3.11**（3.12 也可，安装脚本会自动处理） |
| GPU | NVIDIA，支持 CUDA。**RTX 50 系列（Blackwell / sm_120）需要 CUDA 12.8+ 版 PyTorch** |
| 显存 | 建议 6 GB 以上（16 GB 可跑最重的模型） |
| 磁盘 | 安装约 **6–8 GB**，处理时另需几 GB 临时空间 |
| FFmpeg | **不需要手动安装**，`install.bat` 会自动获取 |

> 本机实测环境：Windows 11 + RTX 5070 Ti (16 GB) + 驱动 CUDA 13.4 + Python 3.11 + PyTorch 2.9.1+cu130。

### FFmpeg 是怎么来的

FFmpeg 是解码、变速、响度归一化和导出的必需组件，安装脚本按两条路获取：

1. **直接下载**（首选）：先试 `gyan.dev` 的 essentials 构建，失败再试 GitHub 上的 BtbN 构建。
2. **pip 兜底**：如果两条直连都不通，脚本会 `pip install imageio-ffmpeg`。
   这个 wheel 内含一份完整的静态 FFmpeg 7.1（约 84 MB），走 PyPI（有镜像与断点续传），
   通常比单一下载源可靠得多。

程序会自动识别以上任何一种来源（也会识别系统 PATH 上已有的 FFmpeg），
运行 `diagnose.bat` 可以看到实际用的是哪一个（`来源: bundled / pip / PATH`）。

> ⚠️ 走 pip 兜底时**没有 `ffprobe`**（该 wheel 只带 `ffmpeg.exe`）。
> 这不会影响任何功能：程序会自动改为解析 ffmpeg 自身的输出读取时长/采样率/声道数。
> 只有在你需要 ffprobe 时才值得单独安装完整 FFmpeg。

**实测提示**：某些网络环境下 `gyan.dev` 可能极慢（实测仅 0.02 MB/s，110 MB 需要约 90 分钟）。
如果你遇到这种情况，直接跳到 pip 兜底会快得多 —— 脚本会自动这么做，你也可以先手动执行：

```bat
.venv\Scripts\python.exe -m pip install imageio-ffmpeg
```


---

## 2. 安装

双击 **`install.bat`**，它会依次完成：

1. 查找 Python 3.11（`py` 启动器 → 常见安装路径 → PATH）；找不到就用 `winget` 安装
2. 创建虚拟环境 `.venv`
3. 安装 **CUDA 版 PyTorch**（约 2–3 GB，从 `download.pytorch.org/whl/cu130`）
4. 安装其余依赖（Demucs 4.1.0、FastAPI、uvicorn、numpy 等）
5. 下载 FFmpeg（约 110 MB）到 `tools\ffmpeg\bin`
6. 打印验证结果与 GPU 状态

安装过程需要联网。**若其中任何一步失败，脚本会明确告诉你失败原因和下一步操作**，不会静默跳过。

### 手动安装（不想用脚本时）

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

然后手动把 FFmpeg 的 `bin` 目录放到 `tools\ffmpeg\bin`。

### 关于 PyTorch 下载（很重要）

PyTorch 的 CUDA 版约 **2.9 GB**，而 **下载源的速度差别极大**，直接决定安装要几分钟还是几小时。实测数据（2026-10-07）：

| 下载源 | 实测速度 | 2.9 GB 预计耗时 |
|---|---|---|
| `download.pytorch.org`（官方） | 1.87 – 3.41 MB/s | 15 – 25 分钟 |
| **`mirror.sjtu.edu.cn`（上海交大镜像）** | **4.95 – 17.85 MB/s** | **3 – 10 分钟** |
| 某些网络到官方源 | 0.02 – 0.22 MB/s | **4 小时以上** |

所以安装程序会**先实测两个源的速度，再自动选快的那个**：

```text
  正在测速：官方源 vs 国内镜像（各约 6 秒）...
    官方源  :   2.29 MB/s
    国内镜像:   5.25 MB/s
  下载源：国内镜像（实测 5.3 MB/s）
```

命令行可以手动控制：

```bat
install.bat                                    :: 默认 cu128 + 自动测速
install.bat --variant cu130                    :: 改用 CUDA 13.0 版（无国内镜像）
install.bat --source official                  :: 强制官方源
install.bat --source mirror                    :: 强制国内镜像
install.bat --pypi-mirror official             :: 其他依赖也用官方 PyPI
```

> **为什么默认是 cu128 而不是 cu130？**
> 两者都支持 Blackwell（sm_120）。cu128 的 CUDA 12.8 运行时能在 13.x 驱动上正常工作
> （驱动向下兼容，且 PyTorch 自带 CUDA 运行时库）。
> 而**只有 cu128 有国内镜像**，在 2.9 GB 这个体量上镜像快好几倍 —— 所以选它。
> 你的驱动是 CUDA 13.4，cu128 与 cu130 都能跑。

### 如果安装中途断了

**直接重新运行 `install.bat` 即可**，pip 会断点续传已经下载的部分，不会从头再来。

### 安装时出现 Python 报错（Traceback）

`install.bat` 自己只做引导，真正的安装在 `app\setup.py`。
若看到 `Traceback ... NameError` 之类，说明安装脚本本身有 bug ——
请把窗口内容或 `logs\install.log` 发出来。
所有退出路径都有 `pause`，窗口不会自动关闭。

### 下载源速度

安装程序会先测速再选源，并自动重试 10 次。
实测镜像可达 **32.7 MB/s**（2.9 GB 约 1.5 分钟），而某些网络到官方源只有 0.02 MB/s。
若自动选择不理想，可手动指定：

```bat
install.bat --source mirror      :: 强制国内镜像
install.bat --source official    :: 强制官方源
install.bat --variant cu130      :: 换 CUDA 13.0 版
```

### 代理

需要代理时先设置环境变量再运行：

```bat
set HTTPS_PROXY=http://127.0.0.1:端口
install.bat --pypi-mirror official --source official
```

### 安装日志

`install.bat` 只做引导（找到 Python），真正的工作在 `app\setup.py` 里，
全部输出同时写入 **`logs\install.log`**（UTF-8 编码）。

> 用记事本或 VS Code 打开该文件时请选择 **UTF-8** 编码。
> 如果用 GBK 打开会显示乱码 —— 这是查看方式的问题，文件本身是正确的 UTF-8。

### 如果安装窗口一闪就关闭

正常情况不会：`install.bat` 的每一条退出路径都有 `pause`。
若确实出现，说明是**在你机器上直接双击运行 `.bat` 本身出了问题**，请这样确认：

```bat
:: 打开 cmd，切到项目目录，然后运行 —— 窗口不会自动关闭
cd /d <项目目录>
install.bat
```

最可能的原因是文件的**换行符被改成了 LF**（某些编辑器或 Git 配置会这样做）。
Windows 的 `cmd.exe` 无法正确解析 LF 换行的批处理文件，会把 `cd /d` 之类的命令
拆成乱码执行。修复办法：

```bat
:: 用 PowerShell 把换行改回 CRLF
(Get-Content install.bat -Raw) -replace "`n", "`r`n" | Set-Content install.bat -NoNewline
```

本项目的 `.bat` 文件一律是 **纯 ASCII + CRLF**，并且不含中文
（中文提示都放在 Python 里输出），就是为了避免这类编码问题。

---

## 3. 启动

双击 **`start.bat`**。它会：

* 检查虚拟环境与依赖是否完整
* 把自带的 FFmpeg 加入 PATH
* 打印 GPU 状态（并记录到日志）
* 启动本地服务并自动打开浏览器

浏览器地址：**http://127.0.0.1:8765/**

> 服务**只监听 127.0.0.1**，局域网内其他设备无法访问 —— 这是本地工具，歌曲不会离开你的电脑。

### 命令行参数

```bat
start.bat --port 8766        :: 换端口
start.bat --cpu              :: 强制 CPU 模式（调试用，很慢）
start.bat --no-browser       :: 不自动开浏览器
start.bat --diagnose         :: 只打印环境诊断后退出
```

---

## 4. 模型下载

程序启动**不会**自动下载模型。首次使用时：

1. 打开网页界面 → 展开「高级设置」
2. 选择模型 → 点击 **下载模型**

模型来自 **HuggingFace 官方仓库 `adefossez/HTDemucs`**（Demucs 作者发布），下载后存放在项目内的：

```text
models\huggingface\hub\models--adefossez--HTDemucs\
```

| 模型 | 说明 | 体积 | 相对耗时 |
|---|---|---|---|
| `htdemucs` | **默认**。速度/质量平衡最好 | ~80 MB | 1× |
| `htdemucs_ft` | 4 模型集成，质量略高 | ~320 MB | ~4× |
| `hdemucs_mmi` | 旧版 Hybrid Demucs，备用对照 | ~320 MB | ~1.5× |

详细的来源、许可证与手动下载方法见 [`models/README.md`](models/README.md)。

**不要从来源不明的网盘或镜像下载模型权重。**

---

## 5. GPU 配置

### 自动检测

程序启动时会在控制台和 `logs\drum-practice-generator.log` 记录：

```text
OS            : Windows ...
Python        : 3.11.x (64-bit) venv
PyTorch       : 2.9.1+cu130
CUDA (torch)  : 13.0
CUDA available: True
GPU           : NVIDIA GeForce RTX 5070 Ti
Compute       : sm_120  (Blackwell)
VRAM          : 16303 MB total, 15200 MB free
FFmpeg        : 7.x (bundled)
Demucs        : 4.1.0
```

### GPU 不可用时

程序**不会崩溃**，会打印并回退到 CPU：

```text
CUDA GPU unavailable. The program will fall back to CPU mode.
```

CPU 模式下处理一首 4 分钟的歌大约需要几分钟到十几分钟。

### 排查

运行 **`diagnose.bat`**（或 `.venv\Scripts\python.exe diagnose.py`）会逐项检查并给出下一步操作。

最常见的问题：**装成了 CPU 版 PyTorch**。
判断方法：看版本号有没有 `+cuXXX` 后缀。

```text
PyTorch       : 2.9.1        <-- 没有 +cu130 → CPU 版，需要重装
PyTorch       : 2.9.1+cu130  <-- 正确
```

重新安装：删掉 `.venv` 后重跑 `install.bat`。

### 关于 RTX 50 系列（Blackwell）

Blackwell 是计算能力 **sm_120**，**必须使用 CUDA 12.8 或更新的 PyTorch 构建**。
本项目默认安装 `torch 2.9.1+cu130`。如果你的驱动较旧，可改用 `cu128`：
编辑 `install.bat`，把 `TORCH_INDEX` 里的 `cu130` 改成 `cu128` 即可。

---

## 6. 使用方法

### 基本流程

1. **导入歌曲** —— 拖入界面，或点击选择文件。支持 MP3 / WAV / FLAC / M4A / OGG / AAC。
   界面会显示文件名、时长、格式、采样率、声道数、文件大小，并自动检测 BPM。
2. **设置鼓声强度** —— 滑块 0%～100%，默认 **10%**。
   也可以用快捷按钮：`完全去鼓(0%)` / `弱鼓练习(10%)` / `原曲(100%)`。
3. **设置速度** —— 填 Target BPM，或直接拖 Playback Speed 滑块。
   **变速不变调。** 两者联动：改一个另一个自动算。
4. **节拍器**（可选）—— 勾选启用，设置音量、拍号（4/4、3/4、6/8）。
5. **输出格式** —— WAV（推荐）/ MP3 / FLAC。
6. 点击 **生成练习曲** → 进度条实时显示当前阶段 → 完成后可在线试听。

### 鼓声强度的含义

```text
0%   = 完全去除鼓
10%  = 极弱鼓（推荐练习起点）
20%  = 轻微保留
50%  = 半鼓
100% = 原始鼓声
```

混音公式：

```text
输出 = (人声 + 贝斯 + 其他乐器) + 鼓 × 鼓声比例
```

> 伴奏轨是用「原曲 − 鼓轨估计」得到的残差，而不是把人声/贝斯/其他三轨相加。
> 这样相位与原曲一致，听起来像「把鼓手静音的原曲」，而不是四轨重建的拼贴。

### 输出文件

输出到 `output\`，文件名自动包含关键参数：

```text
output\
 ├── My_Song_Drums0.wav                 # 完全去鼓
 ├── My_Song_Drums10.wav                # 弱鼓
 ├── My_Song_Drums10_BPM90.wav          # 弱鼓 + 变速到 90 BPM
 └── My_Song_Drums20_BPM100_Click.wav   # 带节拍器
```

勾选 **保留分离出的分轨** 时，还会在 `output\<曲名>_stems\` 下保留 `drums.wav`、`vocals.wav`、`bass.wav`、`other.wav`，方便你进一步加工。

### 日志

`logs\drum-practice-generator.log` 记录每次任务：

```text
时间        : 2026-10-07 12:30:04
歌曲        : test.mp3
时长        : 00:03:42
模型        : htdemucs
设备/GPU    : NVIDIA GeForce RTX 5070 Ti
原始 BPM    : 120
目标 BPM    : 90
鼓声音量    : 10%
速度        : 75%
节拍器      : 4/4 @ 90.0 BPM vol 0.30
分离耗时    : 34.2 s
总处理时间  : 41.8 s
输出文件    : test_Drums10_BPM90.wav
状态        : SUCCESS
```

同时会写一份机器可读的 `logs\history.jsonl`。

---

## 7. 工作原理

```text
读取歌曲 (FFmpeg 解码为 44.1kHz 立体声 float32)
   ↓
分析：BPM 检测、元数据
   ↓
AI 分离：Demucs v4 htdemucs → drums / bass / other / vocals
   ↓
构造伴奏轨 = 原曲 − 鼓轨
   ↓
按比例调整鼓声：输出 = 伴奏 + 鼓 × 比例
   ↓
变速（atempo / Rubber Band，音高不变）
   ↓
混入节拍器（可选）
   ↓
峰值归一化 + 前瞻限幅 + EBU R128 响度归一化
   ↓
导出 WAV / MP3 / FLAC
```

### 为什么选 Demucs v4 htdemucs

* **鼓分离质量**：htdemucs 在 MUSDB HQ 上整体 SDR 9.0 dB，是当前开源方案里质量与易用性平衡最好的。
* **GPU 支持**：纯 PyTorch，CUDA 开箱可用，无自定义 CUDA 算子（不需要编译）。
* **Windows 安装难度**：`pip install demucs` 即可。4.1.0 起音频读写改用 `sphn`，摆脱了旧版 torchaudio 在 Windows 上的 FFmpeg DLL 麻烦。
* **模型大小 / 显存**：htdemucs 仅 ~80 MB，7 秒 segment 在 16 GB 卡上峰值约 6 GB。
* **License**：MIT，个人与商业使用均无限制。

**没有选用**：MDX-Net（2021 比赛提交，无 pip 包、无维护）、BSRNN（仅研究仓库与 Zenodo 检查点，Windows 无可用推理管线）、`htdemucs_6s`（作者明确说明 piano 轨质量差、串音严重）。

架构上分离后端是可插拔的（`app/separation/base.py` 定义了协议），将来要接 MDX/BSRNN 只需注册一个新类。

---

## 8. 常见错误

所有错误都会在界面上给出**具体原因和可操作建议**，而不是只说一句 "Error"。

| 提示 | 原因 | 解决 |
|---|---|---|
| `FFmpeg 未找到` | 未安装或路径不对 | 运行 `install.bat`，或手动把 ffmpeg 放到 `tools\ffmpeg\bin` |
| `CUDA GPU 不可用` | 装的是 CPU 版 PyTorch，或驱动问题 | 运行 `diagnose.bat`；若版本号无 `+cuXXX` 则重装 PyTorch |
| `PyTorch 未安装` | 安装未完成 | 运行 `install.bat` |
| `模型尚未下载` | 首次使用 | 界面里点「下载模型」 |
| `模型下载失败` | 网络/代理问题 | 重试；或用 `models\README.md` 里的手动下载方法 |
| `GPU 显存不足` | segment 太大或其他程序占用显存 | 调小 Segment（如 7 → 5）、改用 htdemucs、关掉其他占显存的程序 |
| `不支持的文件格式` | 扩展名不在支持列表 | 支持 MP3/WAV/FLAC/M4A/OGG/AAC；先用 FFmpeg 转成 WAV |
| `无法解码音频文件` | 文件损坏或受 DRM 保护 | 先用播放器确认能播放；DRM 文件需先转码 |
| `磁盘空间不足` | 剩余空间不够 | 清理磁盘，或设环境变量 `DPG_OUTPUT_DIR` 指到大盘 |
| `输出目录不可写` | 权限/杀毒软件拦截 | 不要把项目放在 `C:\Program Files` 下 |
| `BPM 检测失败` | 音频太短/太安静/无稳定节拍 | **不阻塞流程**，手动填 Original BPM 即可 |

### 自定义目录

```bat
set DPG_OUTPUT_DIR=D:\DrumPractice\output
set DPG_TEMP_DIR=D:\DrumPractice\temp
start.bat
```

### 临时文件

处理中的中间文件放在 `temp\`（`uploads\`、`separation\`、`mix\`），**任务结束后自动清理**。
异常中断时可能残留，可安全手动删除。

---

## 9. 性能优化

> **本机实测数据（RTX 5070 Ti / htdemucs / segment=7 / overlap=0.25）**
>
> | 项目 | 实测 |
> |---|---|
> | 模型加载（本地缓存） | **1.8 秒**（首次联网下载后） |
> | AI 分离（8 秒音频） | **15.7 秒** |
> | 完整流程（8 秒音频，含变速+节拍器+导出） | **16.9 秒** |
> | GPU 显存峰值 | **553 MB** |
> | 输出响度 | −14.0 LUFS（EBU R128） |
>
> 按此推算，4 分钟的歌分离约需 8 分钟。显存占用很低，16 GB 绰绰有余。

* **默认设置已经针对显存优化**：`segment=7`、`overlap=0.25`、`shifts=1`，实测峰值仅 553 MB。
* **模型加载不会重复联网**：首次下载后会缓存到 `models\cache\htdemucs.th`，
  之后每次加载约 2 秒、完全离线。
  （这一点很重要：按名称加载会让 Demucs 去连 HuggingFace 并回退到 AWS，
  在慢速网络上实测每次要 **317 秒**，还有可能重新下载 80 MB。）
* **想要更快**：
  * 用 `htdemucs`（默认），不要用 `htdemucs_ft`（慢 4 倍）
  * `shifts` 保持 1
  * 把 `overlap` 降到 0.1
* **显存不够时**：把 `segment` 降到 5 甚至 4。htdemucs 最大只支持 7.8 秒。
* **变速后端**：默认 FFmpeg `atempo`（已内置）。
  把 `rubberband.exe` 放到 `tools\rubberband` 会自动升级到 Rubber Band —— 极端变速比例下瞬态更好。注意 Rubber Band 是 **GPL** 许可。
* 程序已做的显存/内存保护：一次只跑一个重任务、分离后主动释放 CUDA 缓存、分离/混音分块处理、临时文件自动清理。

### 加速下载（国内网络）

如果 PyTorch 或模型下载很慢，安装程序会**自动测速并选择更快的源**（见上文），
另外还可以手动指定：

```bat
:: 安装时强制走国内镜像（实测可达 32.7 MB/s）
install.bat --source mirror

:: 模型下载也很慢时，可让 HuggingFace 走镜像站
set HF_ENDPOINT=https://hf-mirror.com
start.bat
```

---

## 10. 已知限制

请诚实看待以下限制：

### BPM 自动检测是尽力而为

检测基于 onset 包络的自相关 + 音乐速度先验，在合成的真实鼓型上实测：

* 60–200 BPM 全范围 **没有出现错误的速度**，其中约一半精确命中，
  另一半会报告**干净的半速或二倍速**。
* **为什么**：只靠周期性无法区分一个速度和它的八度。
  一个每两拍重复一次的鼓型，在「半速」上**确实是**周期的；
  要分开二者需要节拍位置跟踪，而不是周期性分析。
* **因此**：界面上 BPM 永远可以手动改。检测不可靠时不会阻塞流程。
  对鼓练习来说，半速节拍器依然可用（每拍一下 → 每两拍一下）。

### 节拍器对齐

节拍器按**目标速度**生成，重音落在每小节第 1 拍（6/8 为第 1、4 拍）。
它**不保证与歌曲的小节线对齐** —— 那需要准确的 downbeat 检测。
如果感觉偏移，可用「高级设置 → 节拍器偏移」微调。

### 其他

* 不支持实时处理：必须先生成文件。
* 分离质量取决于原曲：鼓声经过重压缩、或被其他乐器掩蔽时，残留会明显一些。
* 不支持 DRM 保护的 M4A。

---

## 11. 项目结构

```text
drum-practice-generator/
├── app/
│   ├── main.py              入口（uvicorn 启动、环境横幅）
│   ├── config.py            全部路径与默认值
│   ├── env.py               Python/PyTorch/CUDA/GPU 探测与设备选择
│   ├── errors.py            结构化错误（code + message + suggestions）
│   ├── tools.py             子进程封装（无控制台闪烁、编码兜底）
│   ├── logging_setup.py     日志、启动横幅、任务记录
│   ├── tasks.py             后台任务管理（进度、取消、串行化）
│   ├── pipeline.py          核心编排：完整处理链路
│   ├── metronome.py         节拍器合成
│   ├── diagnose.py          环境诊断
│   ├── api/
│   │   ├── app.py           FastAPI 应用 + 静态前端
│   │   └── routes.py        HTTP 接口
│   ├── audio/
│   │   ├── io.py            float32 WAV 读写（含 24-bit）
│   │   ├── ffmpeg.py        FFmpeg 探测、解码、导出、atempo
│   │   ├── mix.py           鼓声混音、峰值归一化、前瞻限幅器
│   │   ├── loudness.py      EBU R128 响度测量
│   │   └── timestretch.py   变速后端（atempo / Rubber Band）
│   ├── separation/
│   │   ├── base.py          后端协议 + 结果数据结构
│   │   ├── demucs_backend.py Demucs v4 实现
│   │   ├── models.py        模型缓存、状态检测、下载
│   │   └── registry.py      后端注册表
│   └── bpm/
│       └── detect.py        BPM 检测（自相关 + 速度先验 + 节拍跟踪）
├── frontend/                index.html / style.css / app.js
├── models/                  模型缓存（HF 结构）
├── temp/                    临时文件（自动清理）
├── output/                  输出练习曲
├── logs/                    日志 + history.jsonl
├── tools/                   内置 FFmpeg（及可选 Rubber Band）
├── install.bat              安装
├── start.bat                启动
├── diagnose.bat             环境诊断
├── diagnose.py              诊断（Python 入口）
├── selftest.py              离线自测（不需要 GPU/模型/网络）
├── requirements.txt
└── requirements-torch.txt
```

### 离线自测

不需要 GPU、模型或网络，验证 WAV 读写、限幅器、鼓声混音、节拍器、
BPM 检测、命名规则、错误处理与任务管理：

```bat
.venv\Scripts\python.exe selftest.py
.venv\Scripts\python.exe selftest.py -v      :: 显示每一步
```

---

## 12. License

本项目代码：**MIT**。

第三方组件：

| 组件 | 许可 | 说明 |
|---|---|---|
| [Demucs](https://github.com/adefossez/demucs) + 模型权重 | MIT | 由原作者 Alexandre Défossez 发布，权重托管在 HuggingFace `adefossez` 命名空间 |
| [FFmpeg](https://ffmpeg.org)（gyan.dev essentials 构建） | LGPL v2.1+ | 随项目下载，可自由分发使用 |
| [PyTorch](https://pytorch.org) | BSD-3-Clause | CUDA 构建 |
| [Rubber Band Library](https://breakfastquay.com/rubberband/) | **GPL v2+** | **可选**，仅当你手动放入 `rubberband.exe` 时启用；个人使用无限制，二次分发需遵守 GPL |
| [FastAPI](https://fastapi.tiangolo.com) / [uvicorn](https://www.uvicorn.org/) | MIT / BSD-3-Clause | Web 层 |

**本程序不下载来源不明的模型**，所有模型权重均来自上述官方仓库。
