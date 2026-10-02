# StitchSketch

StitchSketch 是一个面向钩织创作者的图纸预览工具。它可以把简洁的钩织文本转换成可视化结构图，帮助用户在正式开织前检查圈数、针目变化、配色和整体轮廓。

目前项目主要支持圆形起针作品的预览，也提供中文白话图解 / 图片文字识别到 Pattern code 的辅助转换能力。

## 功能特点

- 使用 Pattern code 输入每一圈针法，并实时生成图纸预览
- 支持中文白话图解和图片 OCR 辅助转换
- 支持每圈配色、背景色、画布尺寸和旋转角度调整
- 支持保存、打开和删除本地图纸
- 支持导出结构化 JSON
- 支持下载生图参考压缩包，供在线生图模型生成更接近真实毛线质感的参考图

## 安装步骤

建议使用 Python 3.10 或更高版本。

```bash
git clone https://github.com/your-name/StitchSketch.git
cd StitchSketch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你使用 Windows，请将激活虚拟环境的命令替换为：

```bash
.venv\Scripts\activate
```

## 启动方法

```bash
streamlit run app.py
```

启动后，浏览器会打开 StitchSketch 的交互界面。你可以在左侧输入图纸，在右侧查看实时预览。

## 使用生图参考包

完成图纸输入和配色后，在预览区域展开「生图参考包」，选择结构图导出宽度，点击「下载生图参考包」，即可获得 ZIP 压缩包。生成参考包无需安装或运行 ComfyUI。

压缩包包含以下文件：

- `annotated_stitch_map.png`：标注针目符号的结构图，用于约束作品轮廓、圈层、孔洞和针目位置
- `color_palette.png`：背景色和各圈毛线配色参考
- `prompt.txt`：生图提示词，说明如何将图纸转换为真实毛线质感
- `README_FIRST.txt`：与提示词相同的参考说明

用户可以将生图压缩包上传到支持读取压缩包和参考图片的在线生图模型，要求模型根据包内结构图、配色图和提示词，生成更接近真实毛线质感的参考图。如果所用平台不支持 ZIP 上传，请先解压，再上传两张 PNG 图片，并将 `prompt.txt` 的内容作为提示词提交。

提示词会要求保留原图纸的轮廓、孔洞、圈层和配色，将图纸符号与辅助线替换成毛线纤维和立体针目。生成结果用于观察材质与配色效果，针目数量和结构仍应以原始图纸为准。

## Pattern code 简介

每一行表示一圈，例如：

```text
6x
6v
6[x,v]
6[2x,v]
```

常用符号：

- `x` / `sc`：短针
- `t` / `hdc`：中长针
- `f` / `dc`：长针
- `e` / `tr`：长长针
- `ch`：锁针
- `sl` / `sl st`：引拔
- `v` / `inc`：加针
- `a` / `dec`：减针
- `[...] *N` 或 `[...] xN`：重复一组针法
- `(...)`：同一个针目内钩织
- `chsp(...)`：在上一圈锁针洞里钩织

## 项目结构

```text
StitchSketch/
├── app.py                  # Streamlit 用户界面
├── stitchsketch/           # 图纸解析、校验、渲染等核心逻辑
├── examples/               # 示例图纸和示例图片
├── symbol_library/         # 符号库资源
├── requirements.txt        # Python 依赖
└── README.md
```

## 说明

本项目仍在迭代中，当前更适合用于圆形钩织图纸的结构预览和设计验证。生图参考包可供用户自行上传在线生图模型使用，生成效果取决于所选模型及其参考图支持能力。
