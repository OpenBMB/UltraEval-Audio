中文版 @叶润传

## 🎙️ VoxCPM: A Tokenizer-Free TTS System for Hyper-Realistic Speech and High-Fidelity Voice Cloning
VoxCPM: Beyond Tokens: A Foundation Model for Speech Generation with True Vocal Nuance

[Project Page](github链接)

[Hugging Face](huggingface链接)

[ModelScope](modelscope链接)

[Technical Report](Coming Soon)

[Live Playground]（之后放HF、ModelScope Space链接）(https://tts.ali-dev.modelbest.cn/)（https://playground.ali.modelbest.co/audio-chat）

[图片]

### 🔥 News
- [2025-09-15] 🎉 We Open Source the VoxCPM-0.5B weights!
- [2025-09-15] 🎉 We Provide the Gradio DemoPage for VoxCPM-0.5B, try it now!

VoxCPM is a groundbreaking tokenizer-free Text-to-Speech (TTS) system that sets a new standard for realism in speech synthesis. By modeling audio in a continuous space, it avoids the quality degradation of discrete tokens, enabling its two flagship features: hyper-realistic speech generation and high-fidelity voice cloning.

### 🚀 Features
- Hyper-Realistic Speech: Powered by its unique tokenizer-free architecture, VoxCPM produces speech with unparalleled naturalness, capturing subtle prosody and a wide emotional range. Trained on a massive 1.8 million-hour bilingual dataset, it achieves state-of-the-art (SOTA) performance in speech quality.
- High-Fidelity Voice Cloning: The system excels at zero-shot voice cloning, creating a near-perfect replica of a target voice from just a few seconds of audio. It meticulously reproduces the speaker's unique timbre, accent, emotion, and even the ambient background, ensuring a clone of astonishing accuracy.
- Blazing-Fast & Efficient: Engineered for performance, VoxCPM supports streaming synthesis with a Real-Time Factor (RTF) as low as 0.16 on a consumer NVIDIA RTX 4090 GPU, making it ideal for real-time applications.

### 🏗️ Architecture

[图片]


### 📊 Performance
VoxCPM achieves state-of-the-art performance on zero-shot TTS task:

【贴一些官方客观评测指标结果】Seed-TTS-eval， CV3-eval
@王子阳 （开源、闭源、模型大小）














### 🎵 Demo Examples
For more examples, see the Project Page.【@余任杰 】
Try it on  Demo.

### Models
Model
Weight
VoxCPM-0.5B
HF link
VoxCPM-3B
Coming Soon

### Install
``` sh
pip install voxcpm
```
### Model Download (Optional)
By default, when you first run the script, the model will be downloaded automatically, but you can also download the model in advance.
- Download VoxCPM-0.5B
    ```
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id, local_files_only=local_files_only)
    ```
- Download ZipEnhancer and SenseVoice-Small. We use ZipEnhancer to enhance speech prompts and SenseVoice-Small for speech prompt ASR in the web demo.
    ```
    from modelscope import snapshot_download
    snapshot_download('iic/speech_zipenhancer_ans_multiloss_16k_base')
    snapshot_download('iic/SenseVoiceSmall')
    ```

### Usage
#### Basic Usage
```python
import soundfile as sf
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained("HF_REPO_ID")

wav = model.generate(
    text="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech.",
    prompt_wav_path=None,      # optional: path to a reference audio for voice cloning
    prompt_text=None,          # optional: reference text
    cfg_value=2.0,
    inference_timesteps=10,
    normalize=True,
    denoise=True,
    retry_badcase=True,
    retry_badcase_max_times=3,
    retry_badcase_ratio_threshold=6.0,
)

sf.write("output.wav", wav, 16000)
print("saved: output.wav")
```

#### CLI Usage

After installation, the entry point is `voxcpm` (or use `python -m voxcpm.cli`).

```bash
# 1) Direct synthesis (single text)
voxcpm --text "Hello VoxCPM" --output out.wav

# 2) Voice cloning (reference audio + transcript)
voxcpm --text "Hello" \s
  --prompt-audio path/to/voice.wav \
  --prompt-text "reference transcript" \
  --output out.wav \
  --denoise

# 3) Batch processing (one text per line)
voxcpm --input examples/input.txt --output-dir outs
# (optional) Batch + cloning
voxcpm --input examples/input.txt --output-dir outs \
  --prompt-audio path/to/voice.wav \
  --prompt-text "reference transcript" \
  --denoise

# 4) Inference parameters (quality/speed)
voxcpm --text "..." --output out.wav \
  --cfg-value 2.0 --inference-timesteps 10 --normalize

# 5) Model loading
# Prefer local path
voxcpm --text "..." --output out.wav --model-path /path/to/VoxCPM_model_dir
# Or from Hugging Face (auto download/cache)
voxcpm --text "..." --output out.wav \
  --hf-model-id openbmb/VoxCPM --cache-dir ~/.cache/huggingface --local-files-only

# 6) Denoiser control
voxcpm --text "..." --output out.wav \
  --no-denoiser --zipenhancer-path iic/speech_zipenhancer_ans_multiloss_16k_base

# 7) Help
voxcpm --help
python -m voxcpm.cli --help
```

### Start web demo

You can start the UI interface by running `export HF_REPO_ID=HF_REPO_ID && python app.py`, which allows you to perform Voice Cloning and Voice Creation.



### 👩‍🍳 The VoxCPM TTS Synthesis: A Voice Chef's Guide
Welcome to the VoxCPM kitchen! Here’s your recipe for cooking up perfect synthetic speech. Let's get started.

---
🥚 Step 1: Prepare Your Base Ingredients (The Text)
First, you need to decide what kind of text you're cooking with.
1. For Regular Text (The Classic Recipe)
- ✅ Keep "Text Normalization" ON. Just type like you normally would (e.g., "Hello, world! 123"). Our system is your sou-chef, expertly handling numbers, abbreviations, and punctuation behind the scenes.
2. For Advanced Control (The Molecular Gastronomy Technique)
- ❌ Turn "Text Normalization" OFF. This lets you input precise phonemes using the {HH AH0 L OW1} (EN) or {ni3}{hao3} (ZH) format. Perfect for when you need ultimate control over the pronunciation of every syllable.

---
🍳 Step 2: Choose Your Flavor Profile (The Prompt Audio)
This is the secret sauce that gives your audio its unique sound.
1. Cooking with a Prompt Audio (Following a Famous Recipe)
  - A prompt audio provides the desired acoustic characteristics for VoxCPM. The speaker's timbre, speaking style, and even the background sounds and ambiance will be replicated.
  - For a Clean, Studio-Quality Voice:
    - ✅ Enable "Prompt Speech Enhancement". This acts like a noise filter, removing background hiss and rumble to give you a pure, clean voice clone.
2. Cooking au Naturel (Letting the Model Improvise)
  - If no audio is provided, VoxCPM becomes a creative chef! It will infer a fitting speaking style based on the text itself, thanks to the text-smartness of its foundation model, MiniCPM-4.
  - Pro Tip: Feel free to throw any text at it! Poetry, song lyrics, a dramatic monologue—VoxCPM is ready for a challenge.

---
🍝 Step 3: Serving a Full Course (Handling Long-Form Content)
Trying to cook a giant, unbroken block of text? Let's portion it out.
- Simply split your giant text into paragraphs with empty lines between them.
- Why this works: VoxCPM will expertly generate each paragraph and then seamlessly stitch them together into one final, smooth audio file. No seams, just seconds.

---
🧂 Step 4: The Final Seasoning (Fine-Tuning Your Results)
You're ready to serve! But for master chefs who want to tweak the flavor, here are two key spices.
- CFG Scale (How Closely to Follow the Recipe)
  - Default: A great starting point.
  - Voice sounds strained or weird? Lower this value. It tells the model to be more relaxed and improvisational, great for expressive prompts.
  - Need maximum clarity and adherence to the text? Raise it slightly to keep the model on a tighter leash.
- Inference Timesteps (Simmering Time: Quality vs. Speed)
  - Need a quick snack? Use a lower number. Perfect for fast drafts and experiments.
  - Cooking a gourmet meal? Use a higher number. This lets the model "simmer" longer, refining the audio for superior detail and naturalness.

---
Happy creating! 🎉 Start with the default settings and tweak from there to suit your project. The kitchen is yours!


---






### ⚠️ Risks and limitations
- General Model Behavior: While VoxCPM has been trained on a large-scale dataset, it may still produce outputs that are unexpected, biased, or contain artifacts. The model inherits potential biases from its underlying large language model (MiniCPM) and the training data.
- Potential for Misuse of Voice Cloning: VoxCPM's powerful zero-shot voice cloning capability can generate highly realistic synthetic speech. This technology could be misused for creating convincing deepfakes for purposes of impersonation, fraud, or spreading disinformation. Users of this model must not use it to create content that infringes upon the rights of individuals. It is strictly forbidden to use VoxCPM for any illegal or unethical purposes. We strongly recommend that any publicly shared content generated with this model be clearly marked as AI-generated.
- Current Technical Limitations: Although generally stable, the model may occasionally exhibit instability, especially with very long or complex inputs. Furthermore, the current version offers limited direct control over specific speech attributes like emotion or speaking style. Improving generation stability and enhancing controllability are key research directions for our future releases.
- Bilingual Model: VoxCPM is trained primarily on Chinese and English data. Performance on other languages is not guaranteed and may result in unpredictable or low-quality audio.
This model is released for research and development purposes only. We do not recommend its use in production or commercial applications without rigorous testing and safety evaluations. Please use VoxCPM responsibly.



### 📄 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments
- DiTAR for the patch-level continuous speech generation backbone
- MiniCPM for the language model foundation
- CosyVoice for the Flow Matching-based LocDiT

### 📞 Contact
- Organization: OpenBMB, ModelBest, THU-HCSI
- Email: openbmb@gmail.com
- GitHub: https://github.com/OpenBMB/VoxCPM

### 📚 Citation

If you find this model useful, please consider citing.
@misc{chatterboxtts2025,
  author       = {{作者名单，待确认}},
  title        = {{VoxCPM}},
  year         = {2025},
  howpublished = {\url{https://github.com/OpenBMB/VoxCPM}},
  note         = {GitHub repository}
}

作者名单：

Contributions
周逸轩, 曾国洋, 刘鑫, 李翔, 余任杰, 王子阳, 叶润传, ……, 吴志勇*, 刘知远*


Acknowledgement
We would like to also thank 井晨哲, 孙炜岳, 桂建成, 李可涵, 何弈航, 李瑞, 郭竣劭, 史群东, 林弼远, 周界, ...... for their help.

---

（OpenBMB logo，面壁logo，清华吴老师实验室logo）
[./assets/thuhcsi.png]
