# DIY AI

This repo contains my tutorial and implementation for ML,DL,LLM,NLP and ASR, ranging from ML basics to advanced genAI, from algorithm to system, and from MLOps to AI on the edge.
This repo is primarily based on [github](https://github.com), [Pytorch](https://pytorch.org), [Huggingface🤗](https://huggingface.co), and [Colab](https://colab.research.google.com). And I appreciate great material online like Stanford's [Machine Learning](https://cs229.stanford.edu/), Mu Li's [Dive into deep learning](https://d2l.ai/) and Hungyi Lee's [genAI](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php).

## Math Prerequites

I found these great cheatsheets for beginner & refresher:

- **[Probabilities and Statistics](https://stanford.edu/~shervine/teaching/cs-229/refresher-probabilities-statistics)**
- **[Linear Algebra and Calculus](https://stanford.edu/~shervine/teaching/cs-229/refresher-algebra-calculus)**

## ML Basics

## GenAI - Generative AI

Salute to Prof. Lav Varshney for his awesome [genAI course](https://courses.grainger.illinois.edu/ECE598LV/sp2022/).

- **VAE - Variational autoencoder**
- **GAN - Generative adversarial network**
- **Autoregressive**
- **Diffusion**

    Huggingface🤗 [diffusion class](https://github.com/huggingface/diffusion-models-class)

    Huggingface🤗 [diffusers](https://github.com/huggingface/diffusers)

- **Transformer**

    Huggingface🤗 [transformers](https://github.com/huggingface/transformers)

- **PEFT - Parameter-Efficient Fine-Tuning**

    1. Huggingface🤗 [peft](https://github.com/huggingface/peft)
    2. Lora:Check out their [paper](https://arxiv.org/abs/2106.09685) and [source](https://github.com/microsoft/LoRA)

- **Prompt**
- **Detection**

## NLP - Natural Language Processing

- **GPT - Generative Pre-trained Transformer**
    The best materials are definitely [Andrej Karpathy](https://github.com/karpathy) with his [build GPT from sratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&ab_channel=AndrejKarpathy). And he does have several versions of minimal GPTs:

    1. [minGPT](https://github.com/karpathy/minGPT) for education: A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training

    2. [nanoGPT](https://github.com/karpathy/nanoGPT) with teeth: The simplest, fastest repository for training/finetuning medium-sized GPTs.

    3. [LLM101n](https://github.com/karpathy/LLM101n) his latest tutorial thats still under construction.

    I would personally start with minGPT and his youtube to get a sense of it, get hands dirty and use the latter two as reference.

- **BERT - Bidirectional Encoder Representations from Transformers**

## ASR - Automatic Speech Processing

- **Audio Preprocessing**

    Look for tag [Audio](https://pytorch.org/tutorials/index.html) in the tutorials.

    Also the torchaudio's [official tutorial](https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html).

- **CTC - Connectionist Temporal Classification**
- **RNNT - RNN Transducer**
- **SSL - Wav2Vec2.0**
- **WSL - Whisper**

## EdgeAI - AI on the edge

- **Transformer.js**

    See the [doc](https://xenova.github.io/transformers.js/) and [Source](https://github.com/xenova/transformers.js)

- **Llama.cpp**

    Inference of Meta's LLaMA model (and others) in pure C/C++. See their [source](https://github.com/ggerganov/llama.cpp).

- **llm.c**

    LLM training in simple, raw C/CUDA. See their [source](https://github.com/karpathy/llm.c).

## ML Sys

Shout out to Prof.Tianqi Han's [ML Sys](https://catalyst.cs.cmu.edu/15-884-mlsys-sp21/) on how to build and optimize ML systems, and Stanford's [CS129](https://gfxcourses.stanford.edu/cs149/fall23/) and UIUC's [ECE408](https://lumetta.web.engr.illinois.edu/408-S22/) on paralell computing.

- **Intro to CUDA**

    See my notes and code on [ECE408](https://github.com/neo99zhang/ece408), a great course for learning CUDA.


- **vLLM**

- **FlashAttention**

- **DeepSpeed**

- **ML Ops**

## Application

- **Codegen**

    [source](https://github.com/salesforce/CodeGen)

- **Whisper WebGPU**

    [app](https://huggingface.co/spaces/Xenova/whisper-webgpu): ML-powered speech recognition directly in your browser

- **Voice Assistant**

    [source](https://github.com/fatwang2/siri-ultra): Siri + llm inference + Cloudflare worker

- **Naming**

    GPT+RAG for classficial Chinese name
- **AIMusic**
