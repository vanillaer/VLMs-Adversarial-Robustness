# VLMs-Adversarial-Robustness


This is a repository for the papers related to pretrained models and adversarial examples. The papers are mainly about the adversarial attacks and defenses for pretrained models, especially for Vision-Language Models (VLMs). The papers are mainly from top-tier conferences and journals, such as NeurIPS, ICML, ICLR, CVPR, ICCV, and arXiv. The papers are mainly from 2023 to 2024. 



# 1 multimodal adversarial attacks


## 1.0 ICCV2023

[[Set-level Guidance Attack: Boosting Adversarial Transferability of Vision-Language Pre-training Models]] (Oral)

代码：[https://github.com/Zoky-2020/SGA](https://link.zhihu.com/?target=https%3A//github.com/Zoky-2020/SGA)

摘要：视觉语言预训练（VLP）模型在**多模态任务**中表现出对抗样本的脆弱性。 此外，恶意对手可以故意迁移攻击其他黑盒模型。 然而，现有的工作主要集中在调查白盒攻击。 在本文中，我们提出了第一项研究近期 VLP 模型的**对抗迁移性**。 我们观察到，与白盒设置中的强大攻击性能相比，现有方法的对抗迁移性要低得多。迁移性下降的部分原因是跨模式交互的利用不足。特别是，与单模态学习不同，VLP 模型严重依赖于跨模态交互，并且多模态对齐是多对多的，例如，图像可以用各种自然语言来描述。 为此，我们提出了一种高度可迁移的集级指导攻击（SGA），它充分利用模态交互，并将对齐保持增强与跨模态指导相结合。实验结果表明，SGA 可以生成对抗样本，这些样本可以在多个下游视觉语言任务上的不同 VLP 模型之间进行强有力的迁移。 在**图文检索**方面，与现有技术相比，SGA 显着提高了从 ALBEF 到 TCL 的迁移攻击的攻击成功率（至少提高了 9.78％，最高提高了 30.21％）。


## 1.1 arxiv2024

[[OT-Attack: Enhancing Adversarial Transferability of Vision-Language Models via Optimal Transport Optimization]]

上一篇续作

keywords：Adversarial attacks for VLMs, text-img retrieval/matching tasks


## 1.2 arxiv2024

[[SA-Attack: Improving Adversarial Transferability of Vision-Language Pre-training Models via Self-Augmentation]]

上一篇续作

keywords：Adversarial attacks for VLMs, text-img retrieval/matching tasks 



## 1.3 NeurIPS 2023

后面还有一些做VLMs的visual question answering (VQA) and the visual reasoning (VR) task 的 adversarial attacks：

[[VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models]]

Vision-Language (VL) pre-trained models have shown their superiority on many multimodal tasks. However, the adversarial robustness of such models has not been fully explored. Existing approaches mainly focus on exploring the adversarial robustness under the white-box setting, which is unrealistic. In this paper, we aim to investigate a new yet practical task to craft image and text perturbations using pre-trained VL models to attack black-box fine-tuned models on different downstream tasks. Towards this end, we propose VLATTACK to generate adversarial samples by fusing perturbations of images and texts from both single-modal and multimodal levels. At the single-modal level, we propose a new block-wise similarity attack (BSA) strategy to learn image perturbations for disrupting universal representations. Besides, we adopt an existing text attack strategy to generate text perturbations independent of the image-modal attack. At the multimodal level, we design a novel iterative cross-search attack (ICSA) method to update adversarial image-text pairs periodically, starting with the outputs from the single-modal level. We conduct extensive experiments to attack five widely-used VL pre-trained models for six tasks. Experimental results show that VLATTACK achieves the highest attack success rates on all tasks compared with state-of-the-art baselines, which reveals a blind spot in the deployment of pre-trained VL models. Source codes can be found at https://github.com/ericyinyzy/VLAttack.


## 1.4 NeurIPS 2023

[[On Evaluating Adversarial Robustness of Large Vision-Language Models]]

Abstract: Large vision-language models (VLMs) such as GPT-4 have achieved unprecedented performance in response generation, especially with visual inputs, enabling more creative and adaptable interaction than large language models such as ChatGPT. Nonetheless, **multimodal generation** exacerbates safety concerns, since adversaries may successfully evade the entire system by subtly manipulating the most vulnerable modality (e.g., vision). To this end, we propose evaluating the robustness of open-source large VLMs in the most realistic and high-risk setting, where adversaries have only black-box system access and seek to deceive the model into returning the targeted responses. In particular, we first craft targeted adversarial examples against pretrained models such as CLIP and BLIP, and then transfer these adversarial examples to other VLMs such as MiniGPT-4, LLaVA, UniDiffuser, BLIP-2, and Img2Prompt. In addition, we observe that black-box queries on these VLMs can further improve the effectiveness of targeted evasion, resulting in a surprisingly high success rate for **generating targeted responses**. Our findings provide a quantitative understanding regarding the adversarial vulnerability of large VLMs and call for a more thorough examination of their potential security flaws before deployment in practice. Our project page: https://yunqing-me.github.io/AttackVLM/.

keywords: new task?(interaction question), 灰盒攻击加黑盒查询攻击(targeted)



# 2 Adversarial examples in downstream fine-tuning
- $ VLM Attacks:

## 2.1.1 arxiv2023

[[import-inkawhich2023adversarial-Adversarial Attacks on Foundational Vision Models]] 

Rapid progress is being made in developing large, pretrained, task-agnostic foundational vision models such as CLIP, ALIGN, DINOv2, etc. In fact, we are approaching the point where these models do not have to be finetuned downstream, and can simply be used in zero-shot or with a lightweight probing head. Critically, given the complexity of working at this scale, there is a bottleneck where relatively few organizations in the world are executing the training then sharing the models on centralized platforms such as HuggingFace and torch.hub. The goal of this work is to identify several key adversarial vulnerabilities of these models in an effort to make future designs more robust. Intuitively, our attacks manipulate deep feature representations to fool an out-of-distribution (OOD) detector which will be required when using these open-world-aware models to solve closed-set downstream tasks. Our methods reliably make in-distribution (ID) images (w.r.t. a downstream task) be predicted as OOD and vice versa while existing in extremely low-knowledge-assumption threat models. We show our attacks to be potent in whitebox and blackbox settings, as well as when transferred across foundational model types (e.g., attack DINOv2 with CLIP)! This work is only just the beginning of a long journey towards adversarially robust foundational vision models.

keywords：classification task


## 2.1.2 ACM MM2023

[[import-zhou2023advclip-AdvCLIP - Downstream-agnostic Adversarial Examples in Multimodal Contrastive Learning]]

Multimodal contrastive learning aims to train a general-purpose feature extractor, such as CLIP, on vast amounts of raw, unlabeled paired image-text data. This can greatly benefit various complex downstream tasks, including cross-modal image-text retrieval and image classification. Despite its promising prospect, the security issue of cross-modal pre-trained encoder has not been fully explored yet, especially when the pre-trained encoder is publicly available for commercial use. In this work, we propose AdvCLIP, the first attack framework for generating downstream-agnostic adversarial examples based on cross-modal pre-trained encoders. AdvCLIP aims to construct a universal adversarial patch for a set of natural images that can fool all the downstream tasks inheriting the victim cross-modal pre-trained encoder. To address the challenges of heterogeneity between different modalities and unknown downstream tasks, we first build a topological graph structure to capture the relevant positions between target samples and their neighbors. Then, we design a topology-deviation based generative adversarial network to generate a universal adversarial patch. By adding the patch to images, we minimize their embeddings similarity to different modality and perturb the sample distribution in the feature space, achieving unviersal non-targeted attacks. Our results demonstrate the excellent attack performance of AdvCLIP on two types of downstream tasks across eight datasets. We also tailor three popular defenses to mitigate AdvCLIP, highlighting the need for new defense mechanisms to defend cross-modal pre-trained encoders.


keywords：Universal adversarial attacks for VLMs, Classification tasks, text-img retrieval/matching tasks



- $ defence:

## 2.2.1 arxiv2024

[[import-zhang2023adversarial-Adversarial Prompt Tuning for Vision-Language Models]]

With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code is available at https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.

keywords：defense Adversarial attacks for CLIP through adversarial training, Classification tasks



## 2.2.2 ICLR2023

[[import-mao2023understanding-Understanding Zero-Shot Adversarial Robustness for Large-Scale Models]]

Pretrained large-scale vision-language models like CLIP have exhibited strong generalization over unseen tasks. Yet imperceptible adversarial perturbations can significantly reduce CLIP's performance on new tasks. In this work, we identify and explore the problem of \emph{adapting large-scale models for zero-shot adversarial robustness}. We first identify two key factors during model adaption -- training losses and adaptation methods -- that affect the model's zero-shot adversarial robustness. We then propose a text-guided contrastive adversarial training loss, which aligns the text embeddings and the adversarial visual features with contrastive learning on a small set of training data. We apply this training loss to two adaption methods, model finetuning and visual prompt tuning. We find that visual prompt tuning is more effective in the absence of texts, while finetuning wins in the existence of text guidance. Overall, our approach significantly improves the zero-shot adversarial robustness over CLIP, seeing an average improvement of over 31 points over ImageNet and 15 zero-shot datasets. We hope this work can shed light on understanding the zero-shot adversarial robustness of large-scale models.

adv defense(contrastive adv training), fine-tuning, VLMs（CLIP）


## 2.2.3 arxiv2024

[[import-wang2024pretrained-Pre-trained Model Guided Fine-Tuning for Zero-Shot Adversarial Robustness]]

上一篇续作


## 2.2.4 arxiv2024

[[import-cai2023clap-CLAP - Contrastive Learning with Augmented Prompts for Robustness on Pretrained Vision-Language Models]]

Contrastive vision-language models, e.g., CLIP, have garnered substantial attention for their exceptional generalization capabilities. However, their robustness to perturbations has ignited concerns. Existing strategies typically reinforce their resilience against adversarial examples by enabling the image encoder to "see" these perturbed examples, often necessitating a complete retraining of the image encoder on both natural and adversarial samples. In this study, we propose a new method to enhance robustness solely through text augmentation, eliminating the need for retraining the image encoder on adversarial examples. Our motivation arises from the realization that text and image data inherently occupy a shared latent space, comprising latent content variables and style variables. This insight suggests the feasibility of learning to disentangle these latent content variables using text data exclusively. To accomplish this, we introduce an effective text augmentation method that focuses on modifying the style while preserving the content in the text data. By changing the style part of the text data, we empower the text encoder to emphasize latent content variables, ultimately enhancing the robustness of vision-language models. Our experiments across various datasets demonstrate substantial improvements in the robustness of the pre-trained CLIP model.

text data aug as surrogate, VLMs（CLIP）



# 3 backdoor attacks for downstream fine-tuning

## 3.1 arxiv2024

[[import-bai2023badclip-BadCLIP - Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP]]

Contrastive Vision-Language Pre-training, known as CLIP, has shown promising effectiveness in addressing downstream image recognition tasks. However, recent works revealed that the CLIP model can be implanted with a downstream-oriented backdoor. On downstream tasks, one victim model performs well on clean samples but predicts a specific target class whenever a specific trigger is present. For injecting a backdoor, existing attacks depend on a large amount of additional data to maliciously fine-tune the entire pre-trained CLIP model, which makes them inapplicable to data-limited scenarios. In this work, motivated by the recent success of learnable prompts, we address this problem by injecting a backdoor into the CLIP model in the prompt learning stage. Our method named BadCLIP is built on a novel and effective mechanism in backdoor attacks on CLIP, i.e., influencing both the image and text encoders with the trigger. It consists of a learnable trigger applied to images and a trigger-aware context generator, such that the trigger can change text features via trigger-aware prompts, resulting in a powerful and generalizable attack. Extensive experiments conducted on 11 datasets verify that the clean accuracy of BadCLIP is similar to those of advanced prompt learning methods and the attack success rate is higher than 99% in most cases. BadCLIP is also generalizable to unseen classes, and shows a strong generalization capability under cross-dataset and cross-domain settings.

keywords：backdoor attacks for CLIP（especially when text prompts tuning）, Classification tasks




# 4 Standard adv for Test-time Defenses


## ICML2022

[[import-croce2022evaluatinga-Evaluating the Adversarial Robustness of Adaptive Test-time Defenses]]

keywords: survey, evaluation


## neurips2023

[[import-tsai2024convolutional-Convolutional Visual Prompt for Robust Visual Perception]]

OOD task, normal pretrained model, 有一个labeled source domain来训练辅助网络


## Neurips2022（Workshop）

[[import-chen2022visual-Visual Prompting for Adversarial Robustness]]


## arXiv2023

[[import-tsai2023testtime-Test-time Detection and Repair of Adversarial Samples via Masked Autoencoder]]

SSL optimization objective surrogate



## arxiv2023

[[import-li2023languagedriven-Language-Driven Anchors for Zero-Shot Adversarial Robustness]]




## ICLR2021 （W）

[[import-wang2021fighting-Fighting Gradients with Gradients - Dynamic Defenses against Adversarial Attacks]]


## CVPR2023

[[import-frosio2023best-The Best Defense Is a Good Offense - Adversarial Augmentation Against Adversarial Attacks]]

certified defence


## arxiv2023

[[import-palumbo2023two-Two Heads are Better than One - Towards Better Adversarial Robustness by Combining Transduction and Rejection]]





# 5 其他一些非多模态，但是与pretained-model相关的文章：

## 5.1 ICCV2023

[Improving Generalization of Adversarial Training via Robust Critical Fine-Tuning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2308.02533)

代码：[GitHub - microsoft/robustlearn: Robust machine learning for responsible AI](https://link.zhihu.com/?target=https%3A//github.com/microsoft/robustlearn)

摘要：深度神经网络容易受到对抗样本的影响，在关键应用中构成重大安全风险。 对抗训练（AT）是一种行之有效的增强对抗鲁棒性的技术，但它通常以泛化能力下降为代价。 本文提出了鲁棒性关键微调（RiFT），这是一种在不影响对抗鲁棒性的情况下增强泛化性的新方法。 RiFT 的核心思想是通过在其非鲁棒关键模块上微调经过对抗训练的模型来利用冗余能力来实现鲁棒性。 为此，我们引入了模块鲁棒临界性（MRC），这是一种评估给定模块在最坏情况权重扰动下对模型鲁棒性的重要性的度量。 使用这种方法，我们将具有最低 MRC 值的模块识别为非鲁棒关键模块，并对其权重进行微调以获得微调权重。 随后，我们在对抗训练的权重和微调权重之间进行线性插值，以得出最佳的微调模型权重。我们展示了 RiFT 在 CIFAR10、CIFAR100 和 Tiny-ImageNet 数据集上训练的 ResNet18、ResNet34 和 WideResNet34-10 模型上的功效。 我们的实验表明，本方法可以将泛化和分布外鲁棒性显着提高约 1.5％，同时保持甚至稍微增强对抗鲁棒性。


## 5.2 ICCV2023

[[import-zhou2023downstreamagnostic-Downstream-agnostic Adversarial Examples]]

Self-supervised learning usually uses a large amount of unlabeled data to pre-train an encoder which can be used as a general-purpose feature extractor, such that downstream users only need to perform fine-tuning operations to enjoy the benefit of "large model". Despite this promising prospect, the security of pre-trained encoder has not been thoroughly investigated yet, especially when the pre-trained encoder is publicly available for commercial use. In this paper, we propose AdvEncoder, the first framework for generating downstream-agnostic universal adversarial examples based on the pre-trained encoder. AdvEncoder aims to construct a universal adversarial perturbation or patch for a set of natural images that can fool all the downstream tasks inheriting the victim pre-trained encoder. Unlike traditional adversarial example works, the pre-trained encoder only outputs feature vectors rather than classification labels. Therefore, we first exploit the high frequency component information of the image to guide the generation of adversarial examples. Then we design a generative attack framework to construct adversarial perturbations/patches by learning the distribution of the attack surrogate dataset to improve their attack success rates and transferability. Our results show that an attacker can successfully attack downstream tasks without knowing either the pre-training dataset or the downstream dataset. We also tailor four defenses for pre-trained encoders, the results of which further prove the attack ability of AdvEncoder.


keywords：Universal adversarial attacks for pretrained models, Classification tasks



## 5.3 NeurIPS 2022

[[import-ban2022pretrained-Pre-trained Adversarial Perturbations]]

Self-supervised pre-training has drawn increasing attention in recent years due to its superior performance on numerous downstream tasks after fine-tuning. However, it is well-known that deep learning models lack the robustness to adversarial examples, which can also invoke security issues to pre-trained models, despite being less explored. In this paper, we delve into the robustness of pre-trained models by introducing Pre-trained Adversarial Perturbations (PAPs), which are universal perturbations crafted for the pre-trained models to maintain the effectiveness when attacking fine-tuned ones without any knowledge of the downstream tasks. To this end, we propose a Low-Level Layer Lifting Attack (L4A) method to generate effective PAPs by lifting the neuron activations of low-level layers of the pre-trained models. Equipped with an enhanced noise augmentation strategy, L4A is effective at generating more transferable PAPs against the fine-tuned models. Extensive experiments on typical pre-trained vision models and ten downstream tasks demonstrate that our method improves the attack success rate by a large margin compared to the state-of-the-art methods.

keywords：Adversarial attacks for pretrained models, Classification tasks


# Standard adv examples
- $ attacks:

## ICCV2023

[[import-chen2023advdiffuser-AdvDiffuser - Natural Adversarial Example Synthesis with Diffusion Models]]

diffusion model for generating adv example 




# VLM-adv相关survey:


- Multi-Modal Learning System的综述：

[[import-zhao2024survey-A Survey on Safe Multi-Modal Learning System]]

With the wide deployment of multimodal learning systems (MMLS) in real-world scenarios, safety concerns have become increasingly prominent. The absence of systematic research into their safety is a significant barrier to progress in this field. To bridge the gap, we present the first taxonomy for MMLS safety, identifying four essential pillars of these concerns. Leveraging this taxonomy, we conduct in-depth reviews for each pillar, highlighting key limitations based on the current state of development. Finally, we pinpoint unique challenges in MMLS safety and provide potential directions for future research.




- poison attack的综述：

[[import-cina2023wild-Wild Patterns Reloaded - A Survey of Machine Learning Security against Training Data Poisoning]]

The success of machine learning is fueled by the increasing availability of computing power and large training datasets. The training data is used to learn new models or update existing ones, assuming that it is sufficiently representative of the data that will be encountered at test time. This assumption is challenged by the threat of poisoning, an attack that manipulates the training data to compromise the model's performance at test time. Although poisoning has been acknowledged as a relevant threat in industry applications, and a variety of different attacks and defenses have been proposed so far, a complete systematization and critical review of the field is still missing. In this survey, we provide a comprehensive systematization of poisoning attacks and defenses in machine learning, reviewing more than 100 papers published in the field in the last 15 years. We start by categorizing the current threat models and attacks, and then organize existing defenses accordingly. While we focus mostly on computer-vision applications, we argue that our systematization also encompasses state-of-the-art attacks and defenses for other data modalities. Finally, we discuss existing resources for research in poisoning, and shed light on the current limitations and open research questions in this research field.



- Adv Robustness的综述：

[[import-liu2023comprehensive-A Comprehensive Study on Robustness of Image Classification Models - Benchmarking and Rethinking]]


- Parameter-Efficient Fine-Tuning综述：

[[import-xin2024parameterefficient-Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models - A Survey]]

