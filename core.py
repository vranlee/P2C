# ------------------------------------------------------------------------
# Supplementary Material - Core Implementation Source Code
# From Points to Clouds: Learning Robust Semantic Distributions for Vision-Language Prompting
# ------------------------------------------------------------------------


class GaussianMixtureNoiseGenerator(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()
        self.num_components = cfg.TRAINER.PROMPT_DENOISING.GMM_COMPONENTS
        self.gmm_means = cfg.TRAINER.PROMPT_DENOISING.GMM_MEANS
        self.gmm_stds = cfg.TRAINER.PROMPT_DENOISING.GMM_STDS

        internal_dtype = torch.float32

        self.mix_weights = torch.ones(self.num_components, dtype=internal_dtype, device=device) / self.num_components
        self.means = torch.tensor(self.gmm_means, dtype=internal_dtype, device=device)
        self.stds = torch.tensor(self.gmm_stds, dtype=internal_dtype, device=device)

        print(f"Initialized GaussianMixtureNoiseGenerator with {self.num_components} components.")
        print(f"Means: {self.gmm_means}, Stds: {self.gmm_stds}")

    def forward(self, tensor_like):

        mix = Categorical(self.mix_weights)
        comp = Normal(self.means, self.stds)
        gmm = MixtureSameFamily(mix, comp)

        noise = gmm.sample(tensor_like.shape)

        return noise.to(device=tensor_like.device, dtype=tensor_like.dtype)

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = 768
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if False:
            ctx_init = "a photo of a"
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")

        self.v_t_mapper_cfg = cfg.TRAINER.MAPLE.V_T_MAPPER
        if self.v_t_mapper_cfg.ENABLED:
            print("Diffusion-Inspired Vision-Text Mapping is ENABLED.")
            hidden_dim = int(ctx_dim * self.v_t_mapper_cfg.HIDDEN_SCALE)
            self.proj = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(ctx_dim, hidden_dim)),
                ('gelu', nn.GELU()),
                ('fc2', nn.Linear(hidden_dim, vis_dim))
            ]))
            print(f"Upgraded V-T mapper to MLP with hidden dim: {hidden_dim}")
        else:
            print("Using standard Linear Vision-Text Mapping.")
            self.proj = nn.Linear(ctx_dim, vis_dim)

        self.ctx = nn.Parameter(ctx_vectors)

        self.use_atp = cfg.TRAINER.ATPROMPT.USE_ATPROMPT
        self.atp_num = cfg.TRAINER.ATPROMPT.ATT_NUM

        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512, dtype=dtype))
                                                       for _ in range(self.compound_prompts_depth - 1)])

        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        if self.use_atp:
            print("USE ATPROPMT-ING 1")
            n_att1 = cfg.TRAINER.ATPROMPT.N_ATT1
            att1_text = cfg.TRAINER.ATPROMPT.ATT1_TEXT
            n_att2 = cfg.TRAINER.ATPROMPT.N_ATT2
            att2_text = cfg.TRAINER.ATPROMPT.ATT2_TEXT
            n_att3 = cfg.TRAINER.ATPROMPT.N_ATT3
            att3_text = cfg.TRAINER.ATPROMPT.ATT3_TEXT

            att_vectors_1 = torch.empty(n_att1, ctx_dim, dtype=dtype)
            att_vectors_2 = torch.empty(n_att2, ctx_dim, dtype=dtype)
            att_vectors_3 = torch.empty(n_att3, ctx_dim, dtype=dtype)

            nn.init.normal_(att_vectors_1, std=0.01)
            prefix1 = " ".join(["X"] * n_att1)
            nn.init.normal_(att_vectors_2, std=0.01)
            prefix2 = " ".join(["X"] * n_att2)
            nn.init.normal_(att_vectors_3, std=0.01)
            prefix3 = " ".join(["X"] * n_att3)

            self.ctx_att1 = nn.Parameter(att_vectors_1)
            self.ctx_att2 = nn.Parameter(att_vectors_2)
            self.ctx_att3 = nn.Parameter(att_vectors_3)

            if self.atp_num == 1:
                prompts = [prefix1 + " " + att1_text + " " + prompt_prefix + " " + name + "." for name in classnames]
            elif self.atp_num == 2:
                prompts = [
                    prefix1 + " " + att1_text + " " + prefix2 + " " + att2_text + " " + prompt_prefix + " " + name + "."
                    for name in classnames]
            elif self.atp_num == 3:
                prompts = [
                    prefix1 + " " + att1_text + " " + prefix2 + " " + att2_text + " " + prefix3 + " " + att3_text + " " + prompt_prefix + " " + name + "."
                    for name in classnames]
            else:
                print("wrong parameter.")
        else:
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])

        if self.use_atp:
            print("USE ATPROPMT-ING 2")
            if self.atp_num == 1:
                self.register_buffer("token_middle1", embedding[:, 1 + n_att1: 1 + n_att1 + 1, :])
                self.register_buffer("token_suffix", embedding[:, 1 + n_att1 + 1 + n_ctx:, :])

            elif self.atp_num == 2:
                self.register_buffer("token_middle1", embedding[:, 1 + n_att1: 1 + n_att1 + 1, :])
                self.register_buffer("token_middle2",
                                     embedding[:, 1 + n_att1 + 1 + n_att2: 1 + n_att1 + 1 + n_att2 + 1, :])
                self.register_buffer("token_suffix", embedding[:, 1 + n_att1 + 1 + n_att2 + 1 + n_ctx:, :])

            elif self.atp_num == 3:
                self.register_buffer("token_middle1", embedding[:, 1 + n_att1: 1 + n_att1 + 1, :])
                self.register_buffer("token_middle2",
                                     embedding[:, 1 + n_att1 + 1 + n_att2: 1 + n_att1 + 1 + n_att2 + 1, :])
                self.register_buffer("token_middle3", embedding[
                    :, 1 + n_att1 + 1 + n_att2 + 1 + n_att3: 1 + n_att1 + 1 + n_att2 + 1 + n_att3 + 1, :])
                self.register_buffer("token_suffix", embedding[:, 1 + n_att1 + 1 + n_att2 + 1 + n_att3 + 1 + n_ctx:, :])
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

        self.pd_cfg = cfg.TRAINER.PROMPT_DENOISING
        if self.pd_cfg.ENABLED:
            print("Diffusion-Inspired Prompt Denoising is ENABLED.")
            self.noise_generator = None
            if self.pd_cfg.NOISE_TYPE == "gaussian_mixture":
                self.noise_generator = GaussianMixtureNoiseGenerator(cfg, self.ctx.device)
            print(f"Noise Type: {self.pd_cfg.NOISE_TYPE}, Noise STD: {self.pd_cfg.NOISE_STD}, Schedule: {self.pd_cfg.SCHEDULE}, Warmup: {self.pd_cfg.WARMUP_EPOCHS}")

    def construct_prompts(self, prefix, ctx, suffix, ctx_att1=None, middle_attribute1=None, ctx_att2=None,
                          middle_attribute2=None, ctx_att3=None, middle_attribute3=None, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        if self.use_atp:
            if self.atp_num == 1:
                prompts = torch.cat(
                    [
                        prefix,
                        ctx_att1,
                        middle_attribute1,
                        ctx,
                        suffix,
                    ],
                    dim=1,
                )
            elif self.atp_num == 2:
                prompts = torch.cat(
                    [
                        prefix,
                        ctx_att1,
                        middle_attribute1,
                        ctx_att2,
                        middle_attribute2,
                        ctx,
                        suffix,
                    ],
                    dim=1,
                )
            elif self.atp_num == 3:
                prompts = torch.cat(
                    [
                        prefix,
                        ctx_att1,
                        middle_attribute1,
                        ctx_att2,
                        middle_attribute2,
                        ctx_att3,
                        middle_attribute3,
                        ctx,
                        suffix,
                    ],
                    dim=1,
                )
        else:
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )

        return prompts

    def _get_noise_scale(self, epoch, max_epoch):
        if not (self.training and self.pd_cfg.ENABLED and epoch is not None and max_epoch is not None):
            return 0.0

        effective_max_epoch = max_epoch - self.pd_cfg.WARMUP_EPOCHS
        effective_epoch = epoch - self.pd_cfg.WARMUP_EPOCHS

        if effective_epoch < 0:
            return self.pd_cfg.NOISE_STD

        progress = min(effective_epoch / effective_max_epoch, 1.0)

        if self.pd_cfg.SCHEDULE == "cosine":
            return self.pd_cfg.NOISE_STD * 0.5 * (1 + math.cos(math.pi * progress))
        elif self.pd_cfg.SCHEDULE == "linear":
            return self.pd_cfg.NOISE_STD * (1 - progress)
        elif self.pd_cfg.SCHEDULE == "sigmoid":
            k = self.pd_cfg.SIGMOID_K
            sigmoid_val = 1 / (1 + math.exp(k * (progress - 0.5)))
            rescaled_sigmoid = (sigmoid_val - 1 / (1 + math.exp(k * 0.5))) / \
                               (1 / (1 + math.exp(-k * 0.5)) - 1 / (1 + math.exp(k * 0.5)))
            return self.pd_cfg.NOISE_STD * (1 - rescaled_sigmoid)
        else:
            return self.pd_cfg.NOISE_STD

    def _generate_noise(self, tensor, noise_scale):
        if noise_scale <= 0:
            return 0

        if self.pd_cfg.NOISE_TYPE == "gaussian_mixture":
            return self.noise_generator(tensor) * noise_scale
        else:
            return torch.randn_like(tensor) * noise_scale

    def forward(self, epoch=None, max_epoch=None):
        prefix = self.token_prefix
        suffix = self.token_suffix

        ctx = self.ctx
        if self.use_atp:
            ctx_att1 = self.ctx_att1
            ctx_att2 = self.ctx_att2
            ctx_att3 = self.ctx_att3

        current_noise_scale = self._get_noise_scale(epoch, max_epoch)

        if current_noise_scale > 0:
            ctx = ctx + self._generate_noise(ctx, current_noise_scale)
            if self.use_atp:
                ctx_att1 = ctx_att1 + self._generate_noise(ctx_att1, current_noise_scale)
                ctx_att2 = ctx_att2 + self._generate_noise(ctx_att2, current_noise_scale)
                ctx_att3 = ctx_att3 + self._generate_noise(ctx_att3, current_noise_scale)

        shared_ctx = self.proj(self.ctx)

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            if self.use_atp:
                ctx_att1 = ctx_att1.unsqueeze(0).expand(self.n_cls, -1, -1)
                ctx_att2 = ctx_att2.unsqueeze(0).expand(self.n_cls, -1, -1)
                ctx_att3 = ctx_att3.unsqueeze(0).expand(self.n_cls, -1, -1)

        if self.use_atp:
            if self.atp_num == 1:
                middle_attribute1 = self.token_middle1
                prompts = self.construct_prompts(prefix, ctx, suffix, ctx_att1, middle_attribute1)
            elif self.atp_num == 2:
                middle_attribute1 = self.token_middle1
                middle_attribute2 = self.token_middle2
                prompts = self.construct_prompts(prefix, ctx, suffix, ctx_att1, middle_attribute1, ctx_att2,
                                                 middle_attribute2)
            elif self.atp_num == 3:
                middle_attribute1 = self.token_middle1
                middle_attribute2 = self.token_middle2
                middle_attribute3 = self.token_middle3
                prompts = self.construct_prompts(prefix, ctx, suffix, ctx_att1, middle_attribute1, ctx_att2,
                                                 middle_attribute2, ctx_att3, middle_attribute3)
            else:
                raise ValueError
        else:
            prompts = self.construct_prompts(prefix, ctx, suffix)

        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))

        return prompts, shared_ctx, self.compound_prompts_text, visual_deep_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.prompt_learner.to(self.dtype)

        self.v_t_mapper_cfg = self.prompt_learner.v_t_mapper_cfg
        if self.v_t_mapper_cfg.ENABLED:
            self.aux_loss_weight = self.v_t_mapper_cfg.LOSS_WEIGHT
            self.aux_noise_std = self.v_t_mapper_cfg.NOISE_STD
            print(
                f"Auxiliary V-T Mapper Denoising Loss enabled with weight={self.aux_loss_weight} and noise_std={self.aux_noise_std}")

    def forward(self, image, label=None, epoch=None, max_epoch=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(epoch=epoch,
                                                                                                            max_epoch=max_epoch)
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, None,
                                            None, None, None)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.training:
            classification_loss = F.cross_entropy(logits, label)
            total_loss = classification_loss

            if self.v_t_mapper_cfg.ENABLED:
                clean_text_prompt = self.prompt_learner.ctx
                clean_vision_prompt = shared_ctx.detach()

                noise = torch.randn_like(clean_text_prompt) * self.aux_noise_std
                noisy_text_prompt = clean_text_prompt + noise

                reconstructed_vision_prompt = self.prompt_learner.proj(noisy_text_prompt)

                consistency_loss = F.mse_loss(reconstructed_vision_prompt, clean_vision_prompt)

                total_loss = total_loss + self.aux_loss_weight * consistency_loss

            return total_loss

        return logits