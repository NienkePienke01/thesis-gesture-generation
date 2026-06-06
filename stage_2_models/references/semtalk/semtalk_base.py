class semtalk_base(nn.Module):
    def __init__(self, args=None):
        super(semtalk_base, self).__init__()
        self.args = args   
        ######## constractive loss ########
        self.hubert_cons_loss = RhythmicIdentificationLoss(temperature=0.1)
        self.hubert_face_cons_loss = RhythmicIdentificationLoss(temperature=0.1)
        self.beat_cons_loss = RhythmicIdentificationLoss(temperature=0.1)
        self.beat_face_cons_loss = RhythmicIdentificationLoss(temperature=0.1)
        self.face_cons_mlp = MLP(256, args.hidden_size, 256)
        self.hands_cons_mlp = MLP(256, args.hidden_size, 256)
        

         # --------- HuBERT 特征预处理：数据集级归一化 + 轻扰动 ----------
        # self.hubert_dataset_norm = DatasetNorm1D(C=1024, fallback_layernorm=True)
        # self.feature_noise_sigma = float(getattr(args, "hubert_noise_sigma", 0.02))
        # self.feat_dropout = FeatureDropout(p=float(getattr(args, "hubert_featdrop_p", 0.05)))
        # self.time_mask = TimeMask(
        #     max_width=int(getattr(args, "hubert_timemask_w", 4)),
        #     num_mask=int(getattr(args, "hubert_timemask_n", 1))
        # )

        # --------- HuBERT 编码到 256 (Face/Body 各一份)，用 GN 替代 BN ----------
        def hubert_encoder_block():
            return nn.Sequential(
                nn.Conv1d(1024, 256, 3, 1, 1, bias=False),
                # nn.GroupNorm(32, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Conv1d(256, 256, 3, 1, 1, bias=False)
            )
        self.hubert_encoder = hubert_encoder_block()
        self.hubert_encoder_body = hubert_encoder_block()
        # 融合阶段 dropout
        # self.fuse_dropout = nn.Dropout(p=float(getattr(args, "fuse_dropout_p", 0.05)))

        ##### predict_residual #####
        self.predict_res_face = predict_residual_zq(latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5)
        self.predict_res_hands = predict_residual_zq(latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5)
        self.predict_res_upper = predict_residual_zq(latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5)
        self.predict_res_lower = predict_residual_zq(latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5)

        ##### con1d #####
        self.face1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.face1d_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.body1d = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=2, stride=2)
        self.upper1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.hands1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.lower1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.spearker_encoder_body1d = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=2, stride=2)
        self.face_latent_mlp = MLP(256, args.hidden_size, 256)
        
        self.audio_pre_encoder_face = MLP(3, args.hidden_size, 256)
        self.audio_pre_encoder_body = MLP(3, args.hidden_size, 256)
        self.at_attn_face = nn.Linear(args.audio_f*2, args.audio_f*2)
        self.at_attn_body = nn.Linear(args.audio_f*2, args.audio_f*2)
        args_top = copy.deepcopy(self.args)
        args_top.vae_layer = 3
        args_top.vae_length = args.motion_f  # args.motion_f 256
        args_top.vae_test_dim = args.pose_dims+3+4 # 330 + 3 + 4 = 337
        self.motion_encoder = VQEncoderV6(args_top) # masked motion to latent bs t 337 to bs t 256
        
        # face decoder  hidden_size:768, audio_f:256, vae_codebook_size:256
        self.feature2face = nn.Linear(args.audio_f*2, args.hidden_size) # 256*2 to 768
        self.face2latent = nn.Linear(args.hidden_size, args.vae_codebook_size) # vae_codebook_size:256
        self.transformer_de_layer = nn.TransformerDecoderLayer(
            d_model=self.args.hidden_size,
            nhead=4,
            dim_feedforward=self.args.hidden_size*2,
            batch_first=True
            )
        self.transformer_de_fu_layer = nn.TransformerDecoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=256*2,
            batch_first=True
            )
        self.hands_face_decoder = nn.TransformerDecoder(self.transformer_de_fu_layer, num_layers=1)
        self.face_hands_decoder = nn.TransformerDecoder(self.transformer_de_fu_layer, num_layers=1)

        self.face_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=4)
        # pose_length = 64
        self.position_embeddings = PeriodicPositionalEncoding(self.args.hidden_size, period=self.args.pose_length, max_seq_len=self.args.pose_length)
        
        # motion decoder
        self.transformer_en_layer = nn.TransformerEncoderLayer(
            d_model=self.args.hidden_size,
            nhead=4,
            dim_feedforward=self.args.hidden_size*2,
            batch_first=True
            )
        self.motion_self_encoder = nn.TransformerEncoder(self.transformer_en_layer, num_layers=1)
        self.audio_feature2motion = nn.Linear(args.audio_f, args.hidden_size) # 256 to 768
        self.feature2motion = nn.Linear(args.motion_f, args.hidden_size) # 256 to 768

        self.bodyhints_face = MLP(args.motion_f, args.hidden_size, args.motion_f) # 256 to 256
        self.bodyhints_body = MLP(args.motion_f, args.hidden_size, args.motion_f) # 256 to 256
        self.motion2latent_upper = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.motion2latent_hands = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.motion2latent_lower = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.wordhints_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=8)
        
        self.upper_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)
        self.hands_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)
        self.lower_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)

        self.upper_hands_decoder = nn.TransformerDecoder(self.transformer_de_fu_layer, num_layers=1)
        self.lower_hands_decoder = nn.TransformerDecoder(self.transformer_de_fu_layer, num_layers=1)

        self.face_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.upper_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.hands_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.lower_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)

        self.mask_embeddings = nn.Parameter(torch.zeros(1, 1, self.args.pose_dims+3+4)) # [1, 1, 337]
        self.motion_down_upper = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_hands = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_lower = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self._reset_parameters()


        self.spearker_encoder_body = nn.Embedding(25, args.hidden_size)
        self.spearker_encoder_face = nn.Embedding(25, args.hidden_size)

    def _reset_parameters(self):
        nn.init.normal_(self.mask_embeddings, 0, self.args.hidden_size ** -0.5)
    
    # 外部可在训练开始前注入 HuBERT 统计量（推荐）
    # @torch.no_grad()
    # def set_hubert_dataset_stats(self, mean_1024: torch.Tensor, std_1024: torch.Tensor):
    #     """
    #     mean/std: shape (1024,)
    #     """
    #     self.hubert_dataset_norm.set_stats(mean_1024.to(self.mask_embeddings.device),
    #                                        std_1024.to(self.mask_embeddings.device))
        
    def forward(self, in_audio=None, in_word=None, mask=None, is_train=False, in_motion=None, use_attentions=True, use_word=True, in_id = None, hubert=None):
         # ---- 1) HuBERT 预处理：标准化 + 训练期扰动 ----
        # hubert = self.hubert_dataset_norm(hubert)                 # 数据集级归一化/或 LN 回退
        # if is_train and self.feature_noise_sigma > 0:
        #     hubert = hubert + torch.randn_like(hubert) * self.feature_noise_sigma
        # if is_train:
        #     hubert = self.feat_dropout(hubert)                        # 通道dropout（仅训练）
        #     hubert = self.time_mask(hubert)                           # 时间遮挡（仅训练）

        in_word_face = self.hubert_encoder(hubert.permute(0, 2, 1)).permute(0, 2, 1)
        in_word_body = self.hubert_encoder_body(hubert.permute(0, 2, 1)).permute(0, 2, 1)
        bs, t, c = in_word_face.shape
        in_audio_face = self.audio_pre_encoder_face(in_audio) # [bs, t, 256]
        in_audio_body = self.audio_pre_encoder_body(in_audio) # [bs, t, 256]

        if use_attentions:           
            alpha_at_face = torch.cat([in_word_face, in_audio_face], dim=-1).reshape(bs, t, c*2)
            alpha_at_face = self.at_attn_face(alpha_at_face).reshape(bs, t, c, 2) # bs, t, c, 2
            alpha_at_face = alpha_at_face.softmax(dim=-1)
            fusion_face = in_word_face * alpha_at_face[:,:,:,1] + in_audio_face * alpha_at_face[:,:,:,0]
            alpha_at_body = torch.cat([in_word_body, in_audio_body], dim=-1).reshape(bs, t, c*2)
            alpha_at_body = self.at_attn_body(alpha_at_body).reshape(bs, t, c, 2)
            alpha_at_body = alpha_at_body.softmax(dim=-1)
            fusion_body = in_word_body * alpha_at_body[:,:,:,1] + in_audio_body * alpha_at_body[:,:,:,0]
        else:
            fusion_face = in_word_face + in_audio_face
            fusion_body = in_word_body + in_audio_body
        # print('fusion_face:', fusion_face.shape)
        # if is_train:
        #     fusion_face = self.fuse_dropout(fusion_face)
        #     fusion_body = self.fuse_dropout(fusion_body)
        masked_embeddings = self.mask_embeddings.expand_as(in_motion) # in_motion [bs, t, 337]
        # mask [bs, t, 337], 前四帧为0，后面为1
        masked_motion = torch.where(mask == 1, masked_embeddings, in_motion) 
        body_hint = self.motion_encoder(masked_motion) # bs t 256
        # print('id:', in_id.shape)
        
        speaker_embedding_face = self.spearker_encoder_face(in_id).squeeze(2) # bs, t, 768
        speaker_embedding_body = self.spearker_encoder_body(in_id).squeeze(2)

        # decode face
        use_body_hints = True
        if use_body_hints:
            body_hint_face = self.bodyhints_face(body_hint)
            fusion_face_a = torch.cat([fusion_face, body_hint_face], dim=2)
        a2g_face = self.feature2face(fusion_face_a)
        face_embeddings = speaker_embedding_face
        face_embeddings = self.position_embeddings(face_embeddings)
        decoded_face = self.face_decoder(tgt=face_embeddings, memory=a2g_face)
        face_latent = self.face2latent(decoded_face)
        face_latent = face_latent.permute(0, 2, 1)
        face_latent = self.face1d(face_latent)
        face_latent = face_latent.permute(0, 2, 1)
        face_latent = self.face_latent_mlp(face_latent)
        face_latent = face_latent.permute(0, 2, 1)
        face_latent = self.face1d_2(face_latent)

        face_latent_prezq = face_latent.permute(0, 2, 1)
        ######### hubert cons loss ###########
        hubert_cons_loss = self.hubert_face_cons_loss(face_latent_prezq, in_word_face)
        # ---- HuBERT 对比损失（对齐 + detach 音频端）----
        # in_word_face_for_loss = in_word_face.detach()
        # in_word_face_for_loss = match_time_to(in_word_face_for_loss, face_latent_prezq.shape[-1]).permute(0, 2, 1)  # -> (B,T,256)
        # hubert_cons_loss = self.hubert_face_cons_loss(face_latent_prezq.permute(0, 2, 1), in_word_face_for_loss)

        body_hint_body = self.bodyhints_body(body_hint) # MLP 256 to 256
        motion_embeddings = self.feature2motion(body_hint_body) # linear 256 to 768
        motion_embeddings = speaker_embedding_body + motion_embeddings # bs, t, 768
        motion_embeddings = self.position_embeddings(motion_embeddings) # bs, t, 768

        # bi-directional self-attention
        motion_refined_embeddings = self.motion_self_encoder(motion_embeddings) 
        
        # audio to gesture cross-modal attention
        if use_word:
            a2g_motion = self.audio_feature2motion(fusion_body) # linear 256 to 768
            motion_refined_embeddings_in = motion_refined_embeddings + speaker_embedding_body # bs, t, 768
            motion_refined_embeddings_in = self.position_embeddings(motion_refined_embeddings)
            word_hints = self.wordhints_decoder(tgt=motion_refined_embeddings_in, memory=a2g_motion)
            motion_refined_embeddings = motion_refined_embeddings + word_hints
        
        # feedforward
        motion_refined_embeddings = motion_refined_embeddings.permute(0, 2, 1)
        motion_refined_embeddings = self.body1d(motion_refined_embeddings)
        motion_refined_embeddings = motion_refined_embeddings.permute(0, 2, 1)
        speaker_embedding_body = speaker_embedding_body.permute(0, 2, 1)
        speaker_embedding_body = self.spearker_encoder_body1d(speaker_embedding_body)
        speaker_embedding_body = speaker_embedding_body.permute(0, 2, 1)

        upper_latent = self.motion2latent_upper(motion_refined_embeddings)
        hands_latent = self.motion2latent_hands(motion_refined_embeddings)
        lower_latent = self.motion2latent_lower(motion_refined_embeddings)

        upper_latent_in = upper_latent + speaker_embedding_body
        upper_latent_in = self.position_embeddings(upper_latent_in)
        hands_latent_in = hands_latent + speaker_embedding_body
        hands_latent_in = self.position_embeddings(hands_latent_in)
        lower_latent_in = lower_latent + speaker_embedding_body
        lower_latent_in = self.position_embeddings(lower_latent_in)

        # transformer decoder
        motion_upper = self.upper_decoder(tgt=upper_latent_in, memory=hands_latent+lower_latent)
        motion_hands = self.hands_decoder(tgt=hands_latent_in, memory=upper_latent+lower_latent)
        motion_lower = self.lower_decoder(tgt=lower_latent_in, memory=upper_latent+hands_latent)
        upper_latent = self.motion_down_upper(motion_upper+upper_latent) # linear 768 to 256
        hands_latent = self.motion_down_hands(motion_hands+hands_latent)
        lower_latent = self.motion_down_lower(motion_lower+lower_latent)

        upper_latent = upper_latent.permute(0, 2, 1)
        upper_latent = self.upper1d(upper_latent)
        upper_latent_prezq = upper_latent.permute(0, 2, 1)
       
        hands_latent = hands_latent.permute(0, 2, 1)
        hands_latent = self.hands1d(hands_latent)
        hands_latent_prezq = hands_latent.permute(0, 2, 1)

        ########## beat cons loss ###########
        beat_cons_loss = self.beat_cons_loss(hands_latent_prezq, in_word_body)
        # ---- Beat 对比损失（对齐 + detach 音频端）----
        # in_word_body_for_loss = in_word_body.detach()
        # in_word_body_for_loss = match_time_to(in_word_body_for_loss, hands_latent_prezq.shape[-1]).permute(0, 2, 1)
        # beat_cons_loss = self.beat_cons_loss(hands_latent_prezq.permute(0, 2, 1), in_word_body_for_loss)

        lower_latent = lower_latent.permute(0, 2, 1)
        lower_latent = self.lower1d(lower_latent)
        lower_latent_prezq = lower_latent.permute(0, 2, 1)

        hands_latent = self.hands_face_decoder(tgt=hands_latent_prezq, memory=face_latent_prezq)
        face_latent = self.face_hands_decoder(tgt=face_latent_prezq, memory=hands_latent_prezq)
        # face_latent = face_latent_prezq
        upper_latent = self.upper_hands_decoder(tgt = upper_latent_prezq, memory = hands_latent+lower_latent_prezq)
        lower_latent = self.lower_hands_decoder(tgt = lower_latent_prezq, memory = upper_latent_prezq+hands_latent)
        
        

        zq_index0_lower = self.lower_classifier(lower_latent)
        zq_index0_face = self.face_classifier(face_latent) # bs, t, 256

        zq1_face, zq2_face, zq3_face, zq4_face, zq5_face, zq_index1_face, zq_index2_face, zq_index3_face, zq_index4_face, zq_index5_face = self.predict_res_face(face_latent, fusion_face)
        # motion spatial encoder
        zq1_lower, zq2_lower, zq3_lower, zq4_lower, zq5_lower, zq_index1_lower, zq_index2_lower, zq_index3_lower, zq_index4_lower, zq_index5_lower = self.predict_res_lower(lower_latent, fusion_body)
        
        zq_index0_upper = self.upper_classifier(upper_latent)
        zq1_upper, zq2_upper, zq3_upper, zq4_upper, zq5_upper, zq_index1_upper, zq_index2_upper, zq_index3_upper, zq_index4_upper, zq_index5_upper = self.predict_res_upper(upper_latent, fusion_body)
        
        zq_index0_hands = self.hands_classifier(hands_latent)
        zq1_hands, zq2_hands, zq3_hands, zq4_hands, zq5_hands, zq_index1_hands, zq_index2_hands, zq_index3_hands, zq_index4_hands, zq_index5_hands = self.predict_res_hands(hands_latent, fusion_body)
        
        cls_face = torch.stack([zq_index0_face, zq_index1_face, zq_index2_face, zq_index3_face, zq_index4_face, zq_index5_face], dim=-1)
        cls_upper = torch.stack([zq_index0_upper, zq_index1_upper, zq_index2_upper, zq_index3_upper, zq_index4_upper, zq_index5_upper], dim=-1)
        cls_lower = torch.stack([zq_index0_lower, zq_index1_lower, zq_index2_lower, zq_index3_lower, zq_index4_lower, zq_index5_lower], dim=-1)
        cls_hands = torch.stack([zq_index0_hands, zq_index1_hands, zq_index2_hands, zq_index3_hands, zq_index4_hands, zq_index5_hands], dim=-1)

        rec_face = torch.stack([face_latent, zq1_face, zq2_face, zq3_face, zq4_face, zq5_face], dim=1).unsqueeze(2)
        rec_upper = torch.stack([upper_latent, zq1_upper, zq2_upper, zq3_upper, zq4_upper, zq5_upper], dim=1).unsqueeze(2)
        rec_lower = torch.stack([lower_latent, zq1_lower, zq2_lower, zq3_lower, zq4_lower, zq5_lower], dim=1).unsqueeze(2)
        rec_hands = torch.stack([hands_latent, zq1_hands, zq2_hands, zq3_hands, zq4_hands, zq5_hands], dim=1).unsqueeze(2)

        return {
            'hubert_cons_loss': hubert_cons_loss,
            'beat_cons_loss': beat_cons_loss,
            "rec_face":rec_face,
            "rec_upper":rec_upper,
            "rec_lower":rec_lower,
            "rec_hands":rec_hands,
            # "rec_face":face_latent,
            # "rec_upper":upper_latent,
            # "rec_lower":lower_latent,
            # "rec_hands":hands_latent,
            "cls_face":cls_face,
            "cls_upper":cls_upper,
            "cls_lower":cls_lower,
            "cls_hands":cls_hands,
            }
    
    def forward_latent(self, in_audio=None, in_word=None, mask=None, is_test=None, in_motion=None, use_attentions=True, use_word=True, in_id = None, hubert=None):
        in_word_face = self.hubert_encoder(hubert.permute(0, 2, 1)).permute(0, 2, 1)
        in_word_body = self.hubert_encoder_body(hubert.permute(0, 2, 1)).permute(0, 2, 1)
        # in_word_body = self.text_encoder_body(in_word_body)
        bs, t, c = in_word_face.shape
        in_audio_face = self.audio_pre_encoder_face(in_audio) # [bs, t, 256]
        in_audio_body = self.audio_pre_encoder_body(in_audio) # [bs, t, 256]
        

        if use_attentions:           
            alpha_at_face = torch.cat([in_word_face, in_audio_face], dim=-1).reshape(bs, t, c*2)
            alpha_at_face = self.at_attn_face(alpha_at_face).reshape(bs, t, c, 2) # bs, t, c, 2
            alpha_at_face = alpha_at_face.softmax(dim=-1)
            fusion_face = in_word_face * alpha_at_face[:,:,:,1] + in_audio_face * alpha_at_face[:,:,:,0]
            alpha_at_body = torch.cat([in_word_body, in_audio_body], dim=-1).reshape(bs, t, c*2)
            alpha_at_body = self.at_attn_body(alpha_at_body).reshape(bs, t, c, 2)
            alpha_at_body = alpha_at_body.softmax(dim=-1)
            fusion_body = in_word_body * alpha_at_body[:,:,:,1] + in_audio_body * alpha_at_body[:,:,:,0]
        else:
            fusion_face = in_word_face + in_audio_face
            fusion_body = in_word_body + in_audio_body
        # print('fusion_face:', fusion_face.shape)

        masked_embeddings = self.mask_embeddings.expand_as(in_motion) # in_motion [bs, t, 337]
        # mask [bs, t, 337], 前四帧为0，后面为1
        masked_motion = torch.where(mask == 1, masked_embeddings, in_motion) 
        body_hint = self.motion_encoder(masked_motion) # bs t 256
        # print('id:', in_id.shape)
        speaker_embedding_face = self.spearker_encoder_face(in_id).squeeze(2) # bs, t, 768
        speaker_embedding_body = self.spearker_encoder_body(in_id).squeeze(2)

        # decode face
        use_body_hints = True
        if use_body_hints:
            body_hint_face = self.bodyhints_face(body_hint)
            fusion_face_a = torch.cat([fusion_face, body_hint_face], dim=2)
        a2g_face = self.feature2face(fusion_face_a)
        face_embeddings = speaker_embedding_face
        face_embeddings = self.position_embeddings(face_embeddings)
        # print('face_embeddings:', face_embeddings.shape)
        decoded_face = self.face_decoder(tgt=face_embeddings, memory=a2g_face)
        face_latent = self.face2latent(decoded_face)
        face_latent = face_latent.permute(0, 2, 1)
        face_latent = self.face1d(face_latent)
        face_latent = face_latent.permute(0, 2, 1)
        face_latent = self.face_latent_mlp(face_latent)
        face_latent = face_latent.permute(0, 2, 1)
        face_latent = self.face1d_2(face_latent)
        face_latent_prezq = face_latent.permute(0, 2, 1)
        ######### hubert cons loss ###########
        hubert_cons_loss = self.hubert_face_cons_loss(face_latent_prezq, in_word_face)
       
        body_hint_body = self.bodyhints_body(body_hint) # MLP 256 to 256
        motion_embeddings = self.feature2motion(body_hint_body) # linear 256 to 768
        motion_embeddings = speaker_embedding_body + motion_embeddings # bs, t, 768
        motion_embeddings = self.position_embeddings(motion_embeddings) # bs, t, 768

        # bi-directional self-attention
        motion_refined_embeddings = self.motion_self_encoder(motion_embeddings) 
        
        # audio to gesture cross-modal attention
        if use_word:
            a2g_motion = self.audio_feature2motion(fusion_body) # linear 256 to 768
            motion_refined_embeddings_in = motion_refined_embeddings + speaker_embedding_body # bs, t, 768
            motion_refined_embeddings_in = self.position_embeddings(motion_refined_embeddings)
            word_hints = self.wordhints_decoder(tgt=motion_refined_embeddings_in, memory=a2g_motion)
            motion_refined_embeddings = motion_refined_embeddings + word_hints
        
        # feedforward
        motion_refined_embeddings = motion_refined_embeddings.permute(0, 2, 1)
        motion_refined_embeddings = self.body1d(motion_refined_embeddings)
        motion_refined_embeddings = motion_refined_embeddings.permute(0, 2, 1)
        speaker_embedding_body = speaker_embedding_body.permute(0, 2, 1)
        speaker_embedding_body = self.spearker_encoder_body1d(speaker_embedding_body)
        speaker_embedding_body = speaker_embedding_body.permute(0, 2, 1)

        upper_latent = self.motion2latent_upper(motion_refined_embeddings)
        hands_latent = self.motion2latent_hands(motion_refined_embeddings)
        lower_latent = self.motion2latent_lower(motion_refined_embeddings)

        upper_latent_in = upper_latent + speaker_embedding_body
        upper_latent_in = self.position_embeddings(upper_latent_in)
        hands_latent_in = hands_latent + speaker_embedding_body
        hands_latent_in = self.position_embeddings(hands_latent_in)
        lower_latent_in = lower_latent + speaker_embedding_body
        lower_latent_in = self.position_embeddings(lower_latent_in)

        # transformer decoder
        motion_upper = self.upper_decoder(tgt=upper_latent_in, memory=hands_latent+lower_latent)
        motion_hands = self.hands_decoder(tgt=hands_latent_in, memory=upper_latent+lower_latent)
        motion_lower = self.lower_decoder(tgt=lower_latent_in, memory=upper_latent+hands_latent)
        upper_latent = self.motion_down_upper(motion_upper+upper_latent) # linear 768 to 256
        hands_latent = self.motion_down_hands(motion_hands+hands_latent)
        lower_latent = self.motion_down_lower(motion_lower+lower_latent)

        upper_latent = upper_latent.permute(0, 2, 1)
        upper_latent = self.upper1d(upper_latent)
        upper_latent_prezq = upper_latent.permute(0, 2, 1)
       
        hands_latent = hands_latent.permute(0, 2, 1)
        hands_latent = self.hands1d(hands_latent)
        hands_latent_prezq = hands_latent.permute(0, 2, 1)

        ########## beat cons loss ###########
        beat_cons_loss = self.beat_cons_loss(hands_latent_prezq, in_word_body)

    
        lower_latent = lower_latent.permute(0, 2, 1)
        lower_latent = self.lower1d(lower_latent)
        lower_latent_prezq = lower_latent.permute(0, 2, 1)

        hands_latent = self.hands_face_decoder(tgt=hands_latent_prezq, memory=face_latent_prezq)
        face_latent = self.face_hands_decoder(tgt=face_latent_prezq, memory=hands_latent_prezq)
        # face_latent = face_latent_prezq
        upper_latent = self.upper_hands_decoder(tgt = upper_latent_prezq, memory = hands_latent+lower_latent_prezq)
        lower_latent = self.lower_hands_decoder(tgt = lower_latent_prezq, memory = upper_latent_prezq+hands_latent)
        return {
            'face_latent':face_latent,
            'upper_latent':upper_latent,
            'lower_latent':lower_latent,
            'hands_latent':hands_latent,
            }