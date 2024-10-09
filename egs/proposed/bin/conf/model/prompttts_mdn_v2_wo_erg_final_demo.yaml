_target_: promptttspp.models.prompttts_mdn_v2_final.model.PromptTTSMDNDurCFG

norm_style_emb: true
mdn_disable_amp: true

phoneme_embedding:
  _target_: promptttspp.layers.embedding.PhonemeEmbedding
  num_vocab: 90
  channels: 256
  do_scale: false
  init_normal: false

encoder:
  _target_: promptttspp.modules.esp.ConformerEncoder
  idim: 256
  attention_dim: 256
  attention_heads: 2
  linear_units: 1024
  num_blocks: 4
  positionwise_layer_type: conv1d
  positionwise_conv_kernel_size: 9
  dropout_rate: 0.2
  pos_enc_layer_type: rel_pos
  selfattention_layer_type: rel_selfattn
  activation_type: swish
  macaron_style: true
  use_cnn_module: true
  cnn_module_kernel: 7
  return_mask: false
  rel_pos_type: legacy

variance_adaptor:
  _target_: promptttspp.modules.variance_adaptor.VarianceAdaptor
  duration_predictor:
    _target_: promptttspp.modules.variance_adaptor.MDNPredictor
    channels: ${...phoneme_embedding.channels}
    out_channels: 1
    kernel_size: 3
    dropout: 0.5
    num_layers: 2
    num_gaussians: 4
    detach: true
    disable_amp: ${...mdn_disable_amp}
  pitch_predictor:
    _target_: promptttspp.modules.variance_adaptor.Predictor
    channels: ${...phoneme_embedding.channels}
    out_channels: 2
    kernel_size: 5
    dropout: ${..duration_predictor.dropout}
    num_layers: 5
    detach: false
  pitch_emb:
    _target_: torch.nn.Conv1d
    in_channels: 1
    out_channels: ${...phoneme_embedding.channels}
    kernel_size: 1
  energy_predictor: null
  energy_emb: null
  frame_prior_network:
    _target_: promptttspp.modules.frame_prior.FramePriorNetwork
    out_channels: ${...phoneme_embedding.channels}
    hidden_channels: ${...phoneme_embedding.channels}
    n_layers: 6
    kernel_size: 17
    p_dropout: 0.1

reference_encoder:
  _target_: promptttspp.modules.style_encoder.StyleEncoder
  idim: 80
  gst_tokens: 10
  gst_heads: 4
  conv_layers: 6
  conv_chans_list: [128, 128, 256, 256, 512, 512]
  conv_kernel_size: 3
  conv_stride: 2
  gru_layers: 1
  gru_units: ${..phoneme_embedding.channels}

prompt_encoder:
  _target_: promptttspp.modules.prompt_encoder.PromptEncoder
  model_name: bert-base-uncased
  in_channels: 768
  mid_channels: 512
  out_channels: ${..phoneme_embedding.channels}

style_mdn:
  _target_: promptttspp.modules.mdn.MDNLayer
  in_dim: ${..phoneme_embedding.channels}
  out_dim: ${..phoneme_embedding.channels}
  num_gaussians: 10
  dim_wise: true

decoder:
  _target_: promptttspp.modules.diffusion.GaussianDiffusion
  in_dim: ${..encoder.attention_dim}
  out_dim: 80
  norm_scale: 6.0
  denoise_fn:
    _target_: promptttspp.modules.denoiser.DiffNet
    in_dim: 80
    encoder_hidden_dim: ${...phoneme_embedding.channels}
    residual_layers: 20
    residual_channels: 256
    kernel_size: 3
    dilation_cycle_length: 4
