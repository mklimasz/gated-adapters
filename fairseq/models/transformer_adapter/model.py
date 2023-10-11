from fairseq.distributed import fsdp_wrap
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder, base_architecture
from fairseq.models.transformer_adapter.transformer_adapter_layer import (
    TransformerAdapterEncoderLayer,
    TransformerAdapterDecoderLayer
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper


@register_model("transformer_adapter")
class TransformerAdapterModel(TransformerModel):
    """The code is based on fairseq.models.xmod.XMODModel.
    (https://aclanthology.org/2022.naacl-main.255.pdf)
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        if args.freeze_non_adapter_params:
            self._freeze_non_adapter()
        if args.unfreeze_layer_norms:
            self._unfreeze_layer_norms()

    def _freeze_non_adapter(self):
        for name, param in self.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

    def _unfreeze_layer_norms(self):
        for name, param in self.named_parameters():
            if "layer_norm" in name:
                param.requires_grad = True

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerAdapterEncoder(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens, gating=None):
        return TransformerAdapterDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
            gating=gating
        )


class TransformerAdapterEncoder(TransformerEncoder):

    def build_encoder_layer(self, cfg, i, gating=None):
        layer = TransformerAdapterEncoderLayer(cfg, i, gating)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


class TransformerAdapterDecoder(TransformerDecoder):

    def build_decoder_layer(self, cfg, no_encoder_attn=False, gating=None):
        layer = TransformerAdapterDecoderLayer(cfg, no_encoder_attn, gating)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


@register_model_architecture("transformer_adapter", "transformer_adapter_base")
def adapter_architecture_base(args):
    # Adapter's like X-MOD.
    # https://aclanthology.org/2022.naacl-main.255.pdf
    args.freeze_non_adapter_params = getattr(args, "freeze_non_adapter_params", True)
    args.unfreeze_layer_norms = getattr(args, "unfreeze_layer_norms", False)
    args.adapter_modules = getattr(args, "adapter_modules", True)
    args.adapter_layer_norm_per_adaper = getattr(args, "adapter_layer_norm_per_adaper", False)
    args.adapter_layer_norm = getattr(args, "adapter_layer_norm", False)
    args.adapter_reuse_layer_norm = getattr(args, "adapter_reuse_layer_norm", True)
    args.ln_before_adapter = getattr(args, "ln_before_adapter", True)
    args.adapter_names = getattr(
        args,
        "adapter_names",
        ["[TAG=LAW]", "[TAG=IT]", "[TAG=SUB]", "[TAG=TALKS]", "[TAG=MED]", "[TAG=REL]"]
    )
    args.adapter_type = getattr(args, "adapter_type", "base")
    base_architecture(args)


@register_model_architecture("transformer_adapter", "transformer_adapter_base_ln")
def adapter_architecture_base_ln(args):
    # X-MOD + fine-tuning all layer norms.
    args.unfreeze_layer_norms = getattr(args, "unfreeze_layer_norms", True)
    adapter_architecture_base(args)


@register_model_architecture("transformer_adapter", "transformer_adapter_bapna_and_firat")
def adapter_base_architecture_bapna_and_firat(args):
    # Default adapter setup.
    # https://aclanthology.org/D19-1165.pdf
    args.adapter_layer_norm_per_adapter = getattr(args, "adapter_layer_norm_per_adapter", True)
    args.adapter_reuse_layer_norm = getattr(args, "adapter_reuse_layer_norm", False)
    args.ln_before_adapter = getattr(args, "ln_before_adapter", False)
    adapter_architecture_base(args)


@register_model_architecture("transformer_adapter", "transformer_gated_adapters_mean")
def gated_adapters_base_architecture_mean(args):
    args.adapter_type = getattr(args, "adapter_type", "gated")
    args.noise_tunable = getattr(args, "noise_tunable", False)
    args.aggregation_func = getattr(args, "aggregation_func", "avg")
    args.adapter_schedule_type = getattr(args, "adapter_schedule_type", "const")
    args.const_adapter_temperature = getattr(args, "const_adapter_temperature", 2.0)
    args.loss_temperature = 0.1
    args.shared_gating = getattr(args, "shared_gating", False)
    adapter_base_architecture_bapna_and_firat(args)
