import gin
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer.transformer import \
    Unitransformer, Bitransformer, get_vocab_embedding_cls, make_layer_stack

from reward.comparative.loss_fn import comparative_paired_rewards_loss

def get_dims_by_name(tensor, dim_name):
    return [d for d in tensor.shape.dims if d.name == dim_name]

class ScalarOutputUnitransformer(Unitransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_vocab_size = None
        self.autoregressive = False
        self._reward_dim = mtf.Dimension("reward", 1)

    def _call_internal(self, context, inputs, targets=None):
        """
        Overrrides Unitransformer._call_internal
        Adds losses to context!
        """
        mesh = inputs.mesh
        if self.ensemble_dim and self.ensemble_dim not in inputs.shape.dims:
            # Training an ensemble where all models are trained on the same examples.
            inputs = mtf.broadcast(inputs, [self.ensemble_dim] + inputs.shape.dims)
            if targets:
                targets = mtf.broadcast(
                    targets, [self.ensemble_dim] + targets.shape.dims)
        if "embedding" in context.shared_params:
            vocab_embedding = context.shared_params["embedding"]
        else:
            vocab_embedding = get_vocab_embedding_cls()(
                mesh,
                self.input_vocab_dim,
                self.model_dim,
                context.variable_dtype,
                name="embedding",
                ensemble_dim=self.ensemble_dim)
        x = vocab_embedding.ids_to_embedding(inputs)
        if self.positional_embedding:
            if "positional_embedding" in context.shared_params:
                pos_emb_var = context.shared_params["positional_embedding"]
            else:
                pos_emb_var = mtf.layers.embedding_weights(
                    mesh, self.max_length_dim, self.model_dim, context.variable_dtype,
                    "positional_embedding", ensemble_dim=self.ensemble_dim)
            if (context.length_dim is not None and
                context.length_dim.size > self.max_length_dim.size):
                message = (
                    "Length dimenison exceeds size of positional embedding table. "
                    "length_dim.size > max_length_dim.size %s vs %s."
                    % (context.length_dim, self.max_length_dim))
                if context.position_is_default:
                    # Definitely getting overflow in this case.
                    raise ValueError(message)
                else:
                    tf.logging.warning(
                        message +
                        " This may be OK if there are several shorter sequences packed "
                        "together.  Otherwise, the later positions will get zeros.")
            if context.position_is_default:
                pos_emb = mtf.rename_dimension(
                    mtf.slice(pos_emb_var, 0, context.length_dim.size,
                            self.max_length_dim.name),
                    self.max_length_dim.name, context.length_dim.name)
            else:
                pos_emb = mtf.gather(
                    pos_emb_var, context.position, self.max_length_dim,
                    output_shape=x.shape)
            x += pos_emb
        x = self.layer_stack.call(context, x)                                                                                                                                                                                                                                                                                        
        rewards = mtf.layers.dense(
            x,
            new_dims=self._reward_dim,
            reduced_dims=get_dims_by_name(x, "d_model"),
            use_bias=False,
            variable_dtype=context.variable_dtype,
            name="reward_head"
        )
        # Squeeze out size 1 reward dimension
        squeezed_shape = mtf.Shape([d for d in rewards.shape.dims if d.name != "reward"])
        rewards = mtf.reshape(rewards, squeezed_shape, name="squeeze_reward_dim")
        # Keep reward only at EOS positions
        targets_length_dim = get_dims_by_name(
            tensor=context.sequence_id,
            dim_name="targets_length"
        )[0]
        shifted_segmentation = mtf.shift(
            context.sequence_id,
            offset=-1,
            dim=targets_length_dim,
            wrap=False
        )
        is_eos = mtf.not_equal(context.sequence_id, shifted_segmentation)
        eos_rewards_long = mtf.cast(is_eos, dtype=rewards.dtype) * rewards
        eos_rewards = mtf.reduce_sum(
            eos_rewards_long,
            reduced_dim=targets_length_dim
        )
        ans_pair_dims = get_dims_by_name(rewards, "ans_pair")
        if ans_pair_dims and context.losses is not None:
            loss = comparative_paired_rewards_loss(
                paired_rewards=eos_rewards,
                ans_pair_dim=ans_pair_dims[0],
                batch_dim=get_dims_by_name(rewards, "batch")[0]
            )
            context.losses.append(loss)
        return eos_rewards

@gin.configurable
def make_reward_bitransformer(
    input_vocab_size=gin.REQUIRED,
    output_vocab_size=gin.REQUIRED,
    layout=None,
    mesh_shape=None,
    encoder_name="encoder",
    decoder_name="decoder"
    ):
    with gin.config_scope("encoder"):
        encoder = Unitransformer(
            layer_stack=make_layer_stack(),
            input_vocab_size=input_vocab_size,
            output_vocab_size=None,
            autoregressive=False,
            name=encoder_name,
            layout=layout,
            mesh_shape=mesh_shape
        )
    with gin.config_scope("decoder"):
        decoder = ScalarOutputUnitransformer(
        layer_stack=make_layer_stack(),
        input_vocab_size=output_vocab_size,
        output_vocab_size=None,
        autoregressive=False,
        name=decoder_name,
        layout=layout,
        mesh_shape=mesh_shape
    )
    return Bitransformer(encoder, decoder)