"""Test model modules."""

import pytest
import torch
from dreem.io import Frame, Instance
from dreem.models.mlp import MLP
from dreem.models.attention_head import ATTWeightHead
from dreem.models import (
    Embedding,
    VisualEncoder,
    Transformer,
    GlobalTrackingTransformer,
)
from dreem.models.transformer import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)


# todo: add named tensor tests
def test_mlp():
    """Test MLP logic."""
    b, n, f = 1, 10, 1024  # batch size, num instances, features

    mlp = MLP(input_dim=f, hidden_dim=f, output_dim=f, num_layers=2, dropout=0.1)

    output_tensor = mlp(torch.rand(size=(b, n, f)))

    assert output_tensor.shape == (b, n, f)


def test_att_weight_head():
    """Test self-attention head logic."""
    b, n, f = 1, 10, 1024  # batch size, num instances, features

    att_weight_head = ATTWeightHead(feature_dim=f, num_layers=2, dropout=0.1)

    q = k = torch.rand(size=(b, n, f))

    attn_weights = att_weight_head(q, k)

    assert attn_weights.shape == (b, n, n)


def test_encoder_timm():
    """Test feature extractor logic using timm backend."""
    b, c, h, w = 1, 1, 100, 100  # batch size, channels, height, width

    features = 512
    input_tensor = torch.rand(b, c, h, w)
    backend = "timm"

    encoder = VisualEncoder(
        model_name="off", in_chans=c, d_model=features, backend=backend
    )
    output = encoder(input_tensor)

    assert output.shape == (b, features)
    assert not output.any()

    encoder = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=features, backend=backend
    )

    assert not isinstance(encoder.feature_extractor, torch.nn.Sequential)

    output = encoder(input_tensor)

    assert output.shape == (b, features)

    c = 3
    input_tensor = torch.rand(b, c, h, w)

    features = 128

    encoder = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=features, backend=backend
    )

    output = encoder(input_tensor)

    assert output.shape == (b, features)

    c = 9
    input_tensor = torch.rand(b, c, h, w)

    encoder = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=features, backend=backend
    )

    output = encoder(input_tensor)

    assert output.shape == (b, features)


def test_encoder_torch():
    """Test feature extractor logic using torchvision backend."""
    b, c, h, w = 1, 1, 100, 100  # batch size, channels, height, width

    features = 512
    input_tensor = torch.rand(b, c, h, w)
    backend = "torch"

    encoder = VisualEncoder(
        model_name="off", in_chans=c, d_model=features, backend=backend
    )
    output = encoder(input_tensor)

    assert output.shape == (b, features)
    assert not output.any()

    encoder = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=features, backend=backend
    )

    assert isinstance(encoder.feature_extractor, torch.nn.Sequential)

    output = encoder(input_tensor)

    assert output.shape == (b, features)

    c = 3
    input_tensor = torch.rand(b, c, h, w)

    features = 128

    encoder = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=features, backend=backend
    )

    output = encoder(input_tensor)

    assert output.shape == (b, features)

    c = 9
    input_tensor = torch.rand(b, c, h, w)

    encoder = VisualEncoder(
        model_name="resnet18", in_chans=c, d_model=features, backend=backend
    )

    output = encoder(input_tensor)

    assert output.shape == (b, features)


def test_embedding_validity():
    """Test embedding usage."""

    # this would throw assertion since embedding should be "pos"
    with pytest.raises(Exception):
        _ = Embedding(emb_type="position", mode="learned", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="position", mode="fixed", features=128)

    with pytest.raises(Exception):
        _ = Embedding(emb_type="temporal", mode="learned", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="position", mode="fixed", features=128)

    with pytest.raises(Exception):
        _ = Embedding(emb_type="pos", mode="learn", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="temp", mode="learn", features=128)

    with pytest.raises(Exception):
        _ = Embedding(emb_type="pos", mode="fix", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="temp", mode="fix", features=128)

    with pytest.raises(Exception):
        _ = Embedding(emb_type="position", mode="learn", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="temporal", mode="learn", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="position", mode="fix", features=128)
    with pytest.raises(Exception):
        _ = Embedding(emb_type="temporal", mode="learn", features=128)

    _ = Embedding(emb_type="temp", mode="learned", features=128)
    _ = Embedding(emb_type="pos", mode="learned", features=128)

    _ = Embedding(emb_type="pos", mode="learned", features=128)


def test_embedding_basic():
    """Test embedding logic."""

    frames = 32
    objects = 10
    d_model = 256
    n_anchors = 1

    N = frames * objects

    boxes = torch.rand(size=(N, n_anchors, 4))
    times = torch.rand(size=(N,))

    pos_emb = Embedding(
        emb_type="pos",
        mode="fixed",
        features=d_model,
        temperature=objects,
        normalize=True,
        scale=10,
    )

    sine_pos_emb = pos_emb(boxes)

    pos_emb = Embedding(emb_type="pos", mode="learned", features=d_model, emb_num=100)
    learned_pos_emb = pos_emb(boxes)

    temp_emb = Embedding(emb_type="temp", mode="learned", features=d_model, emb_num=16)
    learned_temp_emb = temp_emb(times)

    pos_emb_off = Embedding(emb_type="pos", mode="off", features=d_model)
    off_pos_emb = pos_emb_off(boxes)

    temp_emb_off = Embedding(emb_type="temp", mode="off", features=d_model)
    off_temp_emb = temp_emb_off(times)

    learned_emb_off = Embedding(emb_type="off", mode="learned", features=d_model)
    off_learned_emb_boxes = learned_emb_off(boxes)
    off_learned_emb_times = learned_emb_off(times)

    fixed_emb_off = Embedding(emb_type="off", mode="fixed", features=d_model)
    off_fixed_emb_boxes = fixed_emb_off(boxes)
    off_fixed_emb_times = fixed_emb_off(times)

    off_emb = Embedding(emb_type="off", mode="off", features=d_model)
    off_emb_boxes = off_emb(boxes)
    off_emb_times = off_emb(times)

    assert sine_pos_emb.size() == (N, d_model)
    assert learned_pos_emb.size() == (N, d_model)
    assert learned_temp_emb.size() == (N, d_model)

    assert not torch.equal(sine_pos_emb, learned_pos_emb)
    assert not torch.equal(sine_pos_emb, learned_temp_emb)
    assert not torch.equal(learned_pos_emb, learned_temp_emb)

    assert off_pos_emb.size() == (N, d_model)
    assert off_temp_emb.size() == (N, d_model)
    assert off_learned_emb_boxes.size() == (N, d_model)
    assert off_learned_emb_times.size() == (N, d_model)
    assert off_fixed_emb_boxes.size() == (N, d_model)
    assert off_fixed_emb_times.size() == (N, d_model)
    assert off_emb_boxes.size() == (N, d_model)
    assert off_emb_times.size() == (N, d_model)

    assert not off_pos_emb.any()
    assert not off_temp_emb.any()
    assert not off_learned_emb_boxes.any()
    assert not off_learned_emb_times.any()
    assert not off_fixed_emb_boxes.any()
    assert not off_fixed_emb_times.any()
    assert not off_emb_boxes.any()
    assert not off_emb_times.any()


def test_embedding_kwargs():
    """Test embedding config logic."""

    frames = 32
    objects = 10

    N = frames * objects
    n_anchors = 1

    boxes = torch.rand(N, n_anchors, 4)

    # sine embedding

    sine_args = {
        "temperature": objects,
        "scale": frames,
        "normalize": True,
    }
    sine_no_args = Embedding("pos", "fixed", 128)
    sine_with_args = Embedding("pos", "fixed", 128, **sine_args)

    assert sine_no_args.temperature != sine_with_args.temperature

    sine_no_args = sine_no_args(boxes)
    sine_with_args = sine_with_args(boxes)

    assert not torch.equal(sine_no_args, sine_with_args)

    # learned pos embedding

    lp_no_args = Embedding("pos", "learned", 128)

    lp_args = {"emb_num": 100, "over_boxes": False}

    lp_with_args = Embedding("pos", "learned", 128, **lp_args)
    assert lp_no_args.lookup.weight.shape != lp_with_args.lookup.weight.shape

    # learned temp embedding

    lt_no_args = Embedding("temp", "learned", 128)

    lt_args = {"emb_num": 100}

    lt_with_args = Embedding("temp", "learned", 128, **lt_args)
    assert lt_no_args.lookup.weight.shape != lt_with_args.lookup.weight.shape


def test_multianchor_embedding():
    frames = 32
    objects = 10
    d_model = 256
    n_anchors = 10
    features = 256

    N = frames * objects

    boxes = torch.rand(size=(N, n_anchors, 4))

    fixed_emb = Embedding(
        "pos",
        "fixed",
        features=features,
        n_points=n_anchors,
        mlp_cfg={"num_layers": 3, "hidden_dim": 2 * d_model},
    )
    learned_emb = Embedding(
        "pos",
        "learned",
        features=features,
        n_points=n_anchors,
        mlp_cfg={"num_layers": 3, "hidden_dim": 2 * d_model},
    )
    assert not isinstance(fixed_emb.mlp, torch.nn.Identity)
    assert not isinstance(learned_emb.mlp, torch.nn.Identity)

    emb = fixed_emb(boxes)
    assert emb.size() == (N, features)

    emb = learned_emb(boxes)
    assert emb.size() == (N, features)

    fixed_emb = Embedding("pos", "fixed", features=features)
    learned_emb = Embedding("pos", "learned", features=features)
    with pytest.raises(RuntimeError):
        _ = fixed_emb(boxes)
    with pytest.raises(RuntimeError):
        _ = learned_emb(boxes)


def test_transformer_encoder():
    """Test transformer encoder layer logic."""
    feats = 256

    transformer_encoder = TransformerEncoderLayer(
        d_model=feats, nhead=1, dim_feedforward=feats, norm=True
    )

    N, B, D = 10, 1, feats

    # no position
    queries = torch.rand(size=(N, B, D))

    encoder_features = transformer_encoder(queries)

    assert encoder_features.size() == queries.size()

    # with position
    pos_emb = torch.ones_like(queries)

    encoder_features = transformer_encoder(queries, pos_emb=pos_emb)

    assert encoder_features.size() == encoder_features.size()


def test_transformer_decoder():
    """Test transformer decoder layer logic."""
    feats = 512

    transformer_decoder = TransformerDecoderLayer(
        d_model=feats,
        nhead=2,
        dim_feedforward=feats,
        dropout=0.2,
        norm=False,
        decoder_self_attn=True,
    )

    N, B, D = 10, 1, feats

    # no position
    decoder_queries = encoder_features = torch.rand(size=(N, B, D))

    decoder_features = transformer_decoder(decoder_queries, encoder_features)

    assert decoder_features.size() == decoder_queries.size()

    # with position
    pos_emb = query_pos_emb = torch.ones_like(encoder_features)

    decoder_features = transformer_decoder(
        decoder_queries,
        encoder_features,
        ref_pos_emb=pos_emb,
        query_pos_emb=query_pos_emb,
    )

    assert decoder_features.size() == decoder_queries.size()


def test_transformer_basic():
    """Test full transformer logic."""
    feats = 256
    num_frames = 32
    num_detected = 10
    img_shape = (1, 100, 100)

    transformer = Transformer(d_model=feats, num_encoder_layers=1, num_decoder_layers=1)

    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(
                Instance(
                    bbox=torch.rand(size=(1, 4)), features=torch.rand(size=(1, feats))
                )
            )
        frames.append(
            Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        )

    instances = [instance for frame in frames for instance in frame.instances]
    asso_preds = transformer(instances)

    assert asso_preds[0].matrix.size() == (num_detected * num_frames,) * 2


def test_transformer_embedding():
    """Test transformer using embedding."""
    feats = 256
    num_frames = 3
    num_detected = 10
    img_shape = (1, 50, 50)

    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(
                Instance(
                    bbox=torch.rand(size=(1, 4)), features=torch.rand(size=(1, feats))
                )
            )
        frames.append(Frame(video_id=0, frame_id=i, instances=instances))

    instances = [instance for frame in frames for instance in frame.instances]

    embedding_meta = {
        "pos": {"mode": "learned", "emb_num": 16, "normalize": True},
        "temp": {"mode": "learned", "emb_num": 16, "normalize": True},
    }

    transformer = Transformer(
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        embedding_meta=embedding_meta,
        return_embedding=True,
    )

    assert transformer.pos_emb.mode == "learned"
    assert transformer.temp_emb.mode == "learned"

    asso_preds = transformer(instances)

    assert asso_preds[0].matrix.size() == (num_detected * num_frames,) * 2

    pos_emb = torch.concat(
        [instance.get_embedding("pos") for instance in instances], axis=0
    )
    temp_emb = torch.concat(
        [instance.get_embedding("pos") for instance in instances], axis=0
    )

    assert pos_emb.size() == (
        len(instances),
        feats,
    ), pos_emb.shape

    assert temp_emb.size() == (
        len(instances),
        feats,
    ), temp_emb.shape


def test_tracking_transformer():
    """Test GTR logic."""
    feats = 512
    num_frames = 5
    num_detected = 20
    img_shape = (1, 128, 128)

    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(
                Instance(
                    bbox=torch.rand(size=(1, 4)), crop=torch.rand(size=(1, 1, 64, 64))
                )
            )
        frames.append(
            Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        )

    embedding_meta = {
        "pos": {
            "mode": "fixed",
            "temperature": num_detected,
            "scale": num_frames,
            "normalize": True,
        },
        "temp": None,
    }

    encoder_cfg = {"model_name": "resnet18", "pretrained": False, "in_chans": 3}

    tracking_transformer = GlobalTrackingTransformer(
        encoder_cfg=encoder_cfg,
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        embedding_meta=embedding_meta,
        return_embedding=True,
    )
    instances = [instance for frame in frames for instance in frame.instances]
    asso_preds = tracking_transformer(instances)

    assert asso_preds[0].matrix.size() == (num_detected * num_frames,) * 2

    pos_emb = torch.concat(
        [instance.get_embedding("pos") for instance in instances], axis=0
    )
    temp_emb = torch.concat(
        [instance.get_embedding("pos") for instance in instances], axis=0
    )

    assert pos_emb.size() == (
        len(instances),
        feats,
    ), pos_emb.shape

    assert temp_emb.size() == (
        len(instances),
        feats,
    ), temp_emb.shape
