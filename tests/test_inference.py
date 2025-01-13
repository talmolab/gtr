"""Test inference logic."""

import torch
import pytest
import numpy as np
from pytorch_lightning import Trainer
from omegaconf import OmegaConf, DictConfig
from dreem.io import Frame, Instance, Config
from dreem.models import GTRRunner, GlobalTrackingTransformer
from dreem.inference import Tracker, post_processing, metrics
from dreem.inference.track_queue import TrackQueue
from dreem.inference.track import run


def test_track_queue():
    window_size = 8
    max_gap = 10
    img_shape = (3, 1024, 1024)
    n_instances_per_frame = [2] * window_size

    frames = []
    instances_per_frame = []

    tq = TrackQueue(window_size, max_gap)
    for i in range(window_size):
        instances = []
        for j in range(n_instances_per_frame[i]):
            instances.append(Instance(gt_track_id=j, pred_track_id=j))
        instances_per_frame.append(instances)
        frame = Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        frames.append(frame)

        tq.add_frame(frame)

    assert len(tq) == sum(n_instances_per_frame[1:])
    assert tq.n_tracks == max(n_instances_per_frame)
    assert tq.tracks == [i for i in range(max(n_instances_per_frame))]
    assert len(tq.collate_tracks()) == window_size - 1
    assert all([gap == 0 for gap in tq._curr_gap.values()])
    assert tq.curr_track == max(n_instances_per_frame) - 1

    tq.add_frame(
        Frame(
            video_id=0,
            frame_id=window_size + 1,
            img_shape=img_shape,
            instances=[Instance(gt_track_id=0, pred_track_id=0)],
        )
    )

    assert len(tq._queues[0]) == window_size - 1
    assert tq._curr_gap[0] == 0
    assert tq._curr_gap[max(n_instances_per_frame) - 1] == 1

    tq.add_frame(
        Frame(
            video_id=0,
            frame_id=window_size + 1,
            img_shape=img_shape,
            instances=[
                Instance(gt_track_id=1, pred_track_id=1),
                Instance(
                    gt_track_id=max(n_instances_per_frame),
                    pred_track_id=max(n_instances_per_frame),
                ),
            ],
        )
    )

    assert len(tq._queues[max(n_instances_per_frame)]) == 1
    assert tq._curr_gap[1] == 0
    assert tq._curr_gap[0] == 1

    for i in range(max_gap):
        tq.add_frame(
            Frame(
                video_id=0,
                frame_id=window_size + i + 1,
                img_shape=img_shape,
                instances=[Instance(gt_track_id=0, pred_track_id=0)],
            )
        )

    assert tq.n_tracks == 1
    assert tq.curr_track == max(n_instances_per_frame)
    assert 0 in tq._queues.keys()

    tq.end_tracks()

    assert len(tq) == 0


def test_tracker():
    """Test tracker module.

    Tests that tracker works with/without post processing
    """
    feats = 512
    num_frames = 5
    num_detected = 2
    img_shape = (1, 128, 128)
    test_frame = 1
    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(
                Instance(
                    gt_track_id=j,
                    pred_track_id=-1,
                    bbox=torch.rand(size=(1, 4)),
                    crop=torch.rand(size=(1, 1, 64, 64)),
                )
            )
        frames.append(
            Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        )

    embedding_meta = {
        "embedding_type": "fixed_pos",
        "kwargs": {"temperature": num_detected, "scale": num_frames, "normalize": True},
    }

    tracking_transformer = GlobalTrackingTransformer(
        encoder_cfg={"model_name": "resnet18", "pretrained": False, "in_chans": 3},
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        embedding_meta=embedding_meta,
        return_embedding=False,
    )

    tracking_cfg = {
        "overlap_thresh": 0.01,
        "decay_time": None,
        "iou": None,
        "max_center_dist": None,
    }

    tracker = Tracker(**tracking_cfg)

    frames_pred = tracker(tracking_transformer, frames)

    # TODO: Debug saving asso matrices
    # asso_equals = (
    #     frames_pred[test_frame].get_traj_score("decay_time").to_numpy()
    #     == frames_pred[test_frame].get_traj_score("final").to_numpy()
    # ).all()
    # assert asso_equals

    assert len(frames_pred[test_frame].get_pred_track_ids()) == num_detected


# @pytest.mark.parametrize("set_default_device", ["cpu"], indirect=True)
def test_post_processing():  # set_default_device
    """Test postprocessing methods.

    Tests each postprocessing method to ensure that
    it filters only when the condition is satisfied.
    """
    T = 8
    k = 5
    D = 512
    M = 5
    N_t = 5
    N_p = N_t * (T - 1)
    N = N_t * T

    reid_features = torch.rand((1, N_t, D))
    asso_nonk = torch.rand((N_t, N_p))

    decay_time = 0

    assert (
        asso_nonk
        == post_processing.weight_decay_time(asso_nonk, decay_time, reid_features, T, k)
    ).all()

    decay_time = 0.9

    assert not (
        asso_nonk
        == post_processing.weight_decay_time(asso_nonk, decay_time, reid_features, T, k)
    ).all()

    asso_output = torch.rand((N_t, M))
    ious = torch.rand((N_t, M))

    assert (asso_output == post_processing.weight_iou(asso_output, None, ious)).all()

    assert not (
        asso_output == post_processing.weight_iou(asso_output, "mult", ious)
    ).all()

    assert not (
        asso_output == post_processing.weight_iou(asso_output, "max", ious)
    ).all()

    assert not (
        post_processing.weight_iou(asso_output, "mult", ious)
        == post_processing.weight_iou(asso_output, "max", ious)
    ).all()

    im_size = 128
    k_boxes = torch.rand((N_t, 1, 4)) * im_size
    nonk_boxes = torch.rand((N_p, 1, 4)) * im_size
    id_inds = torch.tile(torch.cat((torch.zeros(M - 1), torch.ones(1))), (N_p, 1))

    assert (
        asso_output
        == post_processing.filter_max_center_dist(
            asso_output=asso_output,
            max_center_dist=0,
            id_inds=id_inds,
            curr_frame_boxes=k_boxes,
            prev_frame_boxes=nonk_boxes,
        )
    ).all()

    assert not (
        asso_output
        == post_processing.filter_max_center_dist(
            asso_output=asso_output,
            max_center_dist=1e-9,
            id_inds=id_inds,
            curr_frame_boxes=k_boxes,
            prev_frame_boxes=nonk_boxes,
        )
    ).all()


def test_metrics():
    """Test basic GTR Runner."""
    num_frames = 3
    num_detected = 3
    n_batches = 1
    batches = []

    for i in range(n_batches):
        frames_pred = []
        for j in range(num_frames):
            instances_pred = []
            for k in range(num_detected):
                bboxes = torch.tensor(np.random.uniform(size=(num_detected, 4)))
                bboxes[:, -2:] += 1
                instances_pred.append(
                    Instance(gt_track_id=k, pred_track_id=k, bbox=torch.randn((1, 4)))
                )
            frames_pred.append(Frame(video_id=0, frame_id=j, instances=instances_pred))
        batches.append(frames_pred)

    for batch in batches:
        instances_mm = metrics.to_track_eval(batch)
        clear_mot = metrics.get_pymotmetrics(instances_mm)

        matches, indices, _ = metrics.get_matches(batch)

        switches = metrics.get_switches(matches, indices)

        sw_cnt = metrics.get_switch_count(switches)

        assert sw_cnt == clear_mot["num_switches"] == 0, (
            sw_cnt,
            clear_mot["num_switches"],
        )


def get_ckpt(ckpt_path: str):
    """Save GTR Runner to checkpoint file."""

    class DummyDataset(torch.utils.data.Dataset):

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return None

    dl = torch.utils.data.DataLoader(DummyDataset())
    model = GTRRunner()
    trainer = Trainer(max_steps=1, min_steps=1)
    trainer.fit(model, dl)
    trainer.save_checkpoint(ckpt_path)

    return ckpt_path


def test_track(tmp_path, inference_config):
    ckpt_path = tmp_path / "model.ckpt"
    get_ckpt(ckpt_path)

    out_dir = tmp_path / "preds"
    out_dir.mkdir()

    inference_cfg = OmegaConf.load(inference_config)

    cfg = Config(inference_cfg)

    cfg.set_hparams({"ckpt_path": ckpt_path, "outdir": out_dir})

    run(cfg.cfg)
    assert len(list(out_dir.iterdir())) == len(cfg.cfg.dataset.test_dataset.video_files)
