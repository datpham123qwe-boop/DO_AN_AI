import json
from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.datasets.augmentation import val_transform
from src.datasets.phenobench import PhenoBenchDataset
from src.evaluation.confusion_matrix import ConfusionMatrix, confusion_matrix
from src.evaluation.error_maps import colorize_error_map, make_error_map
from src.evaluation.inference import predict_logits
from src.evaluation.io import inference_protocol, load_model_bundle, prepare_output
from src.evaluation.pipeline import evaluate_dataset, select_failure_cases
from src.evaluation.segmentation_metrics import metrics_from_confusion_matrix
from src.evaluation.analysis import assert_comparable, choose_ce_experiment
from src.spraying.spray_mask import make_spray_mask
from src.spraying.spray_metrics import spray_metrics_from_confusion_matrix


def known_labels():
    expected = np.array([[5, 1, 2], [0, 3, 1], [2, 2, 4]], dtype=np.int64)
    truth, prediction = [], []
    for gt in range(3):
        for pred in range(3):
            truth += [gt] * int(expected[gt, pred])
            prediction += [pred] * int(expected[gt, pred])
    return np.array(prediction).reshape(4, 5), np.array(truth).reshape(4, 5), expected


def test_hand_calculated_metrics_and_asymmetric_orientation():
    pred, truth, expected = known_labels()
    np.testing.assert_array_equal(confusion_matrix(pred, truth), expected)
    metrics = metrics_from_confusion_matrix(expected)
    spray = spray_metrics_from_confusion_matrix(expected)
    assert metrics['per_class']['Background']['iou'] == 5 / 10
    assert metrics['crop_iou'] == 3 / 7
    assert metrics['weed_iou'] == 4 / 11
    assert metrics['miou'] == pytest.approx((5 / 10 + 3 / 7 + 4 / 11) / 3)
    assert metrics['weed_recall'] == spray['weed_coverage'] == 0.5
    assert metrics['per_class']['Weed']['precision'] == 4 / 7
    assert spray['crop_wrongly_sprayed_rate'] == 1 / 4
    assert spray['spray_area_ratio'] == 7 / 20
    assert spray['estimated_pixel_area_reduction'] == 13 / 20
    assert spray['missed_weed_pixels'] == 4


def test_accumulation_uses_counts_not_mean_of_image_recall():
    acc = ConfusionMatrix()
    # One missed weed versus nine covered weeds: recall is 9/10, not (0+1)/2.
    acc.update(np.zeros((1, 1), dtype=int), np.full((1, 1), 2))
    acc.update(np.full((3, 3), 2), np.full((3, 3), 2))
    assert metrics_from_confusion_matrix(acc.compute())['weed_recall'] == 0.9
    copied = acc.compute(); copied.fill(0)
    assert acc.compute().sum() == 10
    acc.reset()
    assert not acc.compute().any()


def test_ignore_pixels_excluded_from_every_spray_denominator():
    target = np.array([[0, 1, 2, 255]])
    pred = np.array([[2, 2, 2, 9]])  # Invalid prediction on ignored GT is irrelevant.
    matrix = confusion_matrix(pred, target)
    assert matrix.sum() == 3
    spray = spray_metrics_from_confusion_matrix(matrix)
    assert spray['spray_area_ratio'] == 1
    np.testing.assert_array_equal(make_spray_mask(pred, target), [[True, True, True, False]])


def test_undefined_metrics_are_json_null():
    matrix = np.zeros((3, 3), dtype=np.int64)
    metrics, spray = metrics_from_confusion_matrix(matrix), spray_metrics_from_confusion_matrix(matrix)
    assert metrics['miou'] is None and metrics['weed_recall'] is None
    assert spray['crop_wrongly_sprayed_rate'] is None
    assert spray['spray_area_ratio'] is None
    encoded = json.dumps({'metrics': metrics, 'spray': spray}, allow_nan=False)
    assert 'null' in encoded and 'NaN' not in encoded
    matrix[0, 0] = 100
    assert metrics_from_confusion_matrix(matrix)['miou'] is None


@pytest.mark.parametrize('pred,target', [
    (np.zeros((2, 2)), np.zeros((2, 2), dtype=int)),
    (np.zeros((2, 2), dtype=int), np.zeros((1, 2), dtype=int)),
    (np.array([[-1]]), np.array([[0]])),
    (np.array([[3]]), np.array([[0]])),
    (np.array([[0]]), np.array([[4]])),
])
def test_invalid_masks_fail(pred, target):
    with pytest.raises(ValueError): confusion_matrix(pred, target)


def test_safety_coverage_and_other_error_colors():
    target = np.array([[1, 2, 2, 0, 1, 255]])
    pred = np.array([[2, 0, 1, 1, 1, 2]])
    errors = make_error_map(pred, target)
    np.testing.assert_array_equal(errors, [[3, 2, 2, 1, 0, 255]])
    rgb = colorize_error_map(errors)
    assert rgb.shape == (1, 6, 3) and rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])
    np.testing.assert_array_equal(rgb[0, 1], [0, 120, 255])


class PointwiseModel(nn.Module):
    def forward(self, x): return x * 0.5 + 1


def test_tiled_logits_preserve_coordinates_and_padded_edges():
    images = torch.randn(2, 3, 35, 41)
    model = PointwiseModel().eval()
    full = predict_logits(model, images, amp=False)
    tiled = predict_logits(model, images, tile_size=16, amp=False)
    torch.testing.assert_close(full, images * 0.5 + 1, rtol=0, atol=0)
    torch.testing.assert_close(full, tiled, rtol=0, atol=0)


class BackgroundModel(nn.Module):
    def forward(self, x):
        logits = x.new_zeros((len(x), 3, *x.shape[-2:]))
        logits[:, 0] = 1
        return logits


@pytest.mark.parametrize('mode,valid_pixels', [('merge', 40), ('ignore', 24)])
def test_pipeline_exports_real_masks_json_figures_and_dataset_counts(tmp_path, mode, valid_pixels):
    root = tmp_path / 'dataset'
    for folder in ('images', 'semantics'): (root / 'val' / folder).mkdir(parents=True)
    raw = np.tile(np.arange(5, dtype=np.uint16), (4, 1))
    for name in ('a.png', 'b.png'):
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(root / 'val/images' / name)
        Image.fromarray(raw).save(root / 'val/semantics' / name)
    dataset = PhenoBenchDataset(root, 'val', val_transform(), mode)
    cfg = {'data': {'mask_mode': mode}, 'training': {'validation_tile_size': None}}
    protocol = inference_protocol(cfg, amp=False)
    output = prepare_output(tmp_path / 'report')
    report = evaluate_dataset(BackgroundModel(), dataset, output, torch.device('cpu'),
                              protocol, {'is_smoke_checkpoint': False}, cfg, 'E1', max_figures=1)
    assert report['segmentation']['valid_pixels'] == valid_pixels
    assert report['segmentation']['weed_recall'] == report['spraying']['weed_coverage'] == 0
    assert report['spraying']['crop_wrongly_sprayed_rate'] == 0
    assert report['spraying']['spray_area_ratio'] == 0
    assert not report['quality_flags']['partial_evaluation']
    assert not report['quality_flags']['official_split_size_matched']
    assert len(report['selected_cases']) == 1
    assert report['selected_cases'][0]['selection_reason'] == 'weed_coverage'
    assert (output / report['selected_cases'][0]['figure']).is_file()
    assert (output / 'figures/confusion_matrix.png').is_file()
    encoded = (output / 'metrics.json').read_text(encoding='utf-8')
    assert json.loads(encoded)['confusion_matrix'] == report['confusion_matrix']
    assert 'NaN' not in encoded
    with Image.open(output / 'spray_masks/a.png') as image: assert not np.array(image).any()
    with pytest.raises(FileExistsError): prepare_output(output)
    preview = evaluate_dataset(BackgroundModel(), dataset, tmp_path / 'preview', torch.device('cpu'),
                               protocol, {}, cfg, 'E1', limit=1, max_figures=0)
    assert preview['quality_flags']['partial_evaluation']
    assert preview['dataset']['evaluated_images'] == 1


def test_bundle_loads_embedded_config_and_blocks_smoke_or_wrong_architecture(tmp_path, monkeypatch):
    model = nn.Conv2d(3, 3, 1)
    monkeypatch.setattr('src.evaluation.io.build_model', lambda *a, **k: nn.Conv2d(3, 3, 1))
    checkpoint = {'model_state_dict': model.state_dict(), 'epoch': 3,
                  'metadata': {'model_name': 'deeplab', 'config': {
                      'model_name': 'deeplab', 'data': {'mask_mode': 'merge'},
                      'training': {'epochs': 20}, 'smoke_test': False}}}
    path = tmp_path / 'best.pt'; torch.save(checkpoint, path)
    (tmp_path / 'training_log.csv').write_text('epoch,train_loss,val_loss\n3,0.2,0.3\n')
    loaded, cfg, info = load_model_bundle(path, torch.device('cpu'))
    assert not loaded.training and info['epoch'] == 3 and len(info['sha256']) == 64
    assert info['training_budget_completed'] is False and info['logged_completed_epochs'] == 3
    import yaml
    wrong_config = tmp_path / 'wrong.yaml'
    wrong_config.write_text(yaml.safe_dump({'training': {'epochs': 3}}))
    with pytest.raises(ValueError, match='training.epochs'):
        load_model_bundle(path, torch.device('cpu'), config_path=wrong_config)
    with pytest.raises(ValueError, match='architecture'):
        load_model_bundle(path, torch.device('cpu'), model_name='segformer')
    checkpoint['metadata']['config']['smoke_test'] = True; torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='Smoke'):
        load_model_bundle(path, torch.device('cpu'))


def test_protocol_defaults_to_checkpoint_validation_geometry():
    cfg = {'training': {'validation_tile_size': 512}}
    assert inference_protocol(cfg)['tile_size'] == 512
    full = inference_protocol(cfg, 'full')
    assert full['tile_size'] is None and not full['matches_checkpoint_validation_geometry']


def comparable_report(model='deeplab', miou=0.5):
    return {'protocol': {**inference_protocol({'training': {}}, amp=False), 'amp_enabled': False},
            'dataset': {'split': 'val', 'evaluated_images': 772, 'filenames_sha256': 'same-files'},
            'class_names': ['Background', 'Crop', 'Weed'],
            'quality_flags': {'partial_evaluation': False, 'smoke_checkpoint': False,
                              'official_split_size_matched': True},
            'config': {'seed': 42, 'data': {'crop_size': 512}, 'loss': {'name': 'ce'},
                       'training': {'epochs': 20, 'batch_size': 4}},
            'checkpoint': {'model_name': model, 'training_budget_completed': True},
            'segmentation': {'miou': miou, 'weed_iou': 0.4}}


def test_ce_selection_uses_declared_rule_and_full_comparable_evaluations():
    first, second = comparable_report(), comparable_report('segformer', 0.6)
    assert choose_ce_experiment({'E1': first, 'E2': second}) == 'E2'
    second['quality_flags']['partial_evaluation'] = True
    with pytest.raises(ValueError, match='preview'):
        choose_ce_experiment({'E1': first, 'E2': second})


def test_ce_selection_waits_for_training_completion_evidence():
    first, second = comparable_report(), comparable_report('segformer')
    second['checkpoint']['training_budget_completed'] = None
    with pytest.raises(ValueError, match='training logs'):
        choose_ce_experiment({'E1': first, 'E2': second})


@pytest.mark.parametrize('change', ['tile', 'cohort', 'precision', 'mapping', 'epochs', 'batch'])
def test_comparison_refuses_inconsistent_protocols(change):
    first, second = comparable_report(), comparable_report('segformer')
    if change == 'tile': second['protocol'].update(mode='tiled', tile_size=512)
    if change == 'cohort': second['dataset']['filenames_sha256'] = 'different-images'
    if change == 'precision': second['protocol']['amp_enabled'] = True
    if change == 'mapping': second['protocol']['mask_mode'] = 'ignore'
    if change == 'epochs': second['config']['training']['epochs'] = 3
    if change == 'batch': second['config']['training']['batch_size'] = 1
    with pytest.raises(ValueError):
        assert_comparable({'E1': first, 'E2': second}, equal_epoch_budget=True, equal_training_protocol=True)
