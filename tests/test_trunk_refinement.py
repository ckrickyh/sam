import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def trunk_refinement():
    """Load the image-processing module without requiring the SAM 3 model package."""
    project_root = Path(__file__).resolve().parents[1]
    module_path = (
        project_root
        / "samplingAnalysis"
        / "adaptive_cielab_otsu_foliage_extractor_fixBranchlet_order_maxGapRatio_ExG_box.py"
    )

    sam3 = types.ModuleType("sam3")
    sam3.__path__ = []
    sam3_model = types.ModuleType("sam3.model")
    sam3_model.__path__ = []
    image_processor = types.ModuleType("sam3.model.sam3_image_processor")
    image_processor.Sam3Processor = type("Sam3Processor", (), {})
    model_builder = types.ModuleType("sam3.model_builder")
    model_builder.build_sam3_image_model = lambda *args, **kwargs: None

    patcher = pytest.MonkeyPatch()
    patcher.setitem(sys.modules, "sam3", sam3)
    patcher.setitem(sys.modules, "sam3.model", sam3_model)
    patcher.setitem(sys.modules, "sam3.model.sam3_image_processor", image_processor)
    patcher.setitem(sys.modules, "sam3.model_builder", model_builder)

    spec = importlib.util.spec_from_file_location("_trunk_refinement_test_target", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    patcher.undo()


def _neutral_image(shape: tuple[int, int]) -> np.ndarray:
    return np.full((*shape, 3), 128, dtype=np.uint8)


def test_repairs_a_small_enclosed_hole_in_the_trunk(trunk_refinement):
    shape = (64, 64)
    raw_trunk = np.zeros(shape, dtype=bool)
    raw_trunk[16:48, 16:48] = True
    hole = (slice(30, 33), slice(30, 33))
    raw_trunk[hole] = False
    refinement_stats = {}

    smooth_trunk, wood_mask = trunk_refinement.heal_and_refine_wood_trunk(
        raw_trunk,
        _neutral_image(shape),
        np.ones(shape, dtype=bool),
        refinement_stats=refinement_stats,
    )

    assert smooth_trunk[hole].all()
    assert wood_mask[hole].all()
    assert refinement_stats == {
        "trunk_holes_filled_count": 1,
        "trunk_hole_pixels_filled": 9,
    }


def test_hole_cutoff_scales_with_trunk_mask_area(trunk_refinement):
    results = []
    for scale in (1, 2):
        shape = (64 * scale, 64 * scale)
        raw_trunk = np.zeros(shape, dtype=bool)
        raw_trunk[16 * scale:48 * scale, 16 * scale:48 * scale] = True
        hole = (
            slice(30 * scale, 33 * scale),
            slice(30 * scale, 33 * scale),
        )
        raw_trunk[hole] = False
        refinement_stats = {}

        smooth_trunk, _ = trunk_refinement.heal_and_refine_wood_trunk(
            raw_trunk,
            _neutral_image(shape),
            np.ones(shape, dtype=bool),
            refinement_stats=refinement_stats,
        )

        assert smooth_trunk[hole].all()
        results.append(refinement_stats)

    assert [result["trunk_holes_filled_count"] for result in results] == [1, 1]
    assert [result["trunk_hole_pixels_filled"] for result in results] == [9, 36]


def test_does_not_fill_an_enclosed_hole_over_two_percent_of_trunk(trunk_refinement):
    shape = (64, 64)
    raw_trunk = np.zeros(shape, dtype=bool)
    raw_trunk[16:48, 16:48] = True
    hole = (slice(24, 40), slice(24, 40))
    raw_trunk[hole] = False
    refinement_stats = {}

    smooth_trunk, _ = trunk_refinement.heal_and_refine_wood_trunk(
        raw_trunk,
        _neutral_image(shape),
        np.ones(shape, dtype=bool),
        refinement_stats=refinement_stats,
    )

    assert not smooth_trunk[32, 32]
    assert refinement_stats == {
        "trunk_holes_filled_count": 0,
        "trunk_hole_pixels_filled": 0,
    }


def test_does_not_add_a_separate_non_green_canopy_component(trunk_refinement):
    shape = (64, 64)
    raw_trunk = np.zeros(shape, dtype=bool)
    raw_trunk[32:52, 32:52] = True

    green_mask = np.zeros(shape, dtype=bool)
    green_mask[6:13, 6:13] = True
    green_mask[8:11, 8:11] = False
    separate_component = (slice(8, 11), slice(8, 11))

    _, wood_mask = trunk_refinement.heal_and_refine_wood_trunk(
        raw_trunk,
        _neutral_image(shape),
        np.ones(shape, dtype=bool),
        green_mask=green_mask,
    )

    assert not wood_mask[separate_component].any()


def test_green_pixels_are_excluded_from_the_wood_mask(trunk_refinement):
    shape = (64, 64)
    raw_trunk = np.zeros(shape, dtype=bool)
    raw_trunk[16:48, 16:48] = True

    green_mask = np.zeros(shape, dtype=bool)
    green_pixels = (slice(6, 10), slice(6, 10))
    green_mask[green_pixels] = True

    smooth_trunk, wood_mask = trunk_refinement.heal_and_refine_wood_trunk(
        raw_trunk,
        _neutral_image(shape),
        np.ones(shape, dtype=bool),
        green_mask=green_mask,
    )

    assert not smooth_trunk[green_pixels].any()
    assert not wood_mask[green_pixels].any()


def test_sam_trunk_mask_takes_priority_over_green_color_mask(trunk_refinement):
    shape = (64, 64)
    raw_trunk = np.zeros(shape, dtype=bool)
    raw_trunk[16:48, 16:48] = True

    green_mask = np.zeros(shape, dtype=bool)
    trunk_pixels = (slice(28, 32), slice(28, 32))
    green_mask[trunk_pixels] = True

    smooth_trunk, wood_mask = trunk_refinement.heal_and_refine_wood_trunk(
        raw_trunk,
        _neutral_image(shape),
        np.ones(shape, dtype=bool),
        green_mask=green_mask,
    )

    assert smooth_trunk[trunk_pixels].all()
    assert wood_mask[trunk_pixels].all()
