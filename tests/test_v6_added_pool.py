"""CPU tests for `research/v6/added_pool.py`: class map, link rules, resize, hash audit. Synthetic data; no download, no results read."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from research.v4 import dedupe
from research.v6 import added_pool, head8


def _jpeg(width: int, height: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, (9, 12, 3), dtype=np.uint8)
    image = Image.fromarray(coarse).resize((width, height), Image.BICUBIC)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=95)
    return out.getvalue()


@pytest.mark.parametrize(("d2", "d3", "d4", "code"), [
    ("Malignant melanocytic proliferations (Melanoma)", "Melanoma in situ", "", "mel"),
    ("Malignant melanocytic proliferations (Melanoma)", "Melanoma metastasis", "", None),
    ("Malignant adnexal epithelial proliferations - Follicular", "Basal cell carcinoma", "", "bcc"),
    ("Malignant epidermal proliferations", "Squamous cell carcinoma in situ", "", "scc"),
    ("Malignant epidermal proliferations", "Squamous cell carcinoma, NOS", "", "scc"),
    ("Malignant epidermal proliferations", "Squamous cell carcinoma, Invasive", "Keratoacanthoma", "scc"),
    ("Benign epidermal proliferations", "Keratoacanthoma", "", "scc"),
    ("Indeterminate epidermal proliferations", "Solar or actinic keratosis", "", "akiec_ak"),
    ("Benign melanocytic proliferations", "Nevus", "", "nv"),
    ("Benign epidermal proliferations", "Solar lentigo", "", "bkl"),
    ("Benign soft tissue proliferations - Fibro-histiocytic", "Dermatofibroma", "", "df"),
    ("Benign soft tissue proliferations - Vascular", "Hemangioma", "", "vasc"),
    ("Malignant soft tissue proliferations - Vascular", "Kaposi sarcoma", "", None),
    ("Flat melanotic pigmentations - not melanocytic nevus", "Scar", "", None),
    ("", "", "", None),
])
def test_class_map_declares_every_row_or_excludes_it(d2: str, d3: str, d4: str, code: str | None) -> None:
    assert added_pool.map_class_8({"diagnosis_2": d2, "diagnosis_3": d3, "diagnosis_4": d4}) == code


def test_a_missing_diagnosis_is_excluded_not_read_as_the_string_nan() -> None:
    row = {"diagnosis_2": float("nan"), "diagnosis_3": float("nan"), "diagnosis_4": float("nan")}
    assert added_pool.map_class_8(row) is None


def test_every_mapped_class_is_on_the_8_class_head() -> None:
    assert {"mel", "bcc", "scc", "akiec_ak", "nv", "bkl", "df", "vasc"} == set(head8.CLASSES_8)


@pytest.mark.parametrize(("age", "band"), [(35, "<40"), (39.9, "<40"), (40, "40-59"), (55, "40-59"), (60, "60+"), (None, "unknown"), (float("nan"), "unknown")])
def test_age_bands_match_the_manifest_rule(age: float | None, band: str) -> None:
    assert added_pool.age_band(age) == band


GUARDED = {"manifest_images": {"M"}, "manifest_lesions": {"ML"}, "manifest_patients": {"MP"}, "hiba_images": {"H"}, "hiba_lesions": {"HL"},
           "hiba_patients": {"HP"}, "extras_images": {"E"}}


def _row(**changes) -> pd.Series:
    base = {"isic_id": "A", "lesion_id": "AL", "patient_id": "AP", "image_type": "dermoscopic", "diagnosis_2": "Benign melanocytic proliferations",
            "diagnosis_3": "Nevus", "diagnosis_4": "", "attribution": "Hospital Clinic de Barcelona"}
    return pd.Series({**base, **changes})


@pytest.mark.parametrize(("changes", "source", "reason"), [
    ({}, "isic2020", None),
    ({"image_type": "clinical: close-up"}, "milk10k", "not_dermoscopic"),
    ({"attribution": "MILK study team"}, "isic2019_test", "isic2019_test_not_the_bcn_part"),
    ({"attribution": "MILK study team"}, "milk10k", None),
    ({"isic_id": "M"}, "isic2020", "linked_to_manifest"),
    ({"lesion_id": "ML"}, "isic2020", "linked_to_manifest"),
    ({"patient_id": "MP"}, "isic2020", "linked_to_manifest"),
    ({"isic_id": "H"}, "isic2020", "linked_to_hiba"),
    ({"lesion_id": "HL"}, "isic2020", "linked_to_hiba"),
    ({"patient_id": "HP"}, "derm12345", "linked_to_hiba"),
    ({"lesion_id": "D6"}, "milk10k", "milk10k_d6"),
    ({"isic_id": "E"}, "isic2020", "already_in_young_extras"),
    ({"diagnosis_2": "", "diagnosis_3": "Scar"}, "isic2019_test", "no_class_on_the_head"),
])
def test_exclusion_rules(changes: dict, source: str, reason: str | None) -> None:
    assert added_pool.exclusion_reason(_row(**changes), source, GUARDED, {"D6"}) == reason


def test_a_lesion_shared_only_with_the_young_extras_stays() -> None:
    assert added_pool.exclusion_reason(_row(lesion_id="in_extras_only"), "isic2020", GUARDED, set()) is None


def test_an_empty_identifier_is_not_a_link() -> None:
    guarded = {k: v - {""} for k, v in {**GUARDED, "manifest_lesions": {"", "ML"}}.items()}
    assert added_pool.exclusion_reason(_row(lesion_id=""), "derm12345", guarded, set()) is None


def _size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        return image.size


def test_resize_sets_the_shorter_side_and_never_enlarges() -> None:
    small, width, height = added_pool.resized_jpeg(_jpeg(1200, 800))
    assert (width, height) == (1200, 800) and _size(small) == (768, 512)
    tall, _, _ = added_pool.resized_jpeg(_jpeg(800, 1600))
    assert _size(tall) == (512, 1024)
    kept, _, _ = added_pool.resized_jpeg(_jpeg(400, 300))
    assert _size(kept) == (400, 300)


def test_the_cache_is_big_enough_for_the_384_member_without_upsampling() -> None:
    assert added_pool.SHORT_SIDE >= round(384 * 256 / 224)
    assert added_pool.SHORT_SIDE * 0.8 ** 0.5 >= 384  # the tightest training crop: scale 0.8


def test_hash_bytes_equals_the_corpus_hash_of_the_same_file(tmp_path) -> None:
    data = _jpeg(600, 450)
    path = tmp_path / "a.jpg"
    path.write_bytes(data)
    assert added_pool.hash_bytes(data) == dedupe.hash_one(str(path))
    assert added_pool.hash_bytes(b"not an image") == (0, 0)


def test_the_hash_survives_the_resize_so_a_copy_is_still_found() -> None:
    data = _jpeg(1600, 1200, seed=3)
    small, _, _ = added_pool.resized_jpeg(data)
    d1, p1 = added_pool.hash_bytes(data)
    d2, p2 = added_pool.hash_bytes(small)
    assert added_pool.within(d1, p1, np.array([d2], dtype=np.uint64), np.array([p2], dtype=np.uint64))


def _hash_rows(hashes: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame({"dhash": [h[0] for h in hashes], "phash": [h[1] for h in hashes]}).astype({"dhash": np.uint64, "phash": np.uint64})


def test_hash_audit_needs_both_hashes_and_keeps_the_first_of_two_copies() -> None:
    a, b, c = (0xF0F0, 0xABCD), (0x0F0F00FF, 0x1234FF00), (0xFFFF0000FFFF, 0xFF00FF00FF00FF)
    near_a = (a[0] ^ 0b111, a[1] ^ 0b11)
    only_d = (a[0], 0xFFFFFFFFFFFF0000)
    target = (np.array([a[0]], dtype=np.uint64), np.array([a[1]], dtype=np.uint64))
    reasons = added_pool.hash_audit(_hash_rows([near_a, only_d, b, b, c]), {"corpus": target})
    assert reasons.tolist() == ["hash_copy_of_corpus", None, None, "hash_copy_of_earlier_added", None]


def test_a_copy_of_a_dropped_row_is_judged_against_the_target_not_the_row() -> None:
    a = (0xF0F0, 0xABCD)
    target = (np.array([a[0]], dtype=np.uint64), np.array([a[1]], dtype=np.uint64))
    reasons = added_pool.hash_audit(_hash_rows([a, a]), {"hiba": target})
    assert reasons.tolist() == ["hash_copy_of_hiba", "hash_copy_of_hiba"]


def test_built_rows_carry_what_the_trainer_requires_and_collapse_correctly() -> None:
    kept = pd.DataFrame({"isic_id": ["a", "b", "c"], "lesion_id": ["IL_1", None, None], "patient_id": ["IP_1", "IP_2", None],
                         "class_8": ["scc", "akiec_ak", "vasc"], "age_approx": [35, None, 70], "sex": ["m", "f", None],
                         "anatom_site_general": [None, None, None], "source": ["isic2020", "derm12345", "isic2019_test"]})
    rows = added_pool.build_rows(kept)
    assert rows["class_index_7"].tolist() == [0, 0, 6] and rows["class_index_8"].tolist() == [6, 0, 7]
    assert rows["class_7"].tolist() == ["akiec", "akiec", "vasc"] and rows["escalating_7"].tolist() == [True, True, False]
    assert rows["age_band"].tolist() == ["<40", "unknown", "60+"]
    assert rows["group_id"].tolist() == ["__v6__IL_1", "__v6__IP_2", "__v6__c"]
    assert rows["image_path"].iloc[0] == "data/external/v6_added/isic2020/a.jpg"
    from research.v6.train_v6 import ADDED_COLUMNS

    assert not rows[list(ADDED_COLUMNS)].isna().any().any()


def _reasons(values: pd.Series) -> list[str | None]:
    return [None if pd.isna(v) else v for v in values]


def test_a_copy_of_a_guarded_image_takes_its_lesion_and_patient_with_it() -> None:
    rows = pd.DataFrame({"lesion_id": ["L1", "L1", "L2", "L3", "L4", None], "patient_id": ["P1", "P9", "P1", "P3", "P4", None],
                         "hash_reason": ["hash_copy_of_corpus", None, None, None, "hash_copy_of_earlier_added", None]})
    assert _reasons(added_pool.linked_to_copies(rows)) == ["hash_copy_of_corpus", "same_lesion_id_as_a_hash_copy", "same_patient_id_as_a_hash_copy",
                                                          None, "hash_copy_of_earlier_added", None]


def test_a_copy_of_the_young_extras_does_not_spread_and_an_empty_id_is_not_a_link() -> None:
    rows = pd.DataFrame({"lesion_id": ["L1", "L1", None, None], "patient_id": ["P1", "P1", None, None],
                         "hash_reason": ["hash_copy_of_young_extras", None, "hash_copy_of_hiba", None]})
    assert _reasons(added_pool.linked_to_copies(rows)) == ["hash_copy_of_young_extras", None, "hash_copy_of_hiba", None]
