"""A third reader: four folds fitted by someone else, on pixels cut a different way.

`tonylica/rsna2026-models` holds a four-fold DINOv2-small bundle at 224 px, 9 slices and a
160 mm crop under the legacy rules - dominant-axis slice order, corner-x laterality, slot
fallback, zero decode fill. Every one of those differs from what this pipeline's members
read, which is the point: it is wrong in different places.

It is not a better model. Its own manifest records fold holdouts of 0.748 to 0.792,
against 0.844 to 0.860 for the public members. Issue #29 measured what happens when a
weaker arm joins the vote across the board - 0.895 fell to 0.891 - so it does not join
across the board. The public 0.916 notebooks give it a vote on four findings only, and the
fractions below are theirs, converted from the member weights they express it in:

    4 legacy members against 20 public ones at weight w gives a final fraction
    f = 4w / (20 + 4w), so w = 15 is f = 0.75 and w = 2.5 is f = 1/3.

Three of the four are findings this pipeline is worst at (#35). That is what a second
reader is for, and it is why the weights are per target rather than one number.

Runs before the RadImageNet arm, because that arm's weights were fitted against a baseline
that already had this one in it.
"""

LEGACY_FILE = "rsna_20260807_v1.pt"

# The share of the final vote this bundle gets, per finding. Everything not named here
# gets none: on the other eight labels it is 0.05 to 0.10 behind the members, and a vote
# there is the dilution #29 measured rather than the diversity this is here for.
LEGACY_ALPHA = {"Lateral Meniscus": 0.75, "Lateral OA": 0.75,
                "Medial OA": 1.0 / 3.0, "Contusion": 0.50}

# A second decode of the test set at the bundle's contract, then four members over seven
# TTA windows. Less than the RadImageNet arm needs because there is no encoder to run
# over every slice, more than nothing because the decode is a decode.
LEGACY_NEEDS_S = 3600.0
LEGACY_RESERVE_S = 900.0
# What the RadImageNet arm needs after this one. Written here rather than read from that
# cell because this cell runs first and the name does not exist yet.
RAD_SHARE_S = 1.5 * 3600


def mounted_file(name):
    """The one mounted file with this name, or None if the dataset is not attached."""
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            return Path(root) / name
    return None


def legacy_predict(dev):
    """The bundle's four folds, rank-averaged, on the pixels they were fitted on."""
    b = torch.load(mounted_file(LEGACY_FILE), map_location="cpu", weights_only=False)
    folds = b.get("fold_states") or []
    if list(b.get("targets", TARGETS)) != TARGETS:
        raise WeightsError("the legacy bundle names different targets")
    if [tuple(s)[0] for s in b.get("slots", SLOTS)] != [s[0] for s in SLOTS]:
        raise WeightsError("the legacy bundle was fitted on different slots")
    if not folds:
        raise WeightsError("the legacy bundle carries no folds")

    group, n_group = int(b.get("group", 3)), int(b.get("n_group", 3))
    variant = str(b.get("model_variant", "dinov2-small")).split("-")[-1]
    # adopt_config_globals refuses a rule or a slot list it cannot reproduce, which is the
    # check that matters here: these weights predate fingerprints, so a wrong pixel path
    # would load, run, and quietly compute something else.
    adopt_config_globals({"img": int(b.get("img", 224)), "group": group,
                          "slices": group * n_group, "crop_mm": 160.0,
                          "band": [0.20, 0.80], "rules": dict(RULES_LEGACY),
                          "slots": [s[0] for s in SLOTS]})
    log(f"legacy bundle: {len(folds)} fold(s), {variant} at {IMG} px x {CACHE_SLICES} "
        f"slices, holdout {min(f.get('score', 0) for f in folds):.4f} to "
        f"{max(f.get('score', 0) for f in folds):.4f}")

    test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
    test_series = pd.read_csv(ROOT / "test_series.csv",
                              dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    plane = dict(zip(test_series.SeriesInstanceUID, test_series.Anatomical_Plane))
    hdr = annotate(walk("test_series"))
    st, cache, mask = build_cache(pick_slots(hdr, plane), plane,
                                  lat_of(hdr, "legacy "), "legacy")
    pos = {str(s): i for i, s in enumerate(st)}
    missing = [u for u in test.StudyInstanceUID if u not in pos]
    if missing:
        raise WeightsError(f"{len(missing)} test studies absent from the legacy cache")
    idx = np.asarray([pos[u] for u in test.StudyInstanceUID], np.int64)
    sex = sex_of(hdr, st, "legacy ")

    preds = []
    for f in folds:
        model = build_model(6, variant="base" if variant == "base" else "small",
                            pool="cls_mean_focal", prior=True, sex=False).to(dev)
        # These weights predate fingerprints, so the state dict is the only thing that
        # can say whether the architecture is the one they were fitted to. Loaded
        # strictly, and what does not line up is named before it raises.
        want, got = model.state_dict(), f["state_dict"]
        odd = [k for k in got if k not in want], [k for k in want if k not in got]
        if any(odd):
            log(f"  legacy fold {f.get('fold')}: {len(odd[0])} key(s) the model does "
                f"not have {odd[0][:4]}, {len(odd[1])} it wants and the bundle lacks "
                f"{odd[1][:4]}")
        model.load_state_dict(got, strict=True)
        p = predict_member(model, cache, mask, idx, dev, IMG, sex=sex)
        preds.append(pd.DataFrame(p).rank(pct=True).to_numpy())
        log(f"  legacy fold {f.get('fold')}: {len(idx)} studies")
        del model
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    del cache, mask
    gc.collect()
    return test["StudyInstanceUID"].astype(str).tolist(), np.mean(np.stack(preds), 0)


def legacy_blend(path="submission.csv"):
    """Give the bundle its four findings, or leave the members' submission alone."""
    sub = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if mounted_file(LEGACY_FILE) is None:
        log("legacy bundle: not attached; the members' submission stands")
        return sub
    left = 9.0 * 3600 - (time.time() - T0) - LEGACY_RESERVE_S - RAD_SHARE_S
    if left < LEGACY_NEEDS_S:
        log(f"legacy bundle: {left / 60:.0f} min left after reserving the RadImageNet "
            f"arm's share, needs {LEGACY_NEEDS_S / 60:.0f}; skipped")
        return sub
    keep = sub.copy()
    try:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ids, leg = legacy_predict(dev)
        if ids != sub["StudyInstanceUID"].astype(str).tolist():
            raise WeightsError("the bundle and the submission disagree on study order")
        base = pd.DataFrame(sub[TARGETS].to_numpy(np.float64)).rank(pct=True).to_numpy()
        out = base.copy()
        for j, t in enumerate(TARGETS):
            a = LEGACY_ALPHA.get(t, 0.0)
            out[:, j] = (1.0 - a) * base[:, j] + a * leg[:, j]
        sub[TARGETS] = out
        if not np.isfinite(sub[TARGETS].to_numpy()).all():
            raise WeightsError("the blended submission is not finite")
        sub.to_csv(path, index=False)
        log(f"legacy bundle: {sorted(LEGACY_ALPHA)} blended, the other "
            f"{len(TARGETS) - len(LEGACY_ALPHA)} left on the members alone")
    except Exception as exc:
        log(f"legacy bundle skipped ({type(exc).__name__}: {exc}); "
            f"the members' submission stands")
        keep.to_csv(path, index=False)
        return keep
    return sub


try:
    legacy_blend()
except Exception as exc:      # the members' submission is already written; keep it
    log(f"legacy bundle: {type(exc).__name__}: {exc}")
