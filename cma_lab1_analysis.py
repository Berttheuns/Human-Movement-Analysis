"""
CMA Lab 1 - 2D sagittale analyse van voet en enkel.

Assenstelsel: Y = looprichting (voorwaarts), Z = verticaal.  Alle hoeken worden
in het Y-Z-vlak (sagittaal vlak) berekend; de X-coordinaat wordt genegeerd.

Trials
    L1G1A02         : statische trial (rechtop staan)
    L1G1A03 - A05   : drie looptrials in +Y-richting

Uitvoer (map lab1_output/)
    events.csv              : IC, foot flat, heel off en FC van elke stance
    stance_values.csv       : hoekwaarden per stance
    <trial>_angles.csv      : gefilterde markerdata + hoeken per frame
    fig_oef1_contacten.png  : hiel- en teenhoogte met IC/FC
    fig_oef2_voet_vloer.png : voet-vloerhoek met IC/FC en rockers
    fig_oef3_enkel.png      : enkelhoek met IC/FC en rockers
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.signal import butter, filtfilt

# --------------------------------------------------------------------------
# Instellingen
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT.parent / "L1G1A.xlsx"
OUTPUT_DIR = ROOT / "lab1_output"

STANDING_SHEET = "L1G1A02"
TRIAL_SHEETS = ["L1G1A03", "L1G1A04", "L1G1A05"]
MARKERS = ["RKNE", "RANK", "RHEE", "RTOE"]

FS = 100.0            # samplefrequentie (Hz) - niet in de Excel vermeld, zie rapport
CUTOFF = 6.0          # Butterworth low-pass cutoff (Hz)
FC_SEARCH = 40        # FC wordt gezocht in de 40 frames na heel off
ROCKER_THR_DEG = 2.0  # foot flat / heel off: voet-vloerhoek binnen +-2 graden

COLORS = {"heel": "#1f77b4", "toe": "#ff7f0e", "ff": "#1f4e79", "ankle": "#8b1a1a"}
ROCKER_COLORS = ["#fde0c5", "#d4ecd0", "#e3d4ec"]
ROCKER_LABELS = ["1e rocker (heel)", "2e rocker (ankle)", "3e rocker (forefoot)"]


# --------------------------------------------------------------------------
# Data inlezen en filteren
# --------------------------------------------------------------------------
def read_trial(sheet: str) -> pd.DataFrame:
    """Leest frame + Y/Z van de vier markers en filtert ze (4e orde Butterworth, 6 Hz)."""
    raw = pd.read_excel(WORKBOOK, sheet_name=sheet, header=None)
    header = raw.iloc[0].ffill()
    data = raw.iloc[3:].apply(pd.to_numeric, errors="coerce").dropna(subset=[0])

    df = pd.DataFrame({"Frame": data[0].astype(int).to_numpy()})
    for marker in MARKERS:
        cols = [c for c in raw.columns if str(header[c]).endswith(marker)]
        y_col, z_col = cols[1], cols[2]          # kolommen X, Y, Z -> Y en Z nodig
        df[f"{marker}_Y"] = data[y_col].to_numpy(float)
        df[f"{marker}_Z"] = data[z_col].to_numpy(float)

    b, a = butter(4, CUTOFF / (FS / 2))
    for col in df.columns[1:]:
        df[col] = filtfilt(b, a, df[col].to_numpy())
    df["Time"] = (df["Frame"] - df["Frame"].iloc[0]) / FS
    return df


def segment_angle(df: pd.DataFrame, distal: str, proximal: str) -> np.ndarray:
    """Hoek (graden) van de vector distal->proximal t.o.v. de +Y-as in het Y-Z-vlak."""
    dy = df[f"{proximal}_Y"] - df[f"{distal}_Y"]
    dz = df[f"{proximal}_Z"] - df[f"{distal}_Z"]
    return np.degrees(np.arctan2(dz, dy)).to_numpy()


# --------------------------------------------------------------------------
# Statische referentie
# --------------------------------------------------------------------------
def static_reference(standing: pd.DataFrame) -> dict:
    """
    In stand nemen we aan dat (1) de zool plat op de grond ligt en (2) de enkel
    in neutrale stand staat (zool loodrecht op de tibia-as).  De hoeken die de
    marker-lijnen dan maken, zijn dus vaste offsets door de markerplaatsing.
    """
    foot = segment_angle(standing, "RHEE", "RTOE")
    tibia = segment_angle(standing, "RANK", "RKNE")
    return {
        "foot_offset": float(np.mean(foot)),     # HEE-TOE lijn t.o.v. zool
        "tibia_static": float(np.mean(tibia)),   # KNE-ANK lijn in stand
        "heel_z": float(standing["RHEE_Z"].mean()),
        "toe_z": float(standing["RTOE_Z"].mean()),
    }


def add_angles(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    # Oef 2: voet-vloerhoek. Positief = tenen omhoog (hielcontact),
    # negatief = hiel omhoog (push-off).
    df["FootFloor"] = segment_angle(df, "RHEE", "RTOE") - ref["foot_offset"]

    # Oef 3: enkelhoek. Hoek tussen tibia en zool; in neutrale stand 90 graden.
    tibia = segment_angle(df, "RANK", "RKNE")
    tibia_sole = tibia - df["FootFloor"]                      # hoek tibia-zool
    tibia_sole_neutral = ref["tibia_static"]                  # = "90 graden" in stand
    df["TibiaSoleRaw"] = tibia_sole
    # Dorsiflexie (tibia kantelt naar de voetrug) verkleint de hoek -> positief.
    df["Ankle"] = -(tibia_sole - tibia_sole_neutral)
    return df


# --------------------------------------------------------------------------
# Gebeurtenissen (oef 1 en 4)
# --------------------------------------------------------------------------
def detect_events(df: pd.DataFrame, ref: dict) -> list[dict]:
    """
    IC : laagste punt van de hielmarker na de swing (verticale snelheid hiel
         gaat van negatief naar nul).
    FC : maximale voorwaartse versnelling van de teenmarker na heel off
         (de voet "schiet" weg van de grond; Hreljac & Marshall, 2000).
    Foot flat : eerste frame na IC met voet-vloerhoek <= +2 graden.
    Heel off  : eerste frame na foot flat met voet-vloerhoek <= -2 graden.
    Enkel stances waarvan IC en FC allebei in de opname vallen, worden gehouden.
    """
    heel_z = df["RHEE_Z"].to_numpy()
    ff = df["FootFloor"].to_numpy()
    toe_ay = np.gradient(np.gradient(df["RTOE_Y"].to_numpy())) * FS**2
    vz = np.gradient(heel_z) * FS
    n = len(df)

    # kandidaat-IC: neerwaartse hielsnelheid wordt >= 0, terwijl de hiel laag is
    # en kort daarvoor duidelijk aan het dalen was (einde swing, > 200 mm/s).
    candidates = []
    for i in range(1, n):
        if vz[i - 1] < 0 <= vz[i] and heel_z[i] < ref["heel_z"] + 15:
            recent = candidates and i - candidates[-1] < 50   # zelfde stance
            if vz[max(0, i - 30):i].min() < -200 and not recent:
                candidates.append(i)

    events = []
    for ic in candidates:
        after = np.arange(ic, n)
        flat = after[ff[after] <= ROCKER_THR_DEG]
        if flat.size == 0:
            continue
        foot_flat = int(flat[0])
        off = after[(after > foot_flat) & (ff[after] <= -ROCKER_THR_DEG)]
        if off.size == 0:
            continue
        heel_off = int(off[0])
        if heel_off + FC_SEARCH >= n:          # opname stopt te vroeg
            continue
        win = np.arange(heel_off, heel_off + FC_SEARCH)
        fc = int(win[np.argmax(toe_ay[win])])
        if not ic < foot_flat < heel_off < fc:
            continue
        events.append({"IC": ic, "FootFlat": foot_flat, "HeelOff": heel_off, "FC": fc})

    # volgende IC (voor stride-tijd); ook een IC waarvan de FC buiten de
    # opname valt, telt hier mee.
    for ev in events:
        later = [c for c in candidates if c > ev["IC"]]
        ev["NextIC"] = later[0] if later else None
    return events


# --------------------------------------------------------------------------
# Figuren
# --------------------------------------------------------------------------
def _mark_contacts(ax, df, events, label=True):
    for k, ev in enumerate(events):
        ax.axvline(df["Time"].iloc[ev["IC"]], color="black", lw=1.0, ls="--",
                   label="IC" if (label and k == 0) else None)
        ax.axvline(df["Time"].iloc[ev["FC"]], color="0.45", lw=1.0, ls=":",
                   label="FC" if (label and k == 0) else None)


def _shade_rockers(ax, df, events):
    t = df["Time"].to_numpy()
    for ev in events:
        bounds = [ev["IC"], ev["FootFlat"], ev["HeelOff"], ev["FC"]]
        for j in range(3):
            ax.axvspan(t[bounds[j]], t[bounds[j + 1]], color=ROCKER_COLORS[j],
                       alpha=0.9, lw=0, zorder=0)


def _rocker_legend_handles():
    return [Patch(color=c, label=l) for c, l in zip(ROCKER_COLORS, ROCKER_LABELS)]


def plot_contacts(trials: dict, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    for ax, (name, (df, events)) in zip(axes, trials.items()):
        ax.plot(df["Time"], df["RHEE_Z"], color=COLORS["heel"], lw=1.5, label="RHEE (hiel)")
        ax.plot(df["Time"], df["RTOE_Z"], color=COLORS["toe"], lw=1.5, label="RTOE (teen)")
        t = df["Time"].to_numpy()
        for k, ev in enumerate(events):
            ax.plot(t[ev["IC"]], df["RHEE_Z"].iloc[ev["IC"]], "v", color="black",
                    ms=7, label="IC" if k == 0 else None, zorder=5)
            ax.plot(t[ev["FC"]], df["RTOE_Z"].iloc[ev["FC"]], "^", color="#2ca02c",
                    ms=7, label="FC" if k == 0 else None, zorder=5)
            ax.axvline(t[ev["IC"]], color="black", lw=0.6, ls="--", alpha=0.6)
            ax.axvline(t[ev["FC"]], color="#2ca02c", lw=0.6, ls=":", alpha=0.8)
        ax.set_title(f"Trial {name}", fontsize=10, loc="left")
        ax.set_ylabel("Hoogte Z (mm)")
        ax.set_xlim(t[0], t[-1])
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, fontsize=9, frameon=False)
    axes[-1].set_xlabel("Tijd (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_angle(trials: dict, column: str, ylabel: str, color: str, path: Path,
               neg_label: str, pos_label: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for ax, (name, (df, events)) in zip(axes, trials.items()):
        _shade_rockers(ax, df, events)
        ax.plot(df["Time"], df[column], color=color, lw=1.6, zorder=3)
        ax.axhline(0, color="0.3", lw=0.8)
        _mark_contacts(ax, df, events)
        ax.set_title(f"Trial {name}", fontsize=10, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlim(df["Time"].iloc[0], df["Time"].iloc[-1])
        ax.grid(alpha=0.25, zorder=1)
        ax.text(1.005, 0.98, pos_label, transform=ax.transAxes, fontsize=7,
                va="top", rotation=90, color="0.35")
        ax.text(1.005, 0.02, neg_label, transform=ax.transAxes, fontsize=7,
                va="bottom", rotation=90, color="0.35")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles + _rocker_legend_handles(), labels + ROCKER_LABELS,
               loc="upper center", ncols=5, fontsize=9, frameon=False)
    axes[-1].set_xlabel("Tijd (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Tabellen
# --------------------------------------------------------------------------
def tabulate(trials: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev_rows, val_rows = [], []
    for name, (df, events) in trials.items():
        t = df["Time"].to_numpy()
        f = df["Frame"].to_numpy()
        for k, ev in enumerate(events, start=1):
            ic, ffl, ho, fc = ev["IC"], ev["FootFlat"], ev["HeelOff"], ev["FC"]
            stance = fc - ic
            stride = (ev["NextIC"] - ic) if ev["NextIC"] is not None else np.nan
            ev_rows.append({
                "Trial": name, "Stap": k,
                "IC_frame": f[ic], "FC_frame": f[fc],
                "IC_s": round(t[ic], 2), "FC_s": round(t[fc], 2),
                "FootFlat_frame": f[ffl], "HeelOff_frame": f[ho],
                "Stance_s": stance / FS,
                "Stride_s": stride / FS,
                "Stance_pct": 100 * stance / stride if stride == stride else np.nan,
                "Rocker1_pct": 100 * (ffl - ic) / stance,
                "Rocker2_pct": 100 * (ho - ffl) / stance,
                "Rocker3_pct": 100 * (fc - ho) / stance,
            })
            seg = slice(ic, fc + 1)
            ankle = df["Ankle"].to_numpy()
            footf = df["FootFloor"].to_numpy()
            # piek plantairflexie rond toe-off: zoek tot 10 frames na FC
            pf_end = min(len(df), fc + 10)
            val_rows.append({
                "Trial": name, "Stap": k,
                "FF_IC": footf[ic],
                "FF_max_r1": footf[ic:ffl + 1].max(),
                "FF_FC": footf[fc],
                "Ankle_IC": ankle[ic],
                "Ankle_min_r1": ankle[ic:ho + 1][: (ffl - ic) + 10].min(),
                "Ankle_max_DF": ankle[seg].max(),
                "Ankle_FC": ankle[fc],
                "Ankle_max_PF": ankle[ho:pf_end].min(),
            })
    return pd.DataFrame(ev_rows), pd.DataFrame(val_rows)


# --------------------------------------------------------------------------
def run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    standing = read_trial(STANDING_SHEET)
    ref = static_reference(standing)

    trials = {}
    for sheet in TRIAL_SHEETS:
        df = add_angles(read_trial(sheet), ref)
        events = detect_events(df, ref)
        df.to_csv(OUTPUT_DIR / f"{sheet}_angles.csv", index=False)
        trials[sheet] = (df, events)

    events_df, values_df = tabulate(trials)
    events_df.to_csv(OUTPUT_DIR / "events.csv", index=False)
    values_df.to_csv(OUTPUT_DIR / "stance_values.csv", index=False)

    figs = {
        "oef1": OUTPUT_DIR / "fig_oef1_contacten.png",
        "oef2": OUTPUT_DIR / "fig_oef2_voet_vloer.png",
        "oef3": OUTPUT_DIR / "fig_oef3_enkel.png",
    }
    plot_contacts(trials, figs["oef1"])
    plot_angle(trials, "FootFloor", "Voet-vloerhoek (°)", COLORS["ff"], figs["oef2"],
               neg_label="hiel omhoog", pos_label="tenen omhoog")
    plot_angle(trials, "Ankle", "Enkelhoek (°)", COLORS["ankle"], figs["oef3"],
               neg_label="plantairflexie", pos_label="dorsiflexie")

    return {"ref": ref, "trials": trials, "events": events_df,
            "values": values_df, "figs": figs}


if __name__ == "__main__":
    res = run()
    pd.set_option("display.width", 200)
    print("Statische referentie:", {k: round(v, 2) for k, v in res["ref"].items()})
    print(res["events"].round(2).to_string(index=False))
    print(res["values"].round(1).to_string(index=False))
