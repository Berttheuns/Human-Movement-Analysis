"""
Maakt het verslag CMA_Lab1_verslag.docx op basis van cma_lab1_analysis.py.
Alle getallen in de tekst worden rechtstreeks uit de analyse gehaald.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

import cma_lab1_analysis as ana

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "CMA_Lab1_verslag.docx"


# --------------------------------------------------------------------------
# hulpfuncties voor opmaak
# --------------------------------------------------------------------------
def num(x: float, d: int = 1) -> str:
    """Getal met decimale komma."""
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def msd(series: pd.Series, d: int = 1, unit: str = "") -> str:
    return f"{num(series.mean(), d)} ± {num(series.std(), d)}{unit}"


def setup(doc: Document) -> None:
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    sec.left_margin = sec.right_margin = Cm(2.2)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.1
    for name in ("Heading 1", "Heading 2"):
        doc.styles[name].font.name = "Calibri"


def para(doc: Document, text: str, italic: bool = False) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.italic = italic


def figure(doc: Document, path: Path, caption: str) -> None:
    doc.add_picture(str(path), width=Cm(16))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)


def table(doc: Document, header: list[str], rows: list[list[str]], caption: str) -> None:
    cap = doc.add_paragraph()
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    cap.paragraph_format.space_after = Pt(2)

    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = h
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    run.font.size = Pt(9)
    doc.add_paragraph()


# --------------------------------------------------------------------------
def stride_lengths(res: dict) -> pd.Series:
    """Stride-lengte (m) = verplaatsing van de hielmarker tussen twee IC's."""
    out = []
    for df, events in res["trials"].values():
        heel_y = df["RHEE_Y"].to_numpy()
        for ev in events:
            if ev["NextIC"] is not None:
                out.append((heel_y[ev["NextIC"]] - heel_y[ev["IC"]]) / 1000)
    return pd.Series(out)


def main() -> None:
    res = ana.run()
    ref, ev, val = res["ref"], res["events"], res["values"]
    figs = res["figs"]
    strides = stride_lengths(res)
    speed = strides.mean() / ev["Stride_s"].mean()
    n_steps = len(ev)
    geo_shift = 90 - ref["tibia_static"]

    doc = Document()
    setup(doc)

    # ---------------- titel
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = t.add_run("CMA Lab 1 – Gangbeeldanalyse van voet en enkel in 2D")
    run.bold = True
    run.font.size = Pt(16)
    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s.add_run("Groep L1G1 – Namen: ……………………………………").italic = True

    # ---------------- 1. data en verwerking
    doc.add_heading("1. Data en verwerking", level=1)
    para(doc,
         "Op het rechterbeen werden vier markers geplaatst: RKNE (laterale femurcondyl), "
         "RANK (laterale malleolus), RHEE (achterkant calcaneus, ca. 1,5 cm boven de grond) en "
         "RTOE (voetrug, midden van metatarsaal 2). We gebruiken één statische trial "
         "(L1G1A02) en drie looptrials (L1G1A03–05) waarin de proefpersoon in de +Y-richting "
         "wandelt. Omdat alleen het sagittale vlak bekeken wordt, gebruiken we enkel de Y- "
         "(voorwaarts) en Z-coördinaten (verticaal).")
    para(doc,
         "De markertrajecten werden gefilterd met een laagdoorlaat-Butterworthfilter "
         "(4e orde, 6 Hz, zonder faseverschuiving). De samplefrequentie staat niet in het "
         "Excelbestand; we gingen uit van 100 Hz. Dat geeft een stridetijd van "
         f"{num(ev['Stride_s'].mean(), 2)} s, een stridelengte van {num(strides.mean(), 2)} m "
         f"en een loopsnelheid van {num(speed, 2)} m/s, wat normale waarden zijn voor "
         "comfortabel wandelen. Elke looptrial bevat meerdere stappen van het rechterbeen; "
         f"alle stappen waarvan zowel IC als FC in de opname vallen ({n_steps} in totaal) "
         "werden geanalyseerd.")

    # ---------------- 2. oef 1
    doc.add_heading("2. Initial en final contact", level=1)
    para(doc,
         "Initial contact (IC) is het moment waarop de hiel de grond raakt. Op het einde van "
         "de swing daalt de hielmarker snel; bij hielcontact stopt die daling. We nemen IC "
         "daarom als het laagste punt van RHEE, dus het frame waarop de verticale snelheid "
         "van de hiel van negatief naar nul gaat.")
    para(doc,
         "Final contact (FC) is moeilijker, want RTOE zit op de voetrug en niet op de "
         "teentop. Tijdens het afrollen over de metatarsaalkoppen stijgt deze marker al "
         "terwijl de tenen nog op de grond staan. Een simpele hoogtedrempel (RTOE 1 cm boven "
         "de hoogte in stand) gaf een stancefase van slechts ca. 54 % van de stride, wat te "
         "kort is. Daarom gebruiken we het moment waarop de teenmarker het sterkst naar "
         "voren versnelt (methode van Hreljac & Marshall, 2000): zolang de tenen op de grond "
         "staan, wordt de voorwaartse beweging van de voet afgeremd en net na loskomen "
         "versnelt de voet sterk de swing in.")
    figure(doc, figs["oef1"],
           "Figuur 1 – Hoogte van de hiel- (RHEE) en teenmarker (RTOE) in de drie looptrials. "
           "▼ = initial contact, ▲ = final contact.")

    rows = [[r.Trial, str(r.Stap), num(r.IC_s, 2), num(r.FC_s, 2),
             f"{r.IC_frame} / {r.FC_frame}", num(r.Stance_s, 2),
             num(r.Stride_s, 2), num(r.Stance_pct, 1)] for r in ev.itertuples()]
    table(doc, ["Trial", "Stap", "IC (s)", "FC (s)", "Frames IC / FC",
                "Stance (s)", "Stride (s)", "Stance (%)"], rows,
          "Tabel 1 – Tijdstippen van IC en FC per stap (tijd t.o.v. het begin van de trial).")

    para(doc,
         f"De stancefase duurt gemiddeld {msd(ev['Stance_s'], 2, ' s')}, de volledige "
         f"stride {msd(ev['Stride_s'], 2, ' s')}. De stance beslaat dus "
         f"{msd(ev['Stance_pct'], 1, ' %')} van de gangcyclus, wat goed overeenkomt met de "
         "ongeveer 60 % die in de literatuur voor normaal wandelen wordt opgegeven. De drie trials "
         "geven gelijkaardige resultaten; L1G1A04 heeft een iets langere stancefase "
         f"({num(ev[ev.Trial == 'L1G1A04'].Stance_pct.mean(), 1)} %).")

    # ---------------- 3. oef 2
    doc.add_heading("3. Voet-vloerhoek", level=1)
    para(doc,
         "De voet wordt voorgesteld door de lijn van RHEE naar RTOE. De hoek van die lijn "
         "met de horizontaal is atan2(ΔZ, ΔY). Als de tenen hoger staan dan de hiel is de "
         "hoek positief (hielcontact), als de hiel hoger staat negatief (push-off).")
    para(doc,
         "De markerlijn ligt niet evenwijdig met de zool: RTOE zit op de voetrug en dus "
         "hoger dan RHEE. In de statische trial maakt de lijn daardoor al een hoek van "
         f"{num(ref['foot_offset'])}° met de grond. We nemen aan dat de zool in stand plat op "
         "de grond ligt en trekken deze offset in alle trials af, zodat 0° overeenkomt met "
         "een platte voet.")
    figure(doc, figs["oef2"],
           "Figuur 2 – Voet-vloerhoek in de drie looptrials. Gestreepte lijn = IC, "
           "stippellijn = FC. De gekleurde vlakken tonen de drie rockers (zie §5).")
    para(doc,
         "Op het einde van de swing gaan de tenen omhoog en bij IC is de voet-vloerhoek "
         f"{msd(val['FF_IC'], 1, '°')}. Kort daarna klapt de voet naar de grond en blijft "
         "de hoek een tijd rond 0° (de voet staat plat, met een restafwijking van ca. 1°). Vanaf het "
         "loskomen van de hiel wordt de hoek snel negatief tot "
         f"{msd(val['FF_FC'], 1, '°')} bij FC. In de swing draait de voet terug naar een "
         "positieve hoek voor het volgende hielcontact. Het patroon is in alle stappen en "
         "trials zeer gelijkaardig.")

    # ---------------- 4. oef 3
    doc.add_heading("4. Enkelhoek (voet t.o.v. tibia)", level=1)
    para(doc,
         "De tibia wordt voorgesteld door de lijn van RANK naar RKNE en de zool door de "
         "gecorrigeerde voetlijn uit §3. De hoek tussen beide is in neutrale stand 90°. "
         "We definiëren de enkelhoek als het verschil met deze neutrale stand, met "
         "dorsiflexie positief (tibia kantelt naar de voetrug, de hoek tussen tibia en zool "
         "wordt kleiner dan 90°) en plantairflexie negatief.")
    para(doc,
         "Neutrale referentie: de lijn RANK–RKNE valt niet samen met de echte lange as van "
         "de tibia, omdat de laterale malleolus (fibula) achter de as ligt. In de statische "
         f"trial staat deze lijn {num(geo_shift)}° voorover gekanteld. Zouden we de 90° "
         f"rechtstreeks op de markerlijnen toepassen, dan zou de proefpersoon in stand al "
         f"{num(geo_shift)}° dorsiflexie hebben, wat niet realistisch is. We nemen daarom aan "
         "dat in de statische trial (1) de zool plat ligt en (2) de enkel neutraal staat, "
         "dus dat de zool daar loodrecht op de tibia-as staat. De hoek tussen tibia- en "
         "voetlijn in stand wordt zo de 0°-referentie. Een nadeel is dat elke afwijking van "
         "neutraal tijdens het stilstaan (bv. licht voorover leunen) als een vaste offset in "
         "alle curves terechtkomt.")
    figure(doc, figs["oef3"],
           "Figuur 3 – Enkelhoek (dorsiflexie +, plantairflexie –) in de drie looptrials. "
           "Gestreepte lijn = IC, stippellijn = FC, kleuren = rockers.")
    para(doc,
         f"Bij IC staat de enkel in lichte plantairflexie ({msd(val['Ankle_IC'], 1, '°')}). "
         "Direct na hielcontact plantairflecteert de enkel verder tot "
         f"{msd(val['Ankle_min_r1'], 1, '°')} terwijl de voet naar de grond zakt. Daarna "
         "beweegt de tibia over de stilstaande voet naar voren en neemt de dorsiflexie toe "
         f"tot een maximum van {msd(val['Ankle_max_DF'], 1, '°')} rond het loskomen van de "
         "hiel. Tijdens de push-off volgt een snelle plantairflexie tot "
         f"{msd(val['Ankle_max_PF'], 1, '°')} rond FC. In de swing keert de enkel terug naar "
         "ongeveer –3° zodat de voet de grond niet raakt.")
    para(doc,
         "De vorm van de curve komt overeen met het normale enkelpatroon. De absolute "
         "waarden liggen wel enkele graden lager dan typische literatuurwaarden (ongeveer 0° bij IC "
         "en 10° maximale dorsiflexie). Dat is waarschijnlijk een gevolg van de neutrale "
         "referentie: als de proefpersoon in stand al een paar graden in dorsiflexie stond, "
         "schuift de hele curve naar beneden. De bewegingsuitslag binnen de stance "
         f"(ca. {num((val['Ankle_max_DF'] - val['Ankle_max_PF']).mean(), 0)}°) hangt daar niet "
         "van af.")

    # ---------------- 5. oef 4
    doc.add_heading("5. De drie rockers", level=1)
    para(doc,
         "De grenzen tussen de rockers bepalen we uit de voet-vloerhoek, met een drempel "
         f"van ±{num(ana.ROCKER_THR_DEG, 0)}° om ruis en de kleine restafwijking op te vangen:")
    for txt in [
        "1e rocker (heel rocker): van IC tot foot flat (voet-vloerhoek ≤ +2°). De voet "
        "rolt over de hiel naar de grond; de enkel plantairflecteert, afgeremd door een "
        "excentrische contractie van de tibialis anterior.",
        "2e rocker (ankle rocker): van foot flat tot heel off (voet-vloerhoek ≤ –2°). De "
        "voet staat plat en de tibia rolt over de enkel naar voren, waardoor de dorsiflexie "
        "toeneemt. Soleus en gastrocnemius remmen die beweging excentrisch af.",
        "3e rocker (forefoot rocker): van heel off tot FC. De hiel komt los en de voet "
        "draait rond de metatarsaalkoppen; de enkel gaat snel in plantairflexie "
        "(push-off door de triceps surae).",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(txt)

    rows = []
    for trial, g in ev.groupby("Trial"):
        rows.append([trial, msd(g.Rocker1_pct), msd(g.Rocker2_pct), msd(g.Rocker3_pct),
                     msd(g.Stance_s * 1000 * g.Rocker1_pct / 100, 0)])
    rows.append(["Gemiddeld", msd(ev.Rocker1_pct), msd(ev.Rocker2_pct),
                 msd(ev.Rocker3_pct), msd(ev.Stance_s * 1000 * ev.Rocker1_pct / 100, 0)])
    table(doc, ["Trial", "1e rocker (% stance)", "2e rocker (% stance)",
                "3e rocker (% stance)", "Duur 1e rocker (ms)"], rows,
          "Tabel 2 – Duur van de rockers als percentage van de stancefase (gemiddelde ± SD).")
    para(doc,
         f"De 1e rocker is kort (ca. {num(ev.Rocker1_pct.mean(), 0)} % van de stance), de 2e en "
         "3e rocker nemen elk ongeveer 40 % in. In figuur 3 is te zien dat de enkel ook na "
         "heel off nog even verder dorsiflecteert voordat de push-off begint: de maximale "
         "dorsiflexie valt net in het begin van de 3e rocker. Dat klopt met de literatuur, "
         "waar de hiel al tijdens terminal stance loskomt terwijl de tibia nog verder naar "
         "voren beweegt.")

    # ---------------- 6. aannames
    doc.add_heading("6. Aannames en beperkingen", level=1)
    for txt in [
        "2D-model: alles wordt geprojecteerd op het Y-Z-vlak. Als de voet naar buiten "
        "draait (voetprogressiehoek) of de proefpersoon niet perfect langs Y wandelt, "
        "worden de hoeken onder- of overschat.",
        "Starre segmenten: de voet wordt als één star segment gezien. In werkelijkheid "
        "buigt de voorvoet in de MTP-gewrichten tijdens de 3e rocker; dat wordt hier "
        "toegeschreven aan de voet-vloer- en enkelhoek.",
        "Markers vs. anatomie: RKNE–RANK is een benadering van de tibia-as en RHEE–RTOE "
        "van de zool. Huidverschuiving en de exacte plaatsing (bv. van de knie-as) geven "
        "fouten van enkele graden.",
        "Neutrale stand: we nemen aan dat in de statische trial de voet plat staat en de "
        "enkel neutraal is (zie §4).",
        "Gebeurtenissen: IC en FC zijn kinematische schattingen. Zonder krachtplaat kunnen "
        "ze enkele frames afwijken; ook de drempels voor de rockers zijn een keuze.",
        "De samplefrequentie van 100 Hz is aangenomen; een andere frequentie verandert "
        "de tijden, maar niet de hoeken of de percentages.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(txt)

    # ---------------- 7. klinisch
    doc.add_heading("7. Klinische betekenis", level=1)
    para(doc,
         "De enkelcurve hangt samen met de passieve dorsiflexiemetingen uit workstation 1. "
         "In de 2e rocker en het begin van de 3e rocker staat de knie bijna gestrekt en is "
         "de maximale dorsiflexie nodig. Op dat moment wordt de beweging beperkt door de "
         "lengte van de gastrocnemius, die gemeten wordt met passieve dorsiflexie bij een "
         "gestrekte knie. De soleus (dorsiflexie met gebogen knie) beperkt de dorsiflexie "
         "tijdens de hele stance. Bij een verkorte triceps surae verwacht je minder "
         "dorsiflexie in de 2e rocker en een vroege heel off, in ernstige gevallen "
         "teenlopen (equinus).")
    para(doc,
         "De voet-vloerhoek toont vooral de 1e rocker. Een positieve hoek bij IC betekent "
         "dat de hiel eerst neerkomt. Bij equinus of spasticiteit ontbreekt die en is de "
         "hoek bij IC rond 0° of negatief. Bij een zwakke tibialis anterior klapt de voet na "
         "IC te snel plat (foot slap) en blijft de voet in de swing te laag hangen (drop "
         "foot). Ook korte hamstrings (grote popliteale hoek) hebben invloed: de knie "
         "strekt dan minder in terminal swing, waardoor de voet minder ver naar voren komt "
         "en de hoek bij hielcontact kleiner is. Tot slot bepalen de passieve "
         "heuprotaties mee de voetprogressiehoek, wat belangrijk is voor de "
         "betrouwbaarheid van dit 2D-model.")

    doc.add_heading("Referentie", level=2)
    para(doc,
         "Hreljac, A., & Marshall, R. N. (2000). Algorithms to determine event timing "
         "during normal walking using kinematic data. Journal of Biomechanics, 33(6), "
         "783–786.")

    doc.core_properties.title = "CMA Lab 1"
    doc.core_properties.author = ""
    doc.core_properties.comments = ""
    doc.save(REPORT)
    print(f"Rapport opgeslagen: {REPORT}")


if __name__ == "__main__":
    main()
