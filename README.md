# Rekonstrukcja 3D z RGB-D i augmentacja wyglądu

Kod pracy magisterskiej (Politechnika Poznańska, Informatyka):
**„Czy fotorealistyczna augmentacja syntetycznych obrazów RGB poprawia rekonstrukcję
3D z rzeczywistych pomiarów RGB-D?”** Porównanie dotyczy dostrajania RaySt3R
z RGB BASE, CLASSIC i PHOTO. PHOTO to edycje FLUX.2-klein kontrolowane pod kątem
sylwetki. Geometrie GSO/ShapeNet renderowano w Isaac Sim, a zakłócenia głębi
oszacowano na YCB-Video. Zachowano ocenę HomebrewedDB, badanie wstępne HOPE,
kontrolę szarości RGB, porównanie publicznych modeli oraz osobne badanie TRELLIS.2/B1.

Repozytorium zawiera źródła, konfiguracje i niewielkie manifesty identyfikatorów.
Modele, obrazy, geometrie, checkpointy, wyniki i rękopis należy pozyskać osobno.
Liczebności opisane poniżej pochodzą z rozdziałów „Dane”, „Metoda”, „Protokół”,
„Wyniki” i dodatków „Ustawienia badania” oraz „Generowane geometrie”.

## Struktura

| Katalog | Zawartość |
|---|---|
| `src/masters_rgbd/` | Kontrakty RGB-D, geometria, metryki, model B1 i generator SAPIEN |
| `scripts/data/` | Zbiory, geometrie, Isaac Sim, głębia i etykiety powierzchni |
| `scripts/photo/` | PHOTO, kontrola integralności i sylwetek, dopuszczenie edycji |
| `scripts/train/`, `scripts/predict/` | Dostrajanie RaySt3R i predykcje modeli oraz kontroli |
| `scripts/eval/`, `scripts/analysis/` | Ocena powierzchni i bootstrap kontroli |
| `scripts/figures/` | Rysunki i tabele pracy |
| `scripts/preliminary_hope/`, `scripts/trellis_b1/` | Odrębne badania wstępne i B1 |
| `scripts/common/` | Wspólne adaptery, ścieżki i wiązania SHA-256 |
| `configs/`, `manifests/`, `environments/`, `tests/` | Parametry, tożsamości wejść, środowiska i testy CPU |

## Instalacja i wymagania

Python 3.11, [uv](https://docs.astral.sh/uv/) i git. Z katalogu repozytorium:

```powershell
uv sync --locked --python 3.11
uv run python -m compileall -q src scripts tests
uv run ruff format --check
uv run ruff check
uv run pytest -q
uv run python scripts/eval/check_surface_metrics.py
uv run python scripts/data/check_label_first_hits.py
```

Domyślny lock zawiera PyTorch CPU i zależności analiz. Renderowanie, trening i PHOTO
wymagają NVIDIA GPU i zgodnych sterowników CUDA. Isaac Sim wymaga obsługi RTX;
SAPIEN/TRELLIS i rozszerzenia OctMAE uruchamiaj na Linux według instrukcji autorów.
Wymagania pamięci GPU zależą od modelu i odciążania pamięci.

| Etap | Środowisko |
|---|---|
| RaySt3R | `environments/ray/Containerfile`: Torch 2.7.0 / CUDA 12.8 |
| PHOTO | `environments/photo/requirements.txt`, osobne środowisko GPU |
| B1 | `uv sync --project environments/b1`, indeks CUDA 12.8 |
| TRELLIS/SAPIEN | `environments/trellis/requirements.txt` i kompilowane zależności upstream |
| OctMAE | Środowisko upstream z Torch 2.2.0 wymaganym przez runner |
| Isaac Sim i SAM2 | Instalacja według dokumentacji upstream |

Przykład: `docker build -f environments/ray/Containerfile -t rgbd-ray .`.
W środowisku zewnętrznego modelu udostępnij katalog repozytorium i `src` w
`PYTHONPATH` (Windows: `.;src`, Linux: `.:src`). Interpreter Isaac Sim uruchamia
skrypty renderujące. Profile mają różne wymagania wersji; nie łącz ich w jedno
środowisko. Zależności TRELLIS nie stanowią pełnego historycznego lockfile.

## Zewnętrzne dane i modele

Pobierz samodzielnie dane, implementacje i wagi z oficjalnych źródeł zgodnie
z ich licencjami. To repozytorium nie udziela praw do tych zasobów.

| Zasób | Oficjalne źródło |
|---|---|
| GSO | [Google Scanned Objects](https://research.google/resources/datasets/scanned-objects/), [Gazebo Fuel](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research) |
| ShapeNet | [ShapeNet](https://shapenet.org/) |
| YCB-Video, HomebrewedDB, HOPE w BOP | [BOP datasets](https://bop.felk.cvut.cz/datasets/) |
| HOPE | [repozytorium autorów](https://github.com/swtyree/hope-dataset) |
| RaySt3R | [kod i instrukcje wag](https://github.com/Duisterhof/rayst3r) |
| DINOv2 | [repozytorium Meta](https://github.com/facebookresearch/dinov2) |
| OctMAE | [repozytorium TRI-ML](https://github.com/TRI-ML/OctMAE) |
| FLUX.2-klein-4B | [karta modelu BFL](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| FLUX.2-klein-9B, badanie wstępne | [karta modelu BFL](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| SAM2 | [repozytorium Meta](https://github.com/facebookresearch/sam2) |
| TRELLIS.2 | [kod Microsoft](https://github.com/microsoft/TRELLIS.2), [wagi 4B](https://huggingface.co/microsoft/TRELLIS.2-4B) |
| FLUX.2-dev dla zasobów TRELLIS | [wariant bnb-4bit](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit) |
| RMBG-2.0 | [karta modelu BRIA](https://huggingface.co/briaai/RMBG-2.0) |
| Isaac Sim | [dokumentacja NVIDIA](https://docs.isaacsim.omniverse.nvidia.com/) |
| SAPIEN | [dokumentacja projektu](https://sapien.ucsd.edu/) |

RaySt3R oczekuje `vendor/rayst3r`, `vendor/dinov2`, `weights/rayst3r.pth`
i `weights/dinov2_vitl14_reg4_pretrain.pth` w wybranym katalogu `--work`.
TRELLIS wskaż przez `RGBD_TRELLIS_ROOT`. Zbiory BOP umieść pod
`RGBD_DATASETS`; nazwy względne są jawne w skryptach, np.
`bop-ycbv-real-v1/ycbv` i `bop-hb-evolution-v1/hb`.
HF_TOKEN, jeżeli potrzebny, ustaw lokalnie i nie zapisuj w git.

## Kolejność reprodukcji

Uruchamiaj skrypty z katalogu repozytorium, np.:

```powershell
$env:RGBD_DATASETS = (Join-Path $PWD "data")
$env:RGBD_MODEL_WORK = $PWD.Path
uv run python scripts/data/export_rgbd_scenes.py --help
```

`RGBD_WORKSPACE` wskazuje katalog danych i wyników (domyślnie checkout),
`RGBD_DATASETS` katalog zbiorów, a `RGBD_FIGURES` katalog ilustracji dla skryptów,
które korzystają z tej zmiennej. Pozostałe generatory zapisują pod
`RGBD_WORKSPACE/artifacts/figures`. Kod i konfiguracje są odczytywane z checkoutu.
Ścieżki `runs/...-202609...` stanowią identyfikatory etapów historycznego protokołu.
Skrypty wsadowe używają tych ścieżek jawnie; `--help` opisuje ich interfejs.
Manifesty, reguły dopuszczenia i wiązania SHA-256 trzeba przygotować przed
konsumentami. Nie pomijaj walidacji hashy, aby ominąć brakujący plik.

1. **Geometrie i podział.** `scripts/data/catalog_gso.py`,
   `scripts/data/prepare_gso_assets.py`, `scripts/data/prepare_gso_scenes.py`;
   `scripts/data/audit_shapenet.py`, `scripts/data/audit_shapenet_physics.py`,
   `scripts/data/prepare_shapenet_scenes.py`.
   Następnie `scripts/data/group_geometry_families.py`,
   `scripts/data/create_geometry_manifest.py`, `scripts/data/split_geometry_families.py`.
2. **Zakłócenia głębi.** `scripts/data/select_ycbv_noise_cohort.py` z
   `configs/ycbv_noise_cohort.json`, `scripts/common/bop_adapter.py`,
   `scripts/data/estimate_depth_noise_ycbv.py`, `scripts/data/corrupt_rendered_depth.py`.
3. **Rendery i nadzór.** W Isaac Sim: `scripts/data/prepare_isaac_assets.py`,
   `scripts/data/admit_geometries.py`, `scripts/data/render_scene_batches.py`,
   `scripts/data/render_isaac_scenes.py`. Kontrola:
   `scripts/data/audit_rendered_scenes.py`, `scripts/data/audit_triangle_meshes.py`,
   `scripts/data/export_rgbd_scenes.py`, `scripts/data/verify_rgbd_export.py`.
   Etykiety: `scripts/data/prepare_surface_labels.py`, `scripts/data/run_label_batch.py`,
   `scripts/data/generate_surface_labels.py`, `scripts/data/verify_surface_labels.py`.
   Badanie główne używa 320 geometrii po 20 obserwacji; kontrola syntetyczna
   36 geometrii po 20 obserwacji (rozdziały „Dane” i „Protokół”).
4. **PHOTO i CLASSIC.** `scripts/photo/prepare_edit_inputs.py`,
   `scripts/photo/generate_edits.py`, `scripts/photo/verify_generated_edits.py`,
   `scripts/photo/prepare_silhouette_inputs.py`, `scripts/photo/predict_silhouettes.py`,
   `scripts/photo/score_silhouettes.py`, `scripts/photo/finalize_admission.py`.
   Kryteria: `scripts/photo/admission_policy.py`; receptura: `configs/photo_recipe.json`.
   Zachowaj rzeczywiste decyzje ręczne i odróżnienie dopuszczonych, nieprzejrzanych
   obrazów od indywidualnie zaakceptowanych. Odrzucone edycje wracają do SOURCE
   w obu sparowanych wariantach, bez regeneracji. CLASSIC i wspólny harmonogram
   implementuje `scripts/train/augmentation_schedule.py`.
5. **Dostrajanie.** `scripts/train/prepare_training_plans.py --training <TRAIN6400>
   --photo <dopuszczone-PHOTO> --output <plany>` tworzy plany dla ziaren 0, 1, 2
   z `configs/training.json`. `scripts/train/train_ray_variants.py` uruchamiaj dla
   każdego wariantu. Każdy trening ma 25 600 aktualizacji (dodatek konfiguracji).
   Gdy dany przebieg wymaga poprawki pochodnej, użyj
   `scripts/train/prepare_confidence_amendment.py --output <kontrola> --results <wyniki>
   --seed <ziarno>` i `scripts/train/train_ray_with_stable_confidence.py
   --amendment-control <kontrola>` z tymi samymi argumentami treningu.
   Rozkład poprawek podaje konfiguracja: CLASSIC dla ziarna 2 wymaga zachowania
   oryginalnego treningu do checkpointu 9600 i wznowienia z poprawką (`--resume`).
   Integralność rzeczywistej ekspozycji sprawdza `scripts/train/audit_augmentation_exposure.py`.
6. **HB i predykcje.** `scripts/data/select_homebrewed_observations.py`,
   `scripts/data/audit_homebrewed.py`, `scripts/data/export_homebrewed.py`,
   `scripts/data/verify_homebrewed_export.py`, `scripts/data/prepare_homebrewed_inputs.py`,
   `scripts/data/verify_homebrewed_inputs.py`. Lista referencyjna:
   `manifests/homebrewed_observations.json` (198 obserwacji, 33 obiekty; „Protokół”).
   `scripts/eval/prepare_evaluation_protocol.py`,
   `scripts/eval/prepare_homebrewed_reference.py`,
   `scripts/predict/prepare_prediction_inputs.py`,
   `scripts/predict/predict_ray_homebrewed.py`,
   `scripts/predict/predict_octmae_homebrewed.py`. Predykcja Ray zachowuje FP32
   z `scripts/common/ray_inference.py`. GT pozostaje poza wejściami modeli.
7. **Ocena i kontrole.** `scripts/eval/score_homebrewed.py` oraz
   `scripts/eval/surface_metrics.py`: F5, precyzja, pokrycie, podzbiory powierzchni,
   porażki, limity punktów i bootstrap obiektów, scen oraz obiektów i treningów.
   Parametry: `configs/evaluation.json`. Kontrole z `configs/supplement.json`:
   `scripts/predict/prepare_control_inputs.py`,
   `scripts/predict/prepare_synthetic_reference.py`,
   `scripts/predict/predict_ray_controls.py`, `scripts/eval/score_controls.py`,
   `scripts/analysis/bootstrap_controls.py`. Scoring czyta lokalne artefakty przez `--job`.
8. **Badanie wstępne.** `scripts/preliminary_hope/audit_dataset.py`,
   `scripts/preliminary_hope/audit_camera_calibration.py`,
   `scripts/preliminary_hope/prepare_ray_inputs.py`,
   `scripts/preliminary_hope/prepare_edit_contexts.py`,
   `scripts/preliminary_hope/verify_edit_contexts.py`,
   `scripts/preliminary_hope/prepare_edit_plan.py`,
   `scripts/preliminary_hope/generate_edits.py`,
   `scripts/preliminary_hope/verify_generated_edits.py`,
   `scripts/preliminary_hope/admit_edits.py`,
   `scripts/preliminary_hope/prepare_training_plan.py`,
   `scripts/preliminary_hope/train_ray.py`, `scripts/preliminary_hope/predict_ray.py`,
   `scripts/preliminary_hope/score_ray.py`, `scripts/preliminary_hope/bootstrap_objects.py`,
   `scripts/preliminary_hope/write_result_tables.py`.
   Końcowe zestawienie pilota zawiera także YCB-V i syntetyczne ABO (dodatek B);
   historyczne manifesty tych kohort nie są tutaj kompletne.
9. **TRELLIS.2/B1.** `scripts/trellis_b1/generate_reference_images.py`,
   `scripts/trellis_b1/remove_image_background.py`, `scripts/trellis_b1/generate_meshes.py`,
   `scripts/trellis_b1/prepare_assets.py`, `scripts/trellis_b1/prepare_backgrounds.py`,
   `scripts/trellis_b1/render_sapien_scenes.py`.
   Następnie `scripts/trellis_b1/prepare_query_cache.py` tworzy cache i jego preflight;
   `scripts/trellis_b1/train_predict_comparison.py --prepare` wiąże plan z własnymi
   wejściami (`--inventory`, `--cache-root`, `--preflight-report`, `--dataset-root`).
   Zgodność podziału i harmonogramów jest sprawdzana względem `manifests/trellis_b1.json`.
   Ten sam runner wykonuje trening i predykcje z `--arm`, `--seed`, `--output-root`.
   Ocenę i bootstrap wykonują `scripts/trellis_b1/score_surface_predictions.py`,
   `scripts/trellis_b1/bootstrap_objects_trainings.py`,
   `scripts/trellis_b1/score_point_budgets.py` (z `--config` wskazującym
   `resolved-config.json` treningu) i `scripts/trellis_b1/bootstrap_point_budgets.py`.
10. **Rysunki i tabele.** Po wygenerowaniu wejść uruchom skrypty z poniższej mapy.
    Najpierw `scripts/figures/draw_result_comparisons.py`, potem
    `scripts/figures/write_result_tables.py`, który korzysta z jego ledgeru.
    Wyjście znajduje się w `artifacts/figures/`; `evidence/` rejestruje pochodzenie
    nowych ilustracji i zestawień.

## Mapa wyników, rysunków i tabel

Identyfikatory w nawiasach są etykietami z rękopisu. Mapa dotyczy bieżącej,
skróconej wersji pracy wskazanej przez evidence-map źródłowego HEAD.
Tabele redagowane bezpośrednio w rękopisie mają tutaj generowane dane liczbowe
lub konfiguracje; rękopis nie jest publikowany.

| Wynik / rysunek / tabela w pracy | Skrypt(y) | Wejścia | Wyjście |
|---|---|---|---|
| Rozdz. 3–4: certyfikowane sceny SOURCE | `scripts/data/render_isaac_scenes.py`, `scripts/data/audit_triangle_meshes.py`, `scripts/data/export_rgbd_scenes.py` | Dopuszczone GSO/ShapeNet, rodziny i podział | RGB, czysta głębia, pozy, powierzchnie i manifesty |
| Dodatek A: estymacja zakłóceń głębi (`data-depth-estimation`) | `scripts/data/estimate_depth_noise_ycbv.py`, `scripts/figures/draw_depth_noise.py` | Wybrana kohorta YCB-V, audyt głębi | Parametry zakłóceń, `data-depth-estimation.*` |
| Rozdz. 4: kontrola PHOTO (`photo-main`) | `scripts/photo/score_silhouettes.py`, `scripts/photo/finalize_admission.py`, `scripts/figures/draw_dataset_examples.py` | SOURCE/PHOTO, maski SAM, rzeczywiste decyzje | Dopuszczenie i fallback, `data-photo-main.*` |
| Rozdz. 6: HB (`hb-main`, `main-effect`) | `scripts/eval/score_homebrewed.py`, `scripts/figures/draw_result_comparisons.py` | GT i predykcje wszystkich metod, protokół | `summaries.json`, kontrasty F5, `results-effect.*`, `evidence/results.json` |
| Rozdz. 6: gęstość, precyzja i pokrycie (`density`) | `scripts/eval/score_homebrewed.py`, `scripts/figures/draw_result_comparisons.py` | Te same obserwacje i cztery reguły liczby punktów | `results-density.*`, statystyki powierzchni |
| Rozdz. 6 i dodatek B: modele publiczne, F5 i reguły punktów (`hb-literature`, `hb-all-1`, `hb-surface-all`) | `scripts/predict/predict_ray_homebrewed.py`, `scripts/predict/predict_octmae_homebrewed.py`, `scripts/eval/score_homebrewed.py`, `scripts/figures/write_result_tables.py` | Publiczne wagi, HB, zamrożone wyniki | Predykcje, dane tabel i `appendices/02-wyniki.md` |
| Rozdz. 6: szarość zamiast RGB (`rgb-control`) | `scripts/predict/prepare_control_inputs.py`, `scripts/predict/predict_ray_controls.py`, `scripts/eval/score_controls.py`, `scripts/analysis/bootstrap_controls.py`, `scripts/figures/write_control_tables.py` | Te same checkpointy i HB, neutralne RGB | Sparowane wyniki, bootstrap obiektów i treningów, tabele kontroli |
| Rozdz. 6: geometrie odłożone (`synthetic-control`) | `scripts/predict/prepare_synthetic_reference.py`, `scripts/predict/predict_ray_controls.py`, `scripts/eval/score_controls.py`, `scripts/analysis/bootstrap_controls.py`, `scripts/figures/write_control_tables.py` | Odłożone rendery, GT i te same checkpointy | Ocena i tabela kontroli syntetycznej |
| Dodatek B: badanie wstępne (`pilot-cohorts`) | `scripts/preliminary_hope/score_ray.py`, `scripts/preliminary_hope/bootstrap_objects.py`, `scripts/preliminary_hope/write_result_tables.py`, `scripts/figures/write_result_tables.py` | Predykcje pilota HOPE/YCB-V/ABO, GT i protokół | Wyniki kohort i tabela pilota |
| Dodatek C: TRELLIS/B1 (`trellis-study-design`, `trellis-diversity-means`, `trellis-diversity-contrasts`) | `scripts/trellis_b1/train_predict_comparison.py`, `scripts/trellis_b1/score_surface_predictions.py`, `scripts/trellis_b1/bootstrap_objects_trainings.py`, `scripts/trellis_b1/score_point_budgets.py`, `scripts/trellis_b1/bootstrap_point_budgets.py` | Podział B1, cache, predykcje walidacyjne i reguły punktów | Średnie i sparowane kontrasty w JSON, dane tabel dodatku |
| Rozdz. 6: macierze HB, niewystarczające wejścia, modele publiczne (`hb-matrix-low/medium/high`, `insufficient`, `public-models`) | `scripts/figures/draw_reconstructions.py` | Wybrane identyfikatory, RGB, maski, GT, predykcje, scores | `results-hb-matrix-*`, `results-insufficient-inputs.*`, `results-public-models.*` |
| Rozdz. 3: przykłady, rozkłady, HB (`data-synthetic-gallery`, `data-distributions`, `data-hb-gallery`, `sofa-example`, `data-abo-pipeline`) | `scripts/figures/draw_dataset_examples.py`, `scripts/figures/draw_reconstructions.py` | Manifesty danych, geometrie, RGB-D, decyzje PHOTO | Galerie i rysunki `data-*` |
| Rozdz. 3: główny pipeline (`data-main-pipeline`) | `scripts/figures/draw_geometry_protocol.py` | Konfiguracja i opis etapów | Schemat danych i geometrii |
| Rozdz. 1 i dodatek A: plan badania, ekspozycja, rozkład pochodnych (`study-overview`, `numerical-runs`) | `scripts/figures/draw_augmentation_exposure.py`, `scripts/train/audit_augmentation_exposure.py` | Harmonogramy i logi treningu, `configs/training.json` | Schemat planu, ekspozycja, dane kontroli implementacji |
| Rozdz. 4: architektura (`method-architecture`) | `scripts/figures/draw_ray_architecture.py` | Schemat architektury i rozdzielenia wejść | Rysunek architektury RaySt3R |
| Rozdz. 2 i 5: geometria kamery, niejednoznaczność RGB-D, miary (`camera-geometry`, `rgbd-ambiguity`, `metric-directions`) | `scripts/figures/draw_camera_geometry.py`, `scripts/figures/draw_surface_and_rgbd_theory.py` | Własna geometria schematyczna; bez danych pomiarowych | Rysunki `theory-*` i geometrii kamery |
| Rozdz. 3: generowanie geometrii TRELLIS (`trellis-pipeline`) | `scripts/figures/draw_trellis_pipeline.py` | Schemat FLUX → RMBG → TRELLIS → SAPIEN | Schemat pipeline’u |
| Rozdz. 3–5 i dodatek A: role danych, konfiguracje, reguły nieokreślonych miar, bootstrap i skala ShapeNet | `scripts/eval/surface_metrics.py`, `scripts/data/prepare_shapenet_scenes.py`; `configs/evaluation.json`, `configs/training.json`, `configs/supplement.json` | Protokół i konfiguracje | Definicje oraz dane do tabel metodologicznych |

## Manifesty i ograniczenia reprodukcji

- `manifests/homebrewed_observations.json`: odzyskane identyfikatory obserwacji HB,
  względne ścieżki BOP i sumy kontrolne. Odtwórz pełną rezerwację przez selektor HB
  na wskazanej rewizji oficjalnego zbioru; porównaj identyfikatory i hashe plików.
- `manifests/photo_population.json`: identyfikatory wszystkich wygenerowanych
  obrazów TRAIN, również później odrzuconych. Kolejność spisu archiwum nie jest
  certyfikowaną kolejnością treningu. Plik nie zawiera decyzji dopuszczenia.
- `manifests/trellis_b1.json` i wskazane pliki w `manifests/trellis_b1/`: podział,
  pule kształtów, parametry i dokładne harmonogramy; każdy plik ma mniej niż 1 MB.
  Runner sprawdza sumy kontrolne fragmentów przed odtworzeniem wspólnego planu.
- `configs/ycbv_noise_cohort.json`: deterministyczny wybór kohorty estymacji głębi.
- `configs/code_fingerprints.json`: tożsamość uporządkowanego kodu B1. Zmiana nazw
  i formatowania zmieniła hashe źródeł, więc generuj nowe wiązania artefaktów.

Do **identycznej historycznej reprodukcji** nadal brakuje pełnego manifestu
geometrii SOURCE z rodzinami, podziałem i kolejnością TRAIN6400; oryginalnych
planów PHOTO z ziarnami i sumami obrazów; per-obserwacyjnych decyzji SAM/ręcznych
i rozróżnienia dopuszczenia bez indywidualnej oceny; związanych z nimi rekordów
autoryzacji i zamrożenia eksperymentu; kompletnych manifestów pilota oraz
inwentarza oryginalnych geometrii TRELLIS/SAPIEN. Spis obrazów z archiwum
nie pozwala wyprowadzić tych decyzji ani ich sum kontrolnych.

Dla nowego przebiegu przygotuj pełne manifesty przez etapy 1–4 i zachowaj
rzeczywiste decyzje przed treningiem. Pełny rekord PHOTO używany przez finalizer
wymaga pól i powiązań sprawdzanych w `scripts/photo/admission_policy.py`
(`references`, `rows`, progi SAM, receptura, rzeczywiste statusy przeglądu).
Wiązania eksperymentu i jobs muszą obejmować manifesty wejść, wagi, checkpointy,
kod i protokół oczekiwane w runnerach oraz scorerach. Nie zastępuj brakujących
historycznych decyzji domyślnym zaakceptowaniem wszystkich obrazów. Odtworzenie
takich zapisów dla własnych danych daje nowy przebieg, bez gwarancji tożsamości
ze starymi tabelami.

Historyczny TRELLIS nie ma pełnego przypięcia wersji upstream, poleceń i ziaren
generatora zasobów; sam podział B1 nie odtwarza oryginalnych GLB. Odzyskany
generator SAPIEN, wrappery TRELLIS i wspólny trening Ray pochodzą z archiwum
źródeł. Pozostałe obliczenia pochodzą z HEAD
`c033e1b653c3625977f90acfa6d423dc89604a24` źródłowego repozytorium.

Testy CPU, kompilacja, importy i `--help` nie potwierdzają odtworzenia wyników
na GPU. Dostarczone testy nie potrzebują GPU ani zbiorów; znaczniki `gpu` i `data`
są zadeklarowane dla takich testów integracyjnych. Pełny przebieg wymaga zasobów
zewnętrznych i wymienionych manifestów.

## Licencje i pochodzenie

Nie wybrano licencji tego repozytorium. Autor musi ją rozstrzygnąć.
Zewnętrzny kod i wagi pobierz osobno, zachowując ich warunki.

RaySt3R ma własną licencję akademicką do badań niekomercyjnych:
[licencja upstream](https://github.com/Duisterhof/rayst3r/blob/main/LICENSE).
OctMAE deklaruje CC BY-NC 4.0:
[licencja upstream](https://github.com/TRI-ML/OctMAE/blob/main/LICENSE.md).
TRELLIS.2 deklaruje MIT:
[licencja upstream](https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE).
Warunki wag FLUX, RMBG i modeli Meta sprawdź w kartach modeli; licencja kodu
nie zastępuje licencji wag ani danych.

Nie skopiowano implementacji upstream. Zachowane adaptery i poprawki
wykonania wymagają kontroli granic pochodzenia: `scripts/common/ray_inference.py`,
`scripts/train/rgb_context.py`, `scripts/common/ray_view_chunks.py`,
`scripts/train/confidence_exponential.py` oraz kompatybilność DINOv3 w
`scripts/trellis_b1/generate_meshes.py`. W odczytanych kopiach tych plików nie było
nagłówków licencyjnych, które można zachować przez kopiowanie.
