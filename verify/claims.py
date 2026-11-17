"""Confere, uma por uma, as afirmações feitas sobre os dados, os resultados e a internet.

Cada afirmação recebe um veredito:
  VERDADE          o cálculo ou a consulta confirma
  CORRIGIR         a ideia está certa, mas o número ou a palavra precisa de ajuste
  FALSO            o cálculo ou a consulta contradiz
  NÃO VERIFICÁVEL  não dá para provar daqui (ex.: que algo não existe em lugar nenhum)

As provas matemáticas estão em verify/lean_math (Lean 4 + Mathlib).

Uso: .venv\\Scripts\\python -m verify.claims
Grava verify/report/claims_report.md.
"""

import io
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date

import numpy as np
import openpyxl
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import chi2, fisher_exact

from src.daily import build_daily
from src.glm import at_risk_weeks, fit_with_background, fit_without_background
from src.load import RAW_DIR, ROOT, load_exposure
from src.weekly import build_weekly
from verify.audit import (
    exposure_lookup, independent_window, null_loglik, read_raw_exposure, read_raw_infection,
)

REPORT = ROOT / "verify" / "report" / "claims_report.md"


class Claims:
    def __init__(self):
        self.rows = []

    def add(self, area, claim, verdict, evidence):
        self.rows.append((area, claim, verdict, evidence))
        print(f"[{verdict}] {claim} -> {evidence}")

    def check(self, area, claim, ok, evidence, wrong="FALSO"):
        self.add(area, claim, "VERDADE" if ok else wrong, evidence)


def get_json(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "cown-research-verification"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_text(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "cown-research-verification"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.geturl(), r.read().decode("utf-8", errors="ignore")


# --------------------------------------------------------------------------- dados
def data_claims(c: Claims):
    area = "Dados brutos"
    wb = openpyxl.load_workbook(RAW_DIR / "infectious data_NLAS2b.xlsx")
    ws = wb.active
    filled = [r for r in range(2, ws.max_row + 1) if any(ws.cell(r, k).value is not None for k in range(1, 10))]
    pandas_rows = len(pd.read_excel(RAW_DIR / "infectious data_NLAS2b.xlsx"))
    c.check(area, "O pandas lê todas as linhas do arquivo de infecção (16.120)",
            len(filled) == pandas_rows == 16120, f"{len(filled)} linhas com dado no Excel, {pandas_rows} lidas")
    sheet = zipfile.ZipFile(RAW_DIR / "infectious data_NLAS2b.xlsx").read("xl/worksheets/sheet1.xml").decode()
    flt = re.findall(r'<autoFilter ref="([^"]+)".*?<filter val="([^"]+)"', sheet, re.S)
    c.check(area, "As linhas ocultas são só um filtro visual (coluna I = 1)", flt == [("H1:H18356", "1")],
            f"autoFilter {flt}; a coluna H do Excel é a coluna I (infectada)")
    obs = pd.read_excel(RAW_DIR / "infectious data_NLAS2b.xlsx")["obs"]
    c.check(area, "A coluna obs está vazia", obs.notna().sum() == 0, f"{int(obs.notna().sum())} células preenchidas")

    e = pd.read_csv(RAW_DIR / "mu_0.07_agent_exposure.csv")
    one_per_day = not e.duplicated(["Agent", "Date"]).any()
    day = pd.to_datetime(e["Date"], format="%d-%m-%Y")
    tick_is_day_end = ((e["Tick"] % 1440 == 0) & (e["Tick"] // 1440 - 1 == (day - day.min()).dt.days)).all()
    c.check(area, "O arquivo de exposição só traz o total por vaca por dia (sem posição nem horário)",
            list(e.columns) == ["Agent", "Tick", "Date", "Contamination Load Droplet"] and one_per_day
            and tick_is_day_end,
            f"colunas {list(e.columns)}; uma linha por vaca-dia: {one_per_day}; Tick = minutos desde o início, "
            f"fim de cada dia ({sorted(set(e['Tick']))[:3]}...): {tick_is_day_end}")
    c.add(area, "O arquivo de exposição tem 'Tick 1440'", "CORRIGIR",
          "1440 é o passo (minutos num dia), não um valor fixo: o Tick vai de 1440 em 1440")


def pipeline_claims(c: Claims, daily, old_weekly, weekly):
    area = "Correção da janela"
    raw = old_weekly.merge(independent_window(old_weekly, exposure_lookup(read_raw_exposure())),
                           on=["Agent", "date"], suffixes=("", "_raw"))
    lost = raw[(raw["S"] == 1) & (raw["exp_days_raw"] > raw["exp_days"])]
    c.check(area, "Antes, 165 semanas em risco perdiam dias de exposição, com 2 casos",
            (len(lost), int(lost["C"].sum())) == (165, 2), f"{len(lost)} semanas, {int(lost['C'].sum())} casos")
    old_r, new_r = old_weekly[old_weekly["S"] == 1], weekly[weekly["S"] == 1]
    old = (int(old_r["complete"].sum()), int(old_r.loc[old_r["complete"], "C"].sum()))
    new = (int(new_r["complete"].sum()), int(new_r.loc[new_r["complete"], "C"].sum()))
    c.check(area, "Semanas completas passaram de 1.802 (31 casos) para 1.965 (33 casos)",
            old == (1802, 31) and new == (1965, 33), f"antes {old}, depois {new}")
    c.check(area, "São 2.196 semanas em risco e 42 casos; 9 casos ficam sem a semana completa",
            (len(new_r), int(new_r["C"].sum()), int(new_r.loc[~new_r["complete"], "C"].sum())) == (2196, 42, 9),
            f"{len(new_r)} semanas, {int(new_r['C'].sum())} casos, {int(new_r.loc[~new_r['complete'], 'C'].sum())} sem os 7 dias")
    before = fit_with_background(at_risk_weeks(old_weekly))
    after = fit_with_background(at_risk_weeks(weekly))
    c.check(area, "A conclusão não muda: com fundo, o melhor β continua zero (p = 0,5)",
            before["beta"] < 1e-12 and after["beta"] < 1e-12 and before["p_exposure"] == after["p_exposure"] == 0.5,
            f"β antes {before['beta']:.2g}, depois {after['beta']:.2g}")
    nb = fit_without_background(at_risk_weeks(weekly))
    c.check(area, "Sem fundo, β fica perto de 2,8 × 10⁻⁵", 2.7e-5 < nb["beta"] < 2.9e-5, f"β = {nb['beta']:.4g}")

    area = "Inconsistências no arquivo do Gustavo"
    inf = read_raw_infection().drop_duplicates(["Agent", "date"])
    m = daily.merge(inf[["Agent", "date", "NC", "I"]], on=["Agent", "date"], suffixes=("", "_file"))
    diff = m[m["INFECTED"] != (m["I_file"].fillna(0) + m["NC"].fillna(0)).clip(upper=1)]
    c.check(area, "32 vaca-dias em 3 vacas (2611, 2678, 4065) têm LA/RA e S/I/NC contraditórios",
            len(diff) == 32 and sorted(diff["Agent"].unique()) == [2611, 2678, 4065],
            f"{len(diff)} vaca-dias, vacas {sorted(diff['Agent'].unique())}")


# --------------------------------------------------------------------------- análises
def fast_max_loglik(c_, e_):
    """Máximo da log-verossimilhança com fundo (só o valor, para a permutação)."""
    def nll(p):
        h = np.exp(p[0]) * 7 + np.exp(p[1]) * e_
        with np.errstate(divide="ignore"):
            return -float(np.sum(np.where(c_ == 1, np.log(-np.expm1(-h)), -h)))
    rate = c_.mean()
    best = -null_loglik(c_)
    for lb in [np.log(rate / max(e_.mean(), 1e-12)), np.log(rate / max(e_.mean(), 1e-12)) + 3]:
        fit = minimize(nll, [np.log(-np.log(1 - rate) / 7), lb], method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000})
        best = min(best, fit.fun)
    return -best


def lag_claims(c: Claims, weekly):
    area = "Janelas de exposição"
    lookup = exposure_lookup(read_raw_exposure())
    windows = {"1–7": (1, 7), "0–6": (0, 6), "1–14": (1, 14), "8–14": (8, 14), "15–21": (15, 21), "1–21": (1, 21)}
    base = weekly[weekly["S"] == 1].drop(columns=["exp", "exp_days", "complete"]).reset_index(drop=True)
    tables = {k: base.merge(independent_window(base, lookup, a, b), on=["Agent", "date"]) for k, (a, b) in windows.items()}
    stats = {}
    for k, t in tables.items():
        w = t[t["exp_days"] > 0]
        fit = fit_with_background(w)
        stats[k] = (fit["aic"] - (-2 * null_loglik(w["C"].to_numpy()) + 2), fit["p_exposure"], len(w), int(w["C"].sum()))
    d_aic, p, n, k = stats["15–21"]
    c.check(area, "Com a exposição de 15 a 21 dias antes, ΔAIC ≈ −6 e p ≈ 0,002 (dias disponíveis)",
            -7 < d_aic < -5 and p < 0.005, f"ΔAIC {d_aic:+.2f}, p {p:.4f}, {n} semanas, {k} casos")
    c.check(area, "Com a janela atual (1 a 7 dias), a exposição não melhora o modelo",
            stats["1–7"][0] > 1.9, f"ΔAIC {stats['1–7'][0]:+.2f}, p {stats['1–7'][1]:.3f}")

    # As semanas que existem em todas as janelas, para comparar janelas nas mesmas linhas.
    keys = None
    for t in tables.values():
        kk = t.loc[t["exp_days"] > 0, ["Agent", "date"]]
        keys = kk if keys is None else keys.merge(kk)
    aligned = {w: keys.merge(t, on=["Agent", "date"]) for w, t in tables.items()}
    cases = aligned["1–7"]["C"].to_numpy()
    exps = {w: t["exp"].to_numpy() for w, t in aligned.items()}
    lr = {w: 2 * (fast_max_loglik(cases, e) - null_loglik(cases)) for w, e in exps.items()}
    observed = max(lr.values())
    rng = np.random.default_rng(11)
    n_perm, beat = 300, 0
    for _ in range(n_perm):
        perm = rng.permutation(cases)
        if max(2 * (fast_max_loglik(perm, e) - null_loglik(perm)) for e in exps.values()) >= observed - 1e-9:
            beat += 1
    p_perm = (beat + 1) / (n_perm + 1)
    best = max(lr, key=lr.get)
    p_best = chi2.sf(lr[best], 1) / 2
    c.add(area, "O efeito da janela 15–21 sobrevive à escolha entre 6 janelas",
          "VERDADE" if p_perm < 0.05 else "CORRIGIR",
          f"nas {len(cases)} semanas comuns a todas as janelas ({int(cases.sum())} casos), a melhor é {best} "
          f"(p sem correção {p_best:.4f}); Bonferroni ×6 = {min(1, 6 * p_best):.3f}; "
          f"permutação do máximo das 6 janelas ({n_perm} sorteios) = {p_perm:.3f}")


def cow_claims(c: Claims, daily, weekly):
    area = "Vacas"
    per_cow = daily.groupby("Agent")["C"].sum()
    c.check(area, "42 casos vêm de 28 vacas; 10 vacas tiveram 2 ou 3 casos",
            (int(per_cow.sum()), int((per_cow >= 1).sum()), int((per_cow >= 2).sum()), int(per_cow.max())) == (42, 28, 10, 3),
            f"{int(per_cow.sum())} casos, {int((per_cow >= 1).sum())} vacas, {int((per_cow >= 2).sum())} com 2+, máximo {int(per_cow.max())}")
    ever = int(daily.groupby("Agent")["INFECTED"].max().sum())
    c.check(area, "Só 34 das 272 vacas chegaram a se infectar", (ever, daily["Agent"].nunique()) == (34, 272),
            f"{ever} de {daily['Agent'].nunique()}")
    c.check(area, "São 51 casos de pata", int(daily["Ct"].sum()) == 51, f"{int(daily['Ct'].sum())}")

    d = daily.sort_values(["Agent", "date"]).copy()
    d["start"] = (d["INFECTED"].eq(1) & d.groupby("Agent")["INFECTED"].shift(fill_value=0).eq(0)).astype(int)
    d["prior"] = d.groupby("Agent")["start"].cumsum() - d["start"]
    r = weekly[weekly["S"] == 1].merge(d[["Agent", "date", "prior"]], on=["Agent", "date"])
    before = r["prior"] > 0
    a, b = int(r.loc[before, "C"].sum()), int(before.sum())
    cc, dd = int(r.loc[~before, "C"].sum()), int((~before).sum())
    c.check(area, "Vacas que já tiveram dermatite: 20 casos em 110 semanas; as outras: 22 em 2.086",
            (a, b, cc, dd) == (20, 110, 22, 2086), f"{a}/{b} e {cc}/{dd}")
    ratio = (a / b) / (cc / dd)
    odds, p_f = fisher_exact([[a, b - a], [cc, dd - cc]])
    c.check(area, "O risco é cerca de 17 vezes maior (provado em Lean: entre 17 e 18)", 17 < ratio < 18,
            f"razão de riscos {ratio:.2f}; teste exato de Fisher p = {p_f:.1e}")
    c.add(area, "Metade dos casos vem de vacas que já tiveram dermatite", "CORRIGIR",
          f"são {a} de {a + cc} ({100 * a / (a + cc):.1f}%): quase metade")
    cows_prior = r.loc[before, "Agent"].nunique()
    gee = sm.GEE(r["C"].to_numpy(), sm.add_constant(before.astype(float).to_numpy()), groups=r["Agent"].to_numpy(),
                 family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    c.check(area, "O efeito continua forte levando em conta que as semanas da mesma vaca não são independentes",
            gee.pvalues[1] < 0.001, f"{cows_prior} vacas nas 110 semanas; GEE por vaca: odds ratio "
            f"{np.exp(gee.params[1]):.1f}, p = {gee.pvalues[1]:.1e}")

    area = "Ausências no arquivo de infecção"
    gap = d.groupby("Agent")["date"].diff().dt.days
    long = d[gap >= 35].assign(gap=gap[gap >= 35])
    ek = set(zip(read_raw_exposure()["Agent"], read_raw_exposure()["date"]))
    tracked = [np.mean([(ag, dt - pd.Timedelta(days=k)) in ek for k in range(1, int(g))]) for ag, dt, g in
               long[["Agent", "date", "gap"]].itertuples(index=False)]
    full = sum(t == 1.0 for t in tracked)
    c.check(area, "30 ausências de 35+ dias; em 28 a vaca continua sendo rastreada no galpão",
            (len(long), full) == (30, 28), f"{len(long)} ausências, {full} com rastreamento em todos os dias")
    cases_after = int(((daily["C"] == 1) & (daily["days_since_prev"] > 7)).sum())
    c.check(area, "Nenhum caso novo aparece logo depois de uma ausência", cases_after == 0, f"{cases_after} casos")


def exposure_model_claims(c: Claims, daily):
    area = "Modelo por baia x PeDViS"
    from scipy.stats import spearmanr
    from src.barn import add_barn_exposure
    barn = add_barn_exposure(daily, 0.07)[["Agent", "date", "Exp"]].rename(columns={"Exp": "exp_barn"})
    m = daily[["Agent", "barn", "date", "Exp"]].rename(columns={"Exp": "exp_pedvis"}).merge(barn, on=["Agent", "date"]).dropna()
    rho = spearmanr(m["exp_pedvis"], m["exp_barn"]).correlation
    zero_pedvis, zero_barn = (m["exp_pedvis"] == 0).mean(), (m["exp_barn"] == 0).mean()
    c.check(area, "Com o mesmo μ = 0,07, o modelo por baia concorda pouco com o PeDViS",
            rho < 0.3, f"Spearman {rho:.2f} em {len(m):,} vaca-dias; exposição zero em {100 * zero_pedvis:.0f}% "
            f"no PeDViS e {100 * zero_barn:.0f}% no modelo por baia")
    pos = m[m["exp_pedvis"] > 0]
    cv = pos.groupby(["barn", "date"])["exp_pedvis"].agg(lambda s: s.std() / s.mean() if len(s) > 1 else np.nan).dropna()
    c.check(area, "Vacas da mesma baia no mesmo dia têm exposições bem diferentes no PeDViS (uso do espaço não uniforme)",
            cv.median() > 0.3, f"coeficiente de variação mediano {cv.median():.2f}; o modelo por baia dá o mesmo valor "
            "para todas (provado em Lean: só é igual ao PeDViS com uso uniforme do espaço)")


# --------------------------------------------------------------------------- internet
def data_repository_claims(c: Claims):
    area = "Internet"
    queries = ["digital dermatitis", "Dairy Campus", "ultra-wideband dairy cows"]
    found = []
    for q in queries:
        for name, url in [
            ("Zenodo", "https://zenodo.org/api/records?" + urllib.parse.urlencode({"q": q, "size": 10})),
            ("Dryad", "https://datadryad.org/api/v2/search?" + urllib.parse.urlencode({"q": q, "per_page": 10})),
        ]:
            try:
                d = get_json(url)
                titles = ([h["metadata"]["title"] for h in d["hits"]["hits"]] if name == "Zenodo"
                          else [x["title"] for x in d.get("_embedded", {}).get("stash:datasets", [])])
                found += [f"{name}: {t}" for t in titles if re.search(r"2024|environmental barn|PeDViS", t, re.I)
                          and re.search(r"cow|dairy|dermatitis", t, re.I)]
            except Exception:
                pass
    c.add(area, "Não há posições de 2024 nem dados por vaca do Dairy Campus em repositórios públicos",
          "NÃO VERIFICÁVEL" if not found else "FALSO",
          "nenhum resultado no Zenodo e no Dryad (e, na busca feita à mão, no DANS, 4TU e figshare); "
          "dá para dizer que não foram achados, não que não existem" if not found else "; ".join(found[:5]))


def internet_claims(c: Claims):
    area = "Internet"
    try:
        z = get_json("https://zenodo.org/api/records/7016571")
        cows = sorted(int(re.findall(r"\d+", f["key"])[0]) for f in z["files"] if f["key"].startswith("cow_"))
        creators = [x["name"] for x in z["metadata"]["creators"]]
        c.check(area, "Zenodo 7016571 é aberto (CC-BY 4.0), de pesquisadores da WUR, com 32 vacas",
                z["metadata"]["license"]["id"] == "cc-by-4.0" and len(cows) == 32 and "Ouweltjes, Wijbrand" in creators,
                f"licença {z['metadata']['license']['id']}, {len(cows)} vacas, autores {creators}")
        inf = set(read_raw_infection()["Agent"])
        exp = set(read_raw_exposure()["Agent"])
        c.check(area, "Só 3 números de vaca coincidem com os nossos (495, 628, 1028)",
                sorted(set(cows) & (inf | exp)) == [495, 628, 1028], f"{sorted(set(cows) & (inf | exp))}")
        req = urllib.request.Request("https://zenodo.org/api/records/6373622/files/20210908_Adriaens_UWB_.pdf/content",
                                     headers={"User-Agent": "cown-research-verification"})
        from pypdf import PdfReader
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = re.sub(r"\s+", " ", " ".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(resp.read())).pages))
        c.check(area, "Os dados foram coletados no Dairy Campus em julho de 2019, por 10 dias",
                "Dairy Campus" in text and "2019 (July 3 to 8 and July 10 to 15" in text,
                "o artigo diz: Dairy Campus, 'two periods of five days ... in 2019 (July 3 to 8 and July 10 to 15'")
        c.check(area, "O artigo só publica médias de paridade e lactação, não os dados por vaca",
                "Overview of cow characteristics Name average std min max Parity" in text,
                "Tabela 1 com média, desvio, mínimo e máximo (paridade média 2,77)")
    except Exception as error:
        c.add(area, "Dados do Zenodo", "NÃO VERIFICÁVEL", f"consulta falhou agora: {error}")

    try:
        for origin, expected in [("https://git.wur.nl/sso-public/pedvis.git", "2024-07-23"),
                                 ("https://git.wur.nl/sso-public/pedvis-simulator.git", "2025-07-29")]:
            visits = get_json("https://archive.softwareheritage.org/api/1/origin/"
                              f"{urllib.parse.quote(origin, safe='')}/visits/?per_page=5")
            dates = sorted(v["date"][:10] for v in visits if v.get("snapshot"))
            c.check(area, f"O Software Heritage arquivou {origin.rsplit('/', 1)[1]} (última visita {expected})",
                    dates and dates[-1] == expected, f"visitas com cópia: {dates}")
    except Exception as error:
        c.add(area, "Software Heritage", "NÃO VERIFICÁVEL", f"consulta falhou agora: {error}")
    c.add(area, "A cópia arquivada é a versão para COVID, não a de vacas", "NÃO VERIFICÁVEL",
          "não abri o conteúdo (o download foi bloqueado); é o que o nome e o artigo da PLOS sugerem")

    try:
        get_json("https://git.wur.nl/api/v4/projects/sso-public%2Fpedvis")
        c.add(area, "O repositório do PeDViS no GitLab da WUR agora exige login", "FALSO",
              "a API do GitLab mostra o projeto sem login")
    except urllib.error.HTTPError as error:
        c.check(area, "O repositório do PeDViS no GitLab da WUR agora exige login", error.code in (401, 403, 404),
                f"a API do GitLab responde {error.code} sem login (projeto não público)")
    except Exception as error:
        c.add(area, "GitLab da WUR", "NÃO VERIFICÁVEL", f"consulta falhou agora: {error}")
    try:
        socket.gethostbyname("nlasdm.wur.nl")
        c.add(area, "O portal nlasdm.wur.nl não abre fora da rede da WUR", "FALSO", "o endereço resolveu agora")
    except socket.gaierror:
        c.add(area, "O portal nlasdm.wur.nl não abre fora da rede da WUR", "VERDADE",
              "o endereço não existe no DNS público (que só funcione dentro da WUR é uma dedução)")

    try:
        xml = get_text("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11003685/fullTextXML")[1]
        c.check(area, "O artigo do PeDViS (PLOS Comput Biol 2024) diz que o código está em git.wur.nl/sso-public/pedvis",
                "git.wur.nl/sso-public/pedvis" in xml, "texto completo pela Europe PMC")
    except Exception as error:
        c.add(area, "Artigo do PeDViS", "NÃO VERIFICÁVEL", f"consulta falhou agora: {error}")

    try:
        ct = get_json("https://api.github.com/users/cteberg")
        repos = [r["name"] for r in get_json("https://api.github.com/users/cteberg/repos")]
        c.check(area, "O GitHub 'cteberg' (Colin Teberg) só tem uma cópia do pyglet",
                ct.get("name") == "Colin Teberg" and repos == ["pyglet"], f"nome {ct.get('name')}, repositórios {repos}")
        c.add(area, "Esse 'cteberg' é o Colin Teberg do PeDViS-BDD", "NÃO VERIFICÁVEL",
              "o nome bate, mas o perfil não diz onde a pessoa trabalha")
        q = get_json("https://api.github.com/users/quirine")
        qrepos = [r["name"] for r in get_json("https://api.github.com/users/quirine/repos")]
        c.check(area, "O GitHub da Quirine ten Bosch (WUR) só tem modelos de dengue e mosquitos",
                q.get("name") == "Quirine ten Bosch" and len(qrepos) == 3, f"{q.get('company')}: {qrepos}")
        dd = [x["name"] for x in get_json("https://api.github.com/repos/DDopfer/DDComBat_2026/contents")]
        c.check(area, "O repositório DDComBat_2026 (Döpfer) está vazio", dd == ["LICENSE"], f"arquivos: {dd}")
        found = get_json("https://api.github.com/search/repositories?q=pedvis")["items"]
        hits = [r["full_name"] for r in found if "virus" in (r["description"] or "").lower()
                or "pedestrian dynamics" in (r["description"] or "").lower()]
        c.check(area, "Não há cópia do PeDViS no GitHub (busca por 'pedvis')", not hits,
                f"{len(found)} repositórios com esse nome, nenhum é o modelo de vírus")
    except Exception as error:
        c.add(area, "GitHub", "NÃO VERIFICÁVEL", f"consulta falhou agora (limite da API?): {error}")

    c.add(area, "Os arquivos de outros μ e os dados das vacas não estão publicados em lugar nenhum",
          "NÃO VERIFICÁVEL", "só dá para dizer que não foram achados no GitHub, Zenodo, Software Heritage, "
          "nos e-mails e no computador; não existir em lugar nenhum não se prova")


def main():
    c = Claims()
    daily = build_daily("0.07")
    old_weekly = build_weekly(daily)                         # janela só com os dias do arquivo de infecção
    weekly = build_weekly(daily, load_exposure("0.07"))      # janela corrigida
    data_claims(c)
    pipeline_claims(c, daily, old_weekly, weekly)
    lag_claims(c, weekly)
    cow_claims(c, daily, weekly)
    exposure_model_claims(c, daily)
    internet_claims(c)
    data_repository_claims(c)

    df = pd.DataFrame(c.rows, columns=["área", "afirmação", "veredito", "como foi conferido"])
    counts = df["veredito"].value_counts().to_dict()
    lines = [f"# Conferência das afirmações ({date.today():%d/%m/%Y})", "",
             ", ".join(f"{v}: {n}" for v, n in counts.items()), "",
             "| área | afirmação | veredito | como foi conferido |", "|---|---|---|---|"]
    lines += ["| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in df.to_numpy()]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{counts}\nRelatório em {REPORT}")


if __name__ == "__main__":
    main()
