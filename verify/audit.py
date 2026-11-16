"""Auditoria independente do pipeline (dados brutos -> diário -> semanal -> GLM).

Cada verificação refaz o cálculo por outro caminho (laços simples, statsmodels,
integração numérica) e compara com o resultado de src/. Nada em data/raw é escrito.

Uso: .venv\\Scripts\\python -m verify.audit
Grava o relatório em verify/report/audit_report.md (fora do git, tem IDs de vacas).
"""

import hashlib
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.integrate import solve_ivp

from src.barn import add_barn_exposure, barn_exposure_series, shedding_rate
from src.daily import add_status, build_daily
from src.glm import at_risk_weeks, fit_with_background, fit_without_background
from src.load import RAW_DIR, ROOT, load_exposure
from src.weekly import ASSESSMENT_WEEKDAY, build_weekly

# sha256 dos anexos do e-mail "working files" do Gustavo (16/10/2026), conferidos
# contra o workingfiles.zip baixado do Gmail em 22/10/2026
RAW_SHA256 = {
    "daily000000007.xlsx": "cee829441e1df21ed1ceea27c6678caa5f1759a467284169ea34c6da28c183b8",
    "infectious data_NLAS2b.xlsx": "fb9904ff451e9ab262a39b8c7cee609adb3e43c946b65062cbc1881f91190c39",
    "mu_0.07_agent_exposure.csv": "2afd9a2119723b857a3ef3ce931e7c371451b6e3852d7139e34201662c23c0d2",
}
EMAIL_ZIP = Path.home() / "Downloads" / "workingfiles.zip"
REPORT = ROOT / "verify" / "report" / "audit_report.md"


class Report:
    def __init__(self):
        self.lines, self.failures, self.warnings = [], [], []

    def section(self, title):
        self.lines += ["", f"## {title}", ""]

    def check(self, name, ok, detail=""):
        self.lines.append(f"- {'OK' if ok else '**FALHOU**'}: {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            self.failures.append(name)

    def warn(self, name, ok, detail=""):
        """Algo nos dados que não é erro do código, mas precisa de resposta do Gustavo."""
        self.lines.append(f"- {'OK' if ok else '**ATENÇÃO**'}: {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            self.warnings.append(name)

    def note(self, text):
        self.lines += ["", text, ""]

    def table(self, df):
        df = df.map(lambda v: "" if pd.isna(v) else str(v))
        self.lines += ["", "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        self.lines += ["| " + " | ".join(row) + " |" for row in df.to_numpy()]
        self.lines.append("")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- 1
def check_raw_files(r: Report):
    r.section("1. Dados brutos")
    for name, expected in RAW_SHA256.items():
        path = RAW_DIR / name
        r.check(f"`{name}` igual ao anexo do e-mail", path.exists() and sha256(path.read_bytes()) == expected)
    if EMAIL_ZIP.exists():
        with zipfile.ZipFile(EMAIL_ZIP) as z:
            names = z.namelist()
            same = all(sha256(z.read(n)) == RAW_SHA256.get(n) for n in names)
        r.check("`workingfiles.zip` (Downloads) tem os mesmos três arquivos", same and len(names) == 3)
    sources = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "src").glob("*.py")}
    writes = [
        f"{name}:{i}" for name, text in sources.items()
        for i, line in enumerate(text.splitlines(), 1)
        if re.search(r"\.to_(csv|excel|parquet)\(|open\(|unlink|rmtree|os\.remove", line)
    ]
    raw_users = sorted(name for name, text in sources.items() if "RAW_DIR" in text and name != "load.py")
    r.check("só src/load.py usa data/raw, e só para ler", not raw_users, ", ".join(raw_users))
    r.note(f"Linhas que gravam arquivos em src/: {', '.join(writes)}. Todas gravam em data/processed.")


# --------------------------------------------------------------------------- 2
def read_raw_infection():
    raw = pd.read_excel(RAW_DIR / "infectious data_NLAS2b.xlsx")
    return raw.rename(columns={"ID": "Agent", "Barn": "barn"})


def read_raw_exposure():
    """Lido de novo, com o valor como texto, para comparar sem arredondamento."""
    raw = pd.read_csv(RAW_DIR / "mu_0.07_agent_exposure.csv", dtype={"Contamination Load Droplet": str})
    raw["date"] = pd.to_datetime(raw["Date"], format="%d-%m-%Y")
    raw["Exp"] = [float(v) for v in raw["Contamination Load Droplet"]]
    return raw


def check_accounting(r: Report, daily: pd.DataFrame, weekly: pd.DataFrame):
    r.section("2. Nenhuma linha some sem explicação")
    inf = read_raw_infection()
    exact_dups = int(inf[["Agent", "barn", "date", "LA", "RA"]].duplicated().sum())
    cow_days = inf[["Agent", "date"]].drop_duplicates()
    r.note(f"Arquivo de infecção: {len(inf):,} linhas, {inf['Agent'].nunique()} vacas, "
           f"{len(cow_days):,} vaca-dias, {exact_dups:,} linhas repetidas idênticas.")

    raw_keys = set(map(tuple, cow_days.to_numpy()))
    daily_keys = set(zip(daily["Agent"], daily["date"]))
    r.check("todo vaca-dia do arquivo de infecção está no diário, e só eles",
            raw_keys == daily_keys, f"faltam {len(raw_keys - daily_keys)}, sobram {len(daily_keys - raw_keys)}")
    r.check("uma linha por vaca-dia no diário", not daily.duplicated(["Agent", "date"]).any())

    legs = inf.drop_duplicates(["Agent", "date", "LA", "RA"])
    r.check("LA/RA únicos por vaca-dia no bruto", not legs.duplicated(["Agent", "date"]).any())
    m = legs.merge(daily[["Agent", "date", "LA", "RA"]], on=["Agent", "date"], suffixes=("", "_d"))
    r.check("LA e RA do diário iguais aos do bruto", ((m["LA"] == m["LA_d"]) & (m["RA"] == m["RA_d"])).all())
    barns = inf.groupby(["Agent", "date"])["barn"].agg(set)
    in_raw = [b in barns[(a, d)] for a, d, b in zip(daily["Agent"], daily["date"], daily["barn"])]
    r.check("a baia de cada vaca-dia é uma das baias do bruto", all(in_raw))
    r.note(f"Dias de troca de baia (vaca aparece em duas baias): {int(daily['barn_move'].sum())}.")

    exp = read_raw_exposure()
    dup_exp = int(exp.duplicated(["Agent", "date"]).sum())
    r.check("arquivo de exposição sem vaca-dia repetido", dup_exp == 0, f"{dup_exp} repetidos")
    inf_cows, exp_cows = set(inf["Agent"]), set(exp["Agent"])
    only_exp, only_inf = sorted(exp_cows - inf_cows), sorted(inf_cows - exp_cows)
    matched = exp.merge(daily[["Agent", "date", "Exp"]], on=["Agent", "date"], how="inner", suffixes=("", "_d"))
    n_only_exp = int(exp["Agent"].isin(only_exp).sum())
    outside = int(exp["Agent"].isin(inf_cows).sum()) - len(matched)
    r.note(f"Arquivo de exposição: {len(exp):,} linhas, {len(exp_cows)} vacas.")
    r.table(pd.DataFrame([
        ("linhas de exposição usadas no diário", f"{len(matched):,}"),
        (f"linhas de vacas que só estão na exposição ({len(only_exp)} vacas)", f"{n_only_exp:,}"),
        ("linhas de vacas com status, em dias sem status", f"{outside:,}"),
        ("vacas que só estão no arquivo de infecção", f"{len(only_inf)}"),
        ("vaca-dias do diário sem exposição (Exp vazio)", f"{int(daily['Exp'].isna().sum()):,}"),
    ], columns=["contagem", "valor"]))
    r.check("soma confere: usadas + só exposição + fora dos dias = total",
            len(matched) + n_only_exp + outside == len(exp))
    r.check("Exp do diário igual, bit a bit, ao texto do arquivo bruto",
            np.array_equal(matched["Exp"].to_numpy(), matched["Exp_d"].to_numpy()))

    thursdays = daily[daily["date"].dt.dayofweek.eq(ASSESSMENT_WEEKDAY) & ~daily["first_day"]]
    r.check("toda quinta (fora o 1º dia da vaca) vira uma linha da tabela semanal", len(thursdays) == len(weekly))
    off_day_cases = daily[(daily["C"] == 1) & daily["date"].dt.dayofweek.ne(ASSESSMENT_WEEKDAY)]
    r.check("nenhum caso novo fora de quinta", off_day_cases.empty, f"{len(off_day_cases)} casos")
    r.check("todos os casos novos do diário estão na tabela semanal",
            int(daily["C"].sum()) == int(weekly["C"].sum()), f"{int(daily['C'].sum())} casos")


# --------------------------------------------------------------------------- 3
def check_rules_against_example(r: Report):
    r.section("3. Regras de status contra o exemplo do Gustavo")
    ex = pd.read_excel(RAW_DIR / "daily000000007.xlsx")
    r.note(f"`daily000000007.xlsx`: {len(ex)} linhas, {ex['Agent'].nunique()} vacas, de "
           f"{ex['date'].min():%d/%m/%Y} a {ex['date'].max():%d/%m/%Y}. É de 2023 e os dados são de 2024, "
           "então ele só confere as regras, não os valores.")
    ours = add_status(ex[["Agent", "barn", "date", "LA", "RA"]].copy())
    theirs = ex.sort_values(["Agent", "date"], kind="stable").reset_index(drop=True)
    # a primeira linha de cada vaca no exemplo não tem dia anterior; no código ela nunca é caso novo
    later = ~ours["first_day"].to_numpy()
    for col in ["INFECTED", "C", "I", "S", "N", "Ct"]:
        given = theirs[col].notna().to_numpy()  # o exemplo deixa algumas células de C vazias
        diff = given & (ours[col].astype(float).to_numpy() != theirs[col].astype(float).to_numpy())
        r.check(f"coluna {col} igual à do exemplo, fora a 1ª linha de cada vaca", not (diff & later).any(),
                f"{int((diff & later).sum())} diferenças; {int((diff & ~later).sum())} na 1ª linha")
    first_case = theirs[~later & (theirs["C"] == 1)]
    r.note(f"No exemplo, {len(first_case)} vaca(s) já aparece(m) infectada(s) na 1ª linha e conta(m) como caso novo "
           f"({', '.join(map(str, first_case['Agent']))}). No código, a 1ª linha nunca é caso novo, porque não há "
           "dia anterior (é a pendência 4 do README). Nos dados de 2024, isso só afeta vacas sem semana em risco "
           "antes, então não muda a regressão.")


# --------------------------------------------------------------------------- 4
def check_against_file_columns(r: Report, daily: pd.DataFrame, weekly: pd.DataFrame):
    r.section("4. Status recalculado contra as colunas NC/S/I do próprio arquivo")
    inf = read_raw_infection().drop_duplicates(["Agent", "date"]).sort_values(["Agent", "date"])
    m = daily.merge(inf[["Agent", "date", "NC", "S", "I"]], on=["Agent", "date"], suffixes=("", "_file"))
    # no arquivo, NC marca a semana depois de um caso e I o resto do tempo infectada
    infected_file = (m["I_file"].fillna(0) + m["NC"].fillna(0)).clip(upper=1).astype(int)
    diff = m[m["INFECTED"] != infected_file].assign(file_inf=infected_file)
    r.warn("o próprio arquivo concorda consigo mesmo: LA/RA contra as colunas S, I e NC", diff.empty,
           f"{len(diff)} vaca-dias em {diff['Agent'].nunique()} vacas, {int(diff['date'].dt.dayofweek.eq(3).sum())} "
           "deles em quintas")
    if not diff.empty:
        spans = (diff.groupby(["Agent", "LA", "RA", "file_inf"])["date"].agg(["min", "max", "size"]).reset_index())
        r.table(pd.DataFrame({
            "vaca": spans["Agent"], "de": spans["min"].dt.strftime("%d/%m/%Y"),
            "até": spans["max"].dt.strftime("%d/%m/%Y"), "dias": spans["size"],
            "LA/RA dizem": np.where(spans["LA"] | spans["RA"], "infectada", "saudável"),
            "S/I/NC dizem": np.where(spans["file_inf"] == 1, "infectada", "saudável"),
        }))
        r.note("O código usa LA/RA (decisão do README). Parece que as colunas S/I estão deslocadas alguns dias "
               "nessas vacas; vale perguntar ao Gustavo qual das duas está certa.")
    nc = m["NC"].fillna(0).astype(int)
    onset = nc.eq(1) & nc.groupby(m["Agent"]).shift(fill_value=0).eq(0)
    ours, file_onsets = m[m["C"] == 1], m[onset]
    a = set(zip(ours["Agent"], ours["date"]))
    b = set(zip(file_onsets["Agent"], file_onsets["date"]))
    only_ours, only_file = sorted(a - b), sorted(b - a)
    r.note(f"Casos novos: {len(a)} pelo recálculo (C) e {len(b)} começos de NC no arquivo; "
           f"{len(a & b)} nos dois, {len(only_ours)} só no recálculo, {len(only_file)} só na NC.")
    if only_ours or only_file:
        r.table(pd.DataFrame(
            [("só no recálculo", ag, f"{d:%d/%m/%Y}") for ag, d in only_ours]
            + [("só na NC", ag, f"{d:%d/%m/%Y}") for ag, d in only_file],
            columns=["onde", "vaca", "data"]))
    at_risk = set(zip(weekly.loc[weekly["S"] == 1, "Agent"], weekly.loc[weekly["S"] == 1, "date"]))
    r.check("nenhum caso que só a NC conta cai numa semana em risco (não mudariam a regressão)",
            not (set(only_file) & at_risk), "são as pendências 4 e 5 do README")


# --------------------------------------------------------------------------- 5
def independent_status(inf: pd.DataFrame) -> pd.DataFrame:
    """Status refeito com um laço simples por vaca, a partir do bruto."""
    rows = []
    days = inf.drop_duplicates(["Agent", "date"]).sort_values(["Agent", "date"])
    prev_agent, prev = None, None
    for agent, date, la, ra in days[["Agent", "date", "LA", "RA"]].itertuples(index=False):
        if agent != prev_agent:
            prev = None
        infected = int(la == 1 or ra == 1)
        if prev is None:
            c = ct = 0
        else:
            c = int(prev[2] == 0 and infected == 1)
            ct = int(la == 1 and prev[0] == 0) + int(ra == 1 and prev[1] == 0)
        at_risk = prev is not None and prev[2] == 0
        rows.append((agent, date, infected, c, int(infected and not c), 1 - infected, ct, at_risk))
        prev_agent, prev = agent, (la, ra, infected)
    return pd.DataFrame(rows, columns=["Agent", "date", "INFECTED", "C", "I", "S", "Ct", "at_risk"])


def exposure_lookup(table: pd.DataFrame) -> dict:
    return {(a, d): e for a, d, e in zip(table["Agent"], table["date"], table["Exp"]) if pd.notna(e)}


def independent_window(rows: pd.DataFrame, lookup: dict, first_lag=1, last_lag=7) -> pd.DataFrame:
    """Soma da exposição de `first_lag` a `last_lag` dias antes de cada linha, com dicionário."""
    out = []
    for a, d in zip(rows["Agent"], rows["date"]):
        values = [lookup[(a, d - pd.Timedelta(days=k))] for k in range(first_lag, last_lag + 1)
                  if (a, d - pd.Timedelta(days=k)) in lookup]
        out.append((a, d, sum(values) if values else np.nan, len(values)))
    return pd.DataFrame(out, columns=["Agent", "date", "exp", "exp_days"])


def check_reimplementation(r: Report, daily: pd.DataFrame, weekly: pd.DataFrame):
    r.section("5. Reimplementação independente (laços simples)")
    ind = independent_status(read_raw_infection())
    m = daily.merge(ind, on=["Agent", "date"], suffixes=("", "_ind"))
    r.check("mesmo número de vaca-dias", len(m) == len(daily) == len(ind))
    for col in ["INFECTED", "C", "I", "S", "Ct", "at_risk"]:
        diff = int((m[col].astype(int) != m[f"{col}_ind"].astype(int)).sum())
        r.check(f"{col} igual ao do laço", diff == 0, f"{diff} diferenças")
    # contada direto no arquivo de exposição, incluindo os dias em que a vaca some do
    # arquivo de infecção (até 16/11/2026 esses dias ficavam de fora)
    w = weekly.merge(independent_window(weekly, exposure_lookup(read_raw_exposure())),
                     on=["Agent", "date"], suffixes=("", "_ind"))
    r.check("dias com exposição na janela iguais à contagem no arquivo de exposição inteiro",
            (w["exp_days"] == w["exp_days_ind"]).all(), f"{int((w['exp_days'] != w['exp_days_ind']).sum())} diferentes")
    r.check("soma da exposição na janela igual (tolerância relativa 1e-12)",
            np.allclose(w["exp"].fillna(-1), w["exp_ind"].fillna(-1), rtol=1e-12, atol=0))
    s = weekly.merge(ind[["Agent", "date", "at_risk"]], on=["Agent", "date"])
    r.check("S da tabela semanal = vaca saudável no último dia observado", (s["S"] == s["at_risk"].astype(int)).all())


# --------------------------------------------------------------------------- 6
def check_barn_ode(r: Report):
    r.section("6. Exposição por baia contra integração numérica da EDO")
    r.note("dE/dt = φ(μ)·I(t) − μ·E, com a baia em equilíbrio no início; a exposição do dia é a integral "
           "de E no dia. Integrado com solve_ivp (rtol 1e-11), sem usar as fórmulas fechadas do código.")
    rng = np.random.default_rng(1)
    worst = 0.0
    for mu in [0.01, 0.03, 0.07, 0.5, 3.0, 10.0]:
        phi = shedding_rate(mu)
        infected = rng.integers(0, 12, 40).astype(float)
        contamination = phi * infected[0] / mu
        numeric = []
        for n in infected:
            sol = solve_ivp(lambda t, y: [phi * n - mu * y[0], y[0]], (0, 1), [contamination, 0.0],
                            rtol=1e-11, atol=1e-13, method="DOP853")
            contamination, integral = sol.y[:, -1]
            numeric.append(integral)
        code = barn_exposure_series(infected, mu)
        worst = max(worst, float(np.max(np.abs(code - numeric) / np.maximum(1, np.abs(numeric)))))
    r.check("barn_exposure_series igual à integração numérica para μ de 0,01 a 10", worst < 1e-7,
            f"maior erro relativo {worst:.1e}")


# --------------------------------------------------------------------------- 7
def loglik(lambda0, beta, c, exp):
    """Log-verossimilhança escrita de novo, direto da fórmula."""
    h = lambda0 * 7 + beta * exp
    with np.errstate(divide="ignore"):
        return float(np.sum(np.where(c == 1, np.log(-np.expm1(-h)), -h)))


def null_loglik(c):
    rate = c.mean()
    return c.sum() * np.log(rate) + (len(c) - c.sum()) * np.log(1 - rate)


def check_glm(r: Report, weekly: pd.DataFrame):
    r.section("7. Regressão: statsmodels, busca em grade e dados simulados")
    weeks = at_risk_weeks(weekly, complete=True)
    used = weeks[weeks["exp"] > 0]
    c, e = used["C"].to_numpy(), used["exp"].to_numpy()
    ours = fit_without_background(weeks)
    try:
        glm = sm.GLM(c, np.ones((len(c), 1)), offset=np.log(e),
                     family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
        beta_sm = float(np.exp(glm.params[0]))
        r.check("sem fundo: β igual ao GLM cloglog do statsmodels", np.isclose(beta_sm, ours["beta"], rtol=1e-4),
                f"src {ours['beta']:.4g}, statsmodels {beta_sm:.4g}")
        r.check("sem fundo: log-verossimilhança igual", np.isclose(glm.llf, ours["loglik"], rtol=1e-6, atol=1e-6),
                f"src {ours['loglik']:.4f}, statsmodels {glm.llf:.4f}")
    except Exception as error:  # o docstring de src/glm.py diz que pode estourar
        r.note(f"O statsmodels não conseguiu ajustar o modelo sem fundo: {error}")

    bg = fit_with_background(weeks)
    cw, ew = weeks["C"].to_numpy(), weeks["exp"].to_numpy()
    r.check("com fundo: log-verossimilhança do ótimo confere com a fórmula reescrita",
            np.isclose(loglik(bg["lambda0"], bg["beta"], cw, ew), bg["loglik"], rtol=1e-8, atol=1e-8))
    scale = max(ours["beta"], 1e-12)
    grid_best = max(
        loglik(l0, b, cw, ew)
        for l0 in np.geomspace(1e-6, 1e-1, 121)
        for b in np.concatenate([[0], np.geomspace(scale * 1e-4, scale * 1e4, 161)])
    )
    r.check("com fundo: nenhum ponto de uma grade 121 × 162 é melhor que o ótimo encontrado",
            grid_best <= bg["loglik"] + 1e-6, f"ótimo {bg['loglik']:.4f}, melhor da grade {grid_best:.4f}")
    r.check("com fundo: o ótimo não é pior que o modelo só com λ0, calculado de novo",
            bg["loglik"] >= null_loglik(cw) - 1e-9, f"só λ0 {null_loglik(cw):.4f}")

    # dados simulados com a exposição real: o ajuste recupera o β usado na simulação?
    rng = np.random.default_rng(7)
    true_beta = ours["beta"]
    estimates, covered = [], 0
    for _ in range(200):
        sim = used.assign(C=(rng.random(len(e)) < -np.expm1(-true_beta * e)).astype(int))
        if sim["C"].sum() == 0:
            continue
        fit = fit_without_background(sim)
        estimates.append(fit["beta"])
        covered += fit["beta_low"] <= true_beta <= fit["beta_high"]
    coverage = covered / len(estimates)
    r.check("sem fundo: com dados simulados, o IC de 95% contém o β verdadeiro em 90–99% das vezes",
            0.90 <= coverage <= 0.99, f"{100 * coverage:.0f}% de {len(estimates)} simulações; "
            f"mediana do β estimado / β verdadeiro = {np.median(estimates) / true_beta:.2f}")

    detected, beta_hat = [], []
    for _ in range(60):
        sim = weeks.assign(C=(rng.random(len(ew)) < -np.expm1(-(bg["lambda0"] * 7 + true_beta * ew))).astype(int))
        fit = fit_with_background(sim)
        detected.append(fit["p_exposure"] < 0.05)
        beta_hat.append(fit["beta"])
    r.note(f"Com fundo: simulando com λ0 = {bg['lambda0']:.3g} e β = {true_beta:.3g} (o β do modelo sem fundo), "
           f"o ajuste dá mediana de β estimado / β verdadeiro = {np.median(beta_hat) / true_beta:.2f}, e o teste "
           f"detecta o efeito da exposição em {100 * np.mean(detected):.0f}% das 60 simulações. Esse é o poder do "
           "teste com estes dados, se a exposição tivesse esse efeito.")


# --------------------------------------------------------------------------- 8
def window_weeks(weekly, lookup, first_lag, last_lag, complete):
    win = independent_window(weekly, lookup, first_lag, last_lag)
    w = weekly.drop(columns=["exp", "exp_days", "complete"]).merge(win, on=["Agent", "date"])
    w = w[w["S"].eq(1) & w["exp_days"].gt(0)]
    if complete:
        w = w[w["exp_days"].eq(last_lag - first_lag + 1)]
    return w.reset_index(drop=True)


def sensitivity(r: Report, daily: pd.DataFrame, weekly: pd.DataFrame):
    r.section("8. Outras escolhas razoáveis dariam um resultado melhor?")
    r.note("Cada linha muda uma escolha e ajusta o modelo com fundo. ΔAIC = AIC com exposição − AIC só com "
           "λ0 (risco constante): negativo favorece a exposição, abaixo de −2 é evidência. p = teste da razão "
           "de verossimilhança para β = 0. As janelas atrasadas testam a ideia de que a infecção acontece "
           "antes de ser detectada.")
    r.note("Cuidado: foram testadas várias janelas. Uma janela só deve ser escolhida se houver motivo "
           "biológico (por exemplo, o tempo entre a infecção e a lesão ser vista), e não por ter dado o menor p.")
    variants = [
        ("1–7 dias antes (escolha atual)", 1, 7),
        ("0–6 dias (inclui o dia da avaliação)", 0, 6),
        ("1–14 dias antes", 1, 14),
        ("8–14 dias antes (uma semana de atraso)", 8, 14),
        ("15–21 dias antes (duas semanas de atraso)", 15, 21),
        ("1–21 dias antes", 1, 21),
    ]
    barn_daily = add_barn_exposure(daily, 0.03)
    barn_weekly = build_weekly(barn_daily)
    rows = []
    for source, lookup, w, options in [
        ("PeDViS-BDD μ=0,07", exposure_lookup(read_raw_exposure()), weekly, [True, False]),
        ("por baia μ=0,03", exposure_lookup(barn_daily), barn_weekly, [True]),
    ]:
        for label, first, last in variants:
            for complete in options:
                weeks = window_weeks(w, lookup, first, last, complete)
                fit = fit_with_background(weeks)
                c = weeks["C"].to_numpy()
                rows.append((source, label, "todos os dias" if complete else "dias disponíveis",
                             len(weeks), int(c.sum()), f"{fit['beta']:.3g}",
                             f"{fit['aic'] - (-2 * null_loglik(c) + 2):+.2f}", f"{fit['p_exposure']:.3f}"))
    r.table(pd.DataFrame(rows, columns=["exposição", "janela", "semanas usadas", "n semanas", "casos", "β", "ΔAIC", "p"]))


def main():
    r = Report()
    daily = build_daily("0.07")
    weekly = build_weekly(daily, load_exposure("0.07"))
    check_raw_files(r)
    check_accounting(r, daily, weekly)
    check_rules_against_example(r)
    check_against_file_columns(r, daily, weekly)
    check_reimplementation(r, daily, weekly)
    check_barn_ode(r)
    check_glm(r, weekly)
    sensitivity(r, daily, weekly)
    status = [f"**Falharam: {len(r.failures)}.** " + "; ".join(r.failures) if r.failures
              else "**Nenhuma verificação falhou.**",
              f"**Atenção (dados, não código): {len(r.warnings)}.** " + "; ".join(r.warnings) if r.warnings else ""]
    r.lines = ["# Auditoria do pipeline (cown_research)", ""] + status + r.lines
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(r.lines) + "\n", encoding="utf-8")
    print("\n".join(r.lines))
    print(f"\nRelatório em {REPORT}")


if __name__ == "__main__":
    main()
