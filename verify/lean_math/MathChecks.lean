-- Só as partes da Mathlib que as provas usam (carregar a Mathlib inteira precisa de muita memória).
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# Provas das afirmações matemáticas sobre o projeto

`lake build` confere todas as provas. Os números que vêm dos dados (contagens de
semanas e casos) são conferidos em Python por `verify/claims.py`; aqui só se prova
a matemática e a aritmética feita com eles.
-/

open Real

namespace MathChecks

/-! ## 1. A regressão é o próprio modelo de transmissão

Se uma vaca saudável se infecta com taxa λ0 (por dia, fundo) + β·E (exposição E na
semana), a chance de virar caso na semana é p = 1 − e^(−(7·λ0 + β·E)). -/

noncomputable def weekProb (lam0 β E : ℝ) : ℝ := 1 - exp (-(lam0 * 7 + β * E))

/-- O link cloglog do GLM: cloglog(p) = log(−log(1 − p)). -/
noncomputable def cloglog (p : ℝ) : ℝ := log (-log (1 - p))

/-- O cloglog desfaz exatamente a probabilidade de infecção: cloglog(1 − e^(−h)) = log h. -/
theorem cloglog_of_hazard {h : ℝ} : cloglog (1 - exp (-h)) = log h := by
  simp [cloglog, Real.log_exp]

/-- Sem fundo, cloglog(p) = log β + log E: é um GLM binomial com link cloglog e
offset log(E), o modelo de Chang & de Jong (2023). -/
theorem glm_with_offset {β E : ℝ} (hβ : 0 < β) (hE : 0 < E) :
    cloglog (weekProb 0 β E) = log β + log E := by
  have h : (0 : ℝ) * 7 + β * E = β * E := by ring
  rw [weekProb, h, cloglog_of_hazard, Real.log_mul hβ.ne' hE.ne']

/-- A probabilidade está sempre entre 0 (inclusive) e 1 (exclusive). -/
theorem weekProb_between {lam0 β E : ℝ} (h0 : 0 ≤ lam0) (hβ : 0 ≤ β) (hE : 0 ≤ E) :
    0 ≤ weekProb lam0 β E ∧ weekProb lam0 β E < 1 := by
  unfold weekProb
  constructor
  · have : exp (-(lam0 * 7 + β * E)) ≤ 1 := by
      rw [Real.exp_le_one_iff]
      nlinarith [mul_nonneg hβ hE]
    linarith
  · linarith [exp_pos (-(lam0 * 7 + β * E))]

/-- Com β > 0, mais exposição dá sempre mais chance de virar caso. -/
theorem weekProb_increasing {lam0 β : ℝ} (hβ : 0 < β) : StrictMono (weekProb lam0 β) := by
  intro E₁ E₂ h
  unfold weekProb
  have hm := mul_lt_mul_of_pos_left h hβ
  have : exp (-(lam0 * 7 + β * E₂)) < exp (-(lam0 * 7 + β * E₁)) := by
    rw [Real.exp_lt_exp]
    linarith
  linarith

/-- Com β = 0 a exposição não muda nada: é o "risco constante" do resultado atual. -/
theorem weekProb_constant_when_beta_zero (lam0 E₁ E₂ : ℝ) :
    weekProb lam0 0 E₁ = weekProb lam0 0 E₂ := by
  simp [weekProb]

/-! ## 2. A exposição com um μ não determina a exposição com outro μ

No PeDViS, o que um infectado deposita no piso decai à taxa μ. Um depósito de tamanho
`a` feito `k` dias antes contribui `a · e^(−μk)` para a exposição de hoje. O arquivo
de exposição só traz o total por vaca por dia, sem dizer quando e quanto foi
depositado. Basta então mostrar duas histórias de depósito com o mesmo total em μ₁ e
totais diferentes em μ₂. -/

noncomputable def contribution (μ a : ℝ) (k : ℕ) : ℝ := a * exp (-(μ * k))

/-- Depositar `a` um dia antes ou `a·e^μ₁` dois dias antes dá a mesma exposição com μ₁,
mas exposições diferentes com qualquer outro μ₂. -/
theorem other_mu_not_determined {μ₁ μ₂ a : ℝ} (hμ : μ₁ ≠ μ₂) (ha : a ≠ 0) :
    contribution μ₁ a 1 = contribution μ₁ (a * exp μ₁) 2 ∧
    contribution μ₂ a 1 ≠ contribution μ₂ (a * exp μ₁) 2 := by
  unfold contribution
  push_cast
  constructor
  · rw [mul_assoc, ← Real.exp_add]
    congr 2
    ring
  · intro h
    rw [mul_assoc, ← Real.exp_add] at h
    have h2 := Real.exp_injective (mul_left_cancel₀ ha h)
    exact hμ (by linarith)

/-- A versão com os números do projeto: com μ₁ = 0,07 (o arquivo do Gustavo) e
μ₂ = 0,03 (o melhor μ do modelo por baia), as duas histórias se separam. -/
theorem file_at_0_07_does_not_give_0_03 :
    contribution 0.07 1 1 = contribution 0.07 (exp 0.07) 2 ∧
    contribution 0.03 1 1 ≠ contribution 0.03 (exp 0.07) 2 := by
  have := other_mu_not_determined (μ₁ := 0.07) (μ₂ := 0.03) (a := 1) (by norm_num) one_ne_zero
  simpa using this

/-! ## 3. A escala φ(μ) do modelo por baia (src/barn.py)

φ(μ) = μ² / (μ − 1 + e^(−μ)). Com uma vaca infectada num ambiente limpo, a
contaminação é E(s) = (φ/μ)(1 − e^(−μs)) e a exposição do primeiro dia é a integral
de E de 0 a 1, que dá (φ/μ²)(μ − 1 + e^(−μ)). -/

noncomputable def phi (μ : ℝ) : ℝ := μ ^ 2 / (μ - 1 + exp (-μ))

theorem denominator_pos {μ : ℝ} (h : 0 < μ) : 0 < μ - 1 + exp (-μ) := by
  have := Real.add_one_lt_exp (by linarith : -μ < 0).ne
  linarith

/-- A integral de E no primeiro dia, calculada pela Mathlib. -/
theorem first_day_integral {μ : ℝ} (h : 0 < μ) :
    ∫ s in (0 : ℝ)..1, phi μ / μ * (1 - exp (-(μ * s))) =
      phi μ / μ ^ 2 * (μ - 1 + exp (-μ)) := by
  have hμ : μ ≠ 0 := h.ne'
  have key : ∫ s in (0 : ℝ)..1, exp (-(μ * s)) = (1 - exp (-μ)) / μ := by
    simp_rw [show ∀ s : ℝ, -(μ * s) = (-μ) * s from fun s => by ring]
    rw [intervalIntegral.integral_comp_mul_left Real.exp (neg_ne_zero.mpr hμ)]
    simp only [mul_zero, mul_one, integral_exp, exp_zero, smul_eq_mul]
    field_simp
    ring
  rw [intervalIntegral.integral_const_mul, intervalIntegral.integral_sub, key]
  · simp only [intervalIntegral.integral_const, sub_zero, smul_eq_mul, mul_one]
    field_simp
    ring
  · exact intervalIntegrable_const
  · exact (by fun_prop : Continuous fun x : ℝ => exp (-(μ * x))).intervalIntegrable 0 1

/-- φ(μ) faz a exposição do primeiro dia valer exatamente 1, para qualquer μ > 0. -/
theorem first_day_exposure_is_one {μ : ℝ} (h : 0 < μ) :
    phi μ / μ ^ 2 * (μ - 1 + exp (-μ)) = 1 := by
  have hμ : μ ≠ 0 := h.ne'
  have hd := (denominator_pos h).ne'
  unfold phi
  field_simp

/-! ## 4. Meia-vida da contaminação

A contaminação cai pela metade em log(2)/μ dias. Com μ = 0,01, são entre 69 e 70 dias
(por isso 10 dias de rastreamento não bastam para μ pequeno). -/

theorem half_life_mu_0_01 : 69 < log 2 / 0.01 ∧ log 2 / 0.01 < 70 := by
  have h1 := Real.log_two_gt_d9
  have h2 := Real.log_two_lt_d9
  norm_num at h1 h2 ⊢
  constructor <;> linarith

/-! ## 5. Quando o modelo por baia é igual ao PeDViS

O PeDViS divide o piso em células. Em cada passo, cada célula decai e recebe o que as
vacas infectadas soltaram nela, e a exposição de uma vaca é a contaminação das células
por onde ela passou, pesada pelo tempo em cada uma. -/

/-- Um passo do piso: cada célula decai e recebe o que foi solto nela. -/
noncomputable def cellStep {K : ℕ} (decay gain : ℝ) (E shed : Fin K → ℝ) : Fin K → ℝ :=
  fun c => decay * E c + gain * shed c

/-- Somando as células, o total da baia segue a mesma conta do modelo por baia
(src/barn.py): decai e recebe tudo o que as infectadas soltaram. -/
theorem barn_total_follows_barn_model {K : ℕ} (decay gain : ℝ) (E shed : Fin K → ℝ) :
    ∑ c, cellStep decay gain E shed c = decay * ∑ c, E c + gain * ∑ c, shed c := by
  simp only [cellStep]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]

/-- Se a vaca passa a mesma fração do tempo em cada uma das K células, a exposição dela é a
média da baia: é exatamente o modelo por baia. -/
theorem uniform_use_gives_barn_average {K : ℕ} (E : Fin K → ℝ) :
    ∑ c, (1 / (K : ℝ)) * E c = (∑ c, E c) / K := by
  rw [← Finset.mul_sum]
  ring

/-- Se o uso do espaço não é uniforme, duas vacas na mesma baia e no mesmo dia podem ter
exposições diferentes: aí o modelo por baia não é o PeDViS. Nos dados, a exposição do
PeDViS varia muito entre vacas da mesma baia (verify/claims.py). -/
theorem nonuniform_use_differs :
    ∃ E w₁ w₂ : Fin 2 → ℝ, ∑ c, w₁ c = 1 ∧ ∑ c, w₂ c = 1 ∧ ∑ c, w₁ c * E c ≠ ∑ c, w₂ c * E c :=
  ⟨![1, 0], ![1, 0], ![0, 1], by simp [Fin.sum_univ_two], by simp [Fin.sum_univ_two],
    by simp [Fin.sum_univ_two]⟩

/-! ## 6. Contas feitas com os números dos dados

As contagens vêm de `verify/claims.py`. Aqui só se confere a aritmética das frases. -/

/-- Histórico de dermatite: 20 casos em 110 semanas contra 22 em 2.086. A razão de
riscos fica entre 17 e 18. -/
theorem prior_dd_rate_ratio :
    17 < ((20 : ℚ) / 110) / (22 / 2086) ∧ ((20 : ℚ) / 110) / (22 / 2086) < 18 := by
  norm_num

/-- 20 de 42 casos é menos da metade (47,6%): "quase metade", não "metade". -/
theorem prior_dd_share : (47 : ℚ) / 100 < 20 / 42 ∧ (20 : ℚ) / 42 < 1 / 2 := by
  norm_num

/-- Correção da janela: 163 semanas completas a mais, e 231 semanas em risco e 9
casos ficam de fora quando se exigem os 7 dias. -/
theorem window_fix_arithmetic :
    (1965 : ℤ) - 1802 = 163 ∧ (2196 : ℤ) - 1965 = 231 ∧ (42 : ℤ) - 33 = 9 ∧
    (10 : ℚ) / 100 < 231 / 2196 ∧ (231 : ℚ) / 2196 < 11 / 100 ∧
    (21 : ℚ) / 100 < 9 / 42 ∧ (9 : ℚ) / 42 < 22 / 100 := by
  norm_num

end MathChecks
