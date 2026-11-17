/-!
# Regras de status (e-mail do Gustavo de 16/10/2026, `add_status` em src/daily.py)

Cada vaca-dia é comparado com o último dia observado da mesma vaca (`prev`).
No primeiro dia da vaca não há dia anterior (`prev = none`).

As provas valem para esta cópia em Lean. `Main.lean` imprime a tabela com todas as
combinações possíveis, e `tests/test_lean_status.py` confere que o Python dá a mesma
resposta em cada uma delas.
-/

/-- Status das duas patas traseiras num dia (true = infectada). -/
structure Legs where
  la : Bool
  ra : Bool
deriving Repr

def Legs.infected (l : Legs) : Bool := l.la || l.ra

structure Status where
  infected : Nat
  c : Nat   -- caso novo da vaca
  i : Nat   -- infectada, mas não caso novo
  s : Nat   -- saudável
  ct : Nat  -- patas que viraram caso neste dia
deriving Repr

def b2n (b : Bool) : Nat := if b then 1 else 0

def status : Option Legs → Legs → Status
  | none, now =>
    { infected := b2n now.infected, c := 0, i := b2n now.infected, s := b2n !now.infected, ct := 0 }
  | some p, now =>
    let c := b2n (!p.infected && now.infected)
    { infected := b2n now.infected, c := c, i := b2n (now.infected && c == 0),
      s := b2n !now.infected, ct := b2n (now.la && !p.la) + b2n (now.ra && !p.ra) }

/-- Abre todos os casos: primeiro dia ou não, e as quatro patas (16 + 4 combinações). -/
macro "all_cases" : tactic => `(tactic| (
  intro prev now
  cases now with
  | mk la ra =>
    cases prev with
    | none => cases la <;> cases ra <;> decide
    | some p =>
      cases p with
      | mk pla pra => cases la <;> cases ra <;> cases pla <;> cases pra <;> decide))

/-- Todo vaca-dia está em exatamente um estado: S, I ou C. -/
theorem exactly_one_state : ∀ prev now, (status prev now).s + (status prev now).i + (status prev now).c = 1 := by
  all_cases

/-- S = 1 exatamente quando as duas patas estão saudáveis. -/
theorem healthy_iff : ∀ prev now, (status prev now).s = 1 ↔ now.infected = false := by
  all_cases

/-- INFECTED = I + C. -/
theorem infected_is_i_plus_c : ∀ prev now, (status prev now).infected = (status prev now).i + (status prev now).c := by
  all_cases

/-- Primeiro dia da vaca: nunca é caso novo, nem de vaca nem de pata. -/
theorem first_day_no_case : ∀ now, (status none now).c = 0 ∧ (status none now).ct = 0 := by
  intro now
  cases now with
  | mk la ra => cases la <;> cases ra <;> decide

/-- Caso novo só acontece se a vaca estava saudável no dia anterior e está infectada hoje. -/
theorem case_needs_healthy_before : ∀ p now,
    (status (some p) now).c = 1 → p.infected = false ∧ now.infected = true := by
  intro p now
  cases now with
  | mk la ra => cases p with
    | mk pla pra => cases la <;> cases ra <;> cases pla <;> cases pra <;> decide

/-- Vaca saudável ontem e infectada hoje é sempre caso novo. -/
theorem healthy_then_infected_is_case : ∀ p now,
    p.infected = false → now.infected = true → (status (some p) now).c = 1 := by
  intro p now
  cases now with
  | mk la ra => cases p with
    | mk pla pra => cases la <;> cases ra <;> cases pla <;> cases pra <;> decide

/-- No máximo duas patas viram caso no mesmo dia. -/
theorem at_most_two_legs : ∀ prev now, (status prev now).ct ≤ 2 := by
  all_cases

/-- Todo caso novo de vaca tem pelo menos uma pata nova. -/
theorem case_has_new_leg : ∀ prev now, (status prev now).c = 1 → (status prev now).ct ≥ 1 := by
  all_cases

/-! ## Definição de caso novo com k avaliações saudáveis (`min_healthy` em src/weekly.py)

`h` é a lista das checagens anteriores da vaca, da mais recente para a mais antiga
(true = infectada). -/

/-- As `k` checagens mais recentes existem e são todas saudáveis. -/
def healthyRun : Nat → List Bool → Bool
  | 0, _ => true
  | _ + 1, [] => false
  | k + 1, b :: rest => !b && healthyRun k rest

/-- Com k = 1, é a definição original: saudável na checagem anterior (a véspera). -/
theorem healthyRun_one (b : Bool) (rest : List Bool) : healthyRun 1 (b :: rest) = !b := by
  simp [healthyRun]

/-- Exigir uma avaliação saudável a mais nunca coloca uma semana em risco que antes não estava. -/
theorem healthyRun_mono : ∀ (k : Nat) (h : List Bool),
    healthyRun (k + 1) h = true → healthyRun k h = true
  | 0, _, _ => by simp [healthyRun]
  | _ + 1, [], hk => by simp [healthyRun] at hk
  | k + 1, b :: rest, hk => by
    simp only [healthyRun, Bool.and_eq_true] at hk ⊢
    exact ⟨hk.1, healthyRun_mono k rest hk.2⟩

def isNewCase (k : Nat) (h : List Bool) (now : Bool) : Bool := healthyRun k h && now

/-- Todo caso novo pela definição mais exigente também é caso pela menos exigente:
aumentar k só tira casos, nunca cria. -/
theorem stricter_definition_fewer_cases (k : Nat) (h : List Bool) (now : Bool) :
    isNewCase (k + 1) h now = true → isNewCase k h now = true := by
  simp only [isNewCase, Bool.and_eq_true]
  exact fun ⟨hr, hn⟩ => ⟨healthyRun_mono k h hr, hn⟩

/-- Pata nova sem caso novo de vaca só acontece se a vaca já estava infectada
(segunda pata, ou troca de pata como a vaca 3370 em 29/08/2024). -/
theorem new_leg_without_case_means_already_infected : ∀ p now,
    (status (some p) now).ct ≥ 1 → (status (some p) now).c = 0 → p.infected = true := by
  intro p now
  cases now with
  | mk la ra => cases p with
    | mk pla pra => cases la <;> cases ra <;> cases pla <;> cases pra <;> decide
