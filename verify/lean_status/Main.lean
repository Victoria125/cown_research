import Status

/-- Sem argumentos, imprime a tabela das regras de status; com `healthy-run`, a tabela da
definição de caso com k avaliações saudáveis. As duas em CSV, para os testes em Python. -/
def main (args : List String) : IO Unit := do
  if args == ["healthy-run"] then
    -- histórico de 1 a 4 checagens, da mais recente para a mais antiga (1 = infectada)
    IO.println "k,history,S"
    for k in [1, 2, 3] do
      for n in [1, 2, 3, 4] do
        for code in List.range (2 ^ n) do
          let h := (List.range n).map fun i => (code >>> i) % 2 == 1
          let text := String.join (h.map fun b => if b then "1" else "0")
          IO.println s!"{k},{text},{b2n (healthyRun k h)}"
    return
  IO.println "first_day,prev_la,prev_ra,la,ra,INFECTED,C,I,S,Ct"
  let bools := [false, true]
  let row (first : Nat) (pla pra : Nat) (now : Legs) (st : Status) : String :=
    s!"{first},{pla},{pra},{b2n now.la},{b2n now.ra},{st.infected},{st.c},{st.i},{st.s},{st.ct}"
  for la in bools do
    for ra in bools do
      let now : Legs := ⟨la, ra⟩
      IO.println (row 1 0 0 now (status none now))
      for pla in bools do
        for pra in bools do
          IO.println (row 0 (b2n pla) (b2n pra) now (status (some ⟨pla, pra⟩) now))
