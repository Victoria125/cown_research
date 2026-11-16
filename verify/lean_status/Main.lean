import Status

/-- Imprime a tabela de todas as combinações, em CSV, para o teste em Python. -/
def main : IO Unit := do
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
