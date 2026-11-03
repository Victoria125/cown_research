# cown_research

Transmissão ambiental de dermatite digital em vacas leiteiras (Dairy Campus), com Gustavo Monti
(WUR). A exposição de cada vaca vem do modelo PeDViS-BDD; o status das patas vem da avaliação
semanal. O objetivo é estimar a taxa de transmissão com um GLM binomial (link cloglog) e comparar
os valores de μ, como em Chang & de Jong (2023, Epidemics 42:100672).

## Como rodar

Os arquivos do e-mail "working files" (16/10/2026) ficam em `data/raw/`, que está fora do git.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\nbstripout.exe --install --attributes .gitattributes  # tira as saídas dos notebooks no commit

.\.venv\Scripts\python.exe -m src.daily 0.07    # data/processed/daily_mu0.07.parquet
.\.venv\Scripts\python.exe -m src.weekly 0.07   # data/processed/weekly_mu0.07.parquet
.\.venv\Scripts\pytest.exe
```

Para outros valores de μ, basta passar mais de um valor: `python -m src.weekly 0.07 0.1`.

## Arquivos gerados

- **daily**: uma linha por vaca por dia, com as colunas do exemplo do Gustavo (`Agent`, `barn`,
  `date`, `Exp`, `LA`, `RA`, `INFECTED`, `C`, `I`, `S`, `N`, `Ct`) e mais: `barn_move` (dia de
  troca de baia), `first_day`, `days_since_prev` (tamanho do buraco antes da linha), `at_risk`
  (vaca saudável no dia anterior) e `legs_at_risk` (patas saudáveis no dia anterior).
- **weekly**: uma linha por vaca por avaliação (quinta), a unidade da regressão. `S` e
  `legs_at_risk` são o status na véspera, `C` e `Ct` os casos novos na avaliação, `exp` a exposição
  somada nos 7 dias antes, `exp_days` quantos desses dias têm exposição e `complete` se são os 7.

## Decisões

- O status é recalculado a partir de LA e RA. As colunas NC, S e I do arquivo de infecção não são
  usadas (a NC marca a semana inteira depois de um caso).
- A avaliação é semanal, às quintas, e o status vale até a seguinte. Nenhuma pata muda de status
  fora de quinta, e `src/weekly.py` dá erro se isso acontecer.
- Linhas repetidas saem. No dia de troca de baia a vaca aparece nas duas baias; fica a de destino
  (a do próximo dia) e o dia é marcado em `barn_move`.
- Primeiro dia da vaca: não há dia anterior, então não há caso novo. Vaca já infectada no primeiro
  dia conta como I.
- Buraco na sequência de dias: compara com o último dia observado.
- Recuperação: a vaca volta a S e pode ter um caso novo depois.
- Pata nova numa vaca já infectada, ou troca de pata no mesmo dia: `Ct = 1`, mas `C = 0`.
- A regressão usa a tabela semanal, não as linhas diárias: os casos só aparecem nas quintas, então
  nos outros dias `C = 0` por construção.
- A exposição da semana soma os 7 dias antes da avaliação; o próprio dia da avaliação fica de fora.
- Vacas que só aparecem no arquivo de exposição ficam de fora (não têm status). Vaca-dias sem
  exposição ficam com `Exp` vazio, para não quebrar a sequência de status.

## Pendências

Perguntas para o Gustavo:

1. Arquivos de exposição dos outros valores de μ.
2. Por que 59 vacas só têm exposição e 15 vacas só têm status de infecção.
3. A falta de exposição não é aleatória: exigir os 7 dias derruba 18% das semanas em risco (394 de
   2.196), mas 26% dos casos (11 de 42). O que faz o rastreamento falhar perto dos casos (vaca
   tratada ou separada, tag sem bateria)? Na regressão, rodar as duas versões: (a) só semanas
   completas; (b) com os dias disponíveis.
4. Três vacas já infectadas no primeiro dia (2941 e 3370 em 30/05/2024; 3733 em 17/10/2024): aqui
   contam como I; a coluna NC conta como caso novo.
5. Vaca 3370 em 29/08/2024: a pata direita sara e a esquerda adoece no mesmo dia. Pode ser troca de
   lado na anotação, e não uma infecção nova. Aqui conta como `Ct = 1`, `C = 0`; a NC conta como
   caso novo.

A conferir no artigo: se o dia da avaliação e o da avaliação anterior entram na soma da exposição.

Limitação conhecida: há casos novos com exposição zero na semana (ordenha, vacas fora do estudo).
