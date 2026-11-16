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
.\.venv\Scripts\python.exe -m src.glm           # data/processed/glm_*.parquet
.\.venv\Scripts\pytest.exe
```

Para outros valores de μ, basta passar mais de um valor: `python -m src.weekly 0.07 0.1`.
Os resultados ficam em `notebooks/summary.ipynb` (dados) e `notebooks/glm.ipynb` (regressão).

## Arquivos gerados

- **daily**: uma linha por vaca por dia, com as colunas do exemplo do Gustavo (`Agent`, `barn`,
  `date`, `Exp`, `LA`, `RA`, `INFECTED`, `C`, `I`, `S`, `N`, `Ct`) e mais: `barn_move` (dia de
  troca de baia), `first_day`, `days_since_prev` (tamanho do buraco antes da linha), `at_risk`
  (vaca saudável no dia anterior) e `legs_at_risk` (patas saudáveis no dia anterior).
- **weekly**: uma linha por vaca por avaliação (quinta), a unidade da regressão. `S` e
  `legs_at_risk` são o status na véspera, `C` e `Ct` os casos novos na avaliação, `exp` a exposição
  somada nos 7 dias antes (do arquivo de exposição inteiro), `exp_days` quantos desses dias têm
  exposição e `complete` se são os 7.
- **glm_pedvis_mu0.07**: os dois modelos com a exposição do PeDViS-BDD, com as semanas completas e
  com os dias disponíveis.
- **glm_barn_profile**: os dois modelos com a exposição por baia, para cada μ de 0,01 a 10 por dia.

## Modelos

- **Exposição do PeDViS-BDD** (arquivo do Gustavo): por vaca, a partir do rastreamento. Não dá
  para recalcular: na mesma baia e no mesmo dia, as vacas têm exposições muito diferentes.
- **Exposição por baia** (`src/barn.py`): as equações 2 e 3 de Chang & de Jong, com a escala
  φ(μ) = μ² / (μ − 1 + e^−μ). A contaminação da baia cresce com as vacas infectadas e decai à taxa
  μ; cada vaca recebe a exposição da baia dividida pelo número de vacas. Calcula qualquer μ, mas é
  outro modelo: não reproduz o PeDViS-BDD.
- **Regressão** (`src/glm.py`), por semana em risco:
  - sem fundo, `p = 1 − exp(−β·exp)`: o GLM cloglog com offset log(exp) do artigo;
  - com fundo, `p = 1 − exp(−(λ₀·7 + β·exp))`: λ₀ é o risco que não vem do piso.
  Os dois são ajustados por máxima verossimilhança, com IC de β por perfil. μ é escolhido pelo
  AIC; dAIC < 2 é compatível com os dados.

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
  A soma é feita direto nos 7 valores, porque a soma móvel do pandas deixa resíduos de
  arredondamento onde deveria dar zero.
- A janela do PeDViS-BDD soma o arquivo de exposição inteiro, não só os dias do arquivo de infecção:
  a vaca pode sumir do arquivo de infecção por alguns dias com a exposição ainda registrada. O modelo
  por baia só existe nos dias do arquivo de infecção.
- Modelo por baia: a baia começa em equilíbrio com as vacas infectadas do primeiro dia, porque já
  havia infectadas antes do início dos dados. Dias sem vacas na baia contam como zero infectadas.
- O modelo sem fundo não admite caso com exposição zero, então essas semanas ficam de fora. No
  perfil por baia, entram só as semanas com exposição > 0 em todos os μ, para os AIC serem
  comparáveis. O modelo com fundo usa todas as semanas em risco.
- Para comparar o PeDViS-BDD com o modelo por baia, os dois são ajustados nas mesmas semanas.
- Vacas que só aparecem no arquivo de exposição ficam de fora (não têm status). Vaca-dias sem
  exposição ficam com `Exp` vazio, para não quebrar a sequência de status.

## Verificação

Ferramentas para conferir, por outro caminho, tudo o que o pipeline faz:

- `python -m verify.audit` grava `verify/report/audit_report.md` (fora do git, tem IDs de vacas). Confere
  que os arquivos brutos são os anexos do e-mail (sha256), que nenhuma linha some sem explicação, as regras
  contra o exemplo do Gustavo, o status contra as colunas NC/S/I do arquivo, uma reimplementação com laços
  simples, a exposição por baia contra a integração numérica da EDO, a regressão contra o statsmodels e
  contra dados simulados, e como o resultado muda com outras escolhas de janela.
- `verify/lean_status/` prova em Lean 4 as regras de status (S + I + C = 1, caso novo só depois de dia
  saudável, primeiro dia nunca é caso etc.), para as 20 combinações possíveis de patas. `lake build` confere
  as provas; `tests/test_verification.py` confere que o Python dá a mesma resposta que o Lean.
- `tests/test_verification.py` roda as verificações principais com o `pytest`.

Achados da auditoria (16/11/2026):

- **A janela de exposição perdia dias (corrigido).** Ela era montada a partir do arquivo diário, que só tem
  os dias do arquivo de infecção. Quando a vaca sumia desse arquivo, a exposição desses dias ficava de fora,
  mesmo existindo no arquivo de exposição: 165 semanas em risco (2 casos). Agora as semanas completas são
  1.965, não 1.802, com 33 casos em vez de 31. A conclusão não mudou: com fundo, o melhor β continua zero.
- **Com a exposição de 15 a 21 dias antes da avaliação**, o modelo com fundo prefere a exposição ao risco
  constante (ΔAIC ≈ −6, p ≈ 0,002). Pode ser o tempo entre a infecção e a lesão ser vista; várias janelas foram
  testadas, então só vale se houver motivo biológico para esse atraso.
- **O arquivo de infecção se contradiz em 32 vaca-dias** (vacas 2611, 2678 e 4065): LA/RA e S/I/NC dizem
  coisas diferentes. O código usa LA/RA.

## Pendências

Perguntas para o Gustavo:

1. Arquivos de exposição dos outros valores de μ.
2. Por que 59 vacas só têm exposição e 15 vacas só têm status de infecção.
3. A falta de exposição não é aleatória: exigir os 7 dias derruba 11% das semanas em risco (231 de
   2.196), mas 21% dos casos (9 de 42). O que faz o rastreamento falhar perto dos casos (vaca
   tratada ou separada, tag sem bateria)? Na regressão, rodar as duas versões: (a) só semanas
   completas; (b) com os dias disponíveis.
4. Três vacas já infectadas no primeiro dia (2941 e 3370 em 30/05/2024; 3733 em 17/10/2024): aqui
   contam como I; a coluna NC conta como caso novo.
5. Vaca 3370 em 29/08/2024: a pata direita sara e a esquerda adoece no mesmo dia. Pode ser troca de
   lado na anotação, e não uma infecção nova. Aqui conta como `Ct = 1`, `C = 0`; a NC conta como
   caso novo.

A conferir no artigo: se o dia da avaliação e o da avaliação anterior entram na soma da exposição.

Limitação conhecida: há casos novos com exposição zero na semana (ordenha, vacas fora do estudo).
