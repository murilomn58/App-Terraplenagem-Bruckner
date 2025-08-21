
# Terraplenagem — LP + Diagrama de Brückner

## Como usar
Deve-se criar um ambiente virtual de preferência (venv)
1. `pip install -r requirements.txt`
2. `streamlit run app.py`

### Modos de entrada
- **Seções (corte + / aterro -)**: informe `pos_m`, `volume_m3` (positivo=corte, negativo=aterro).
- **Ordenadas de Brückner**: informe `estaca` e `Y_m3`; clique **Gerar seções** para transformar
as ordenadas em incrementos (ΔY).

### Restrições dos exercícios
- Use a caixa *Restrições de uso do material* para bloquear cortes que **não** podem ir para aterro
(ex.: `40-60; 200-210`). O app marca `usar_em_aterro=False` para esses cortes, forçando bota‑fora
ou outro destino.

### Saídas
- **Quadro de Distribuição de Terras (por ondas)**: agrega por ondas de compensação.
- **Gráfico do Bruckner** com compensações, empréstimos e bota‑foras.

