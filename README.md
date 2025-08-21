# Terraplenagem — Momento de Transporte (LP com PuLP)

Aplicativo **Streamlit** que resolve o balanceamento de **cortes/aterros** via programação linear (PuLP + CBC) e gera o **diagrama de Brückner** com destaques visuais e anotações (DMT). Ideal para estudo/relatórios e para consultar rapidamente no celular.

## ✨ Principais recursos
- Entrada por **seções** (corte + / aterro -) **ou** por **ordenadas de Brückner** (Y por estaca).
- Dois estágios de otimização:
  1) **Maximiza** compensação interna corte→aterro (com bloqueios e distância máxima).
  2) **Minimiza custo** completando com **empréstimo** e **bota-fora** (capacidades e custos).
- Gráfico:
  - Curva com reforço **verde** (corte) e **vermelho** (aterro).
  - **Setas alternadas** nas anotações para reduzir poluição visual.
  - Textos compactos (≈ 50%).
  - Marcação com **X** preto nos pontos de entrada.
  - Legenda horizontal sob o eixo X; itens de **Empréstimo** (azul) e **Bota-fora** (vermelho escuro) aparecem quando existem dados.
- Saídas:
  - **Quadro por ondas** (DMT = MT/Vol).
  - **Custo total**.
  - **Download**: CSV do quadro e JSON completo da solução.

## 🧩 Arquivos
- `app.py` — código do app (nome do seu arquivo principal).
- `requirements.txt` — dependências Python.
- `packages.txt` — instala o **CBC** no Streamlit Cloud.
- `runtime.txt` — versão do Python.
- (opcional) `README.md` — este arquivo.

> **requirements.txt (sugestão)**
> ```txt
> streamlit==1.37.0
> pandas
> numpy
> pulp
> matplotlib
> ```

## ▶️ Como rodar localmente
1. Tenha o solver CBC instalado:
   - **Debian/Ubuntu:** `sudo apt-get update && sudo apt-get install -y coinor-cbc`
   - **macOS (Homebrew):** `brew install cbc`
   - **Conda:** `conda install -c conda-forge coincbc`
2. Instale as libs Python:
   ```bash
   pip install -r requirements.txt
