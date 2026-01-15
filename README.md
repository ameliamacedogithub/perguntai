# Perguntai
Trabalho final de conclusão de curso da especialização em Inteligência Artificial Aplicada do IFG

Os comandos a seguir estão prontos para serem executados no Google Colab. Para executar localmente faça os devidos ajustes:

1. Baixar os códigos python do github:
   ```text
    !git clone https://github.com/ameliamacedogithub/perguntai.git
   ```

2. Entre no diretório criado com os arquivos:
  ```python
   %cd perguntai
  ```
 

3. Instalação das bibliotecas através do requirements.txt. Ao finalizar reinicie a sessão:
   ```text
   !pip install -r requirements.txt
   ```


7. Ajuste necessário para resolução de problemas de conflitos de versões das bibliotecas:
   ```text
   # 1. Força a desinstalação de tudo relacionado a langchain para garantir limpeza
   !pip uninstall -y langchain langchain-classic langchain-community langchain-core langchain-openai

   # 2. Instala apenas as versões oficiais e modernas
   !pip install langchain==0.3.0 langchain-community==0.3.0 langchain-openai chromadb flashrank
   ```

Aqui temos duas opções, executar a interface ou realizar o benchmarking

# PerguntAI – Interface

Recupera a chave da OpenAI e Ngrok que cria um tunel para interface Streamlit rodar. O ngrok abre uma nova aba no seu navegador, clique em Visit site. O túnel só é necessário se estiver rodando no Google Colab. Caso contrário !streamlit run interface.py.

```python
import os
from google.colab import userdata
from pyngrok import ngrok

# 1. RECUPERA TODOS OS SECRETS AQUI NA CÉLULA
try:
    # Chaves da OpenAI
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    # Chave do Ngrok
    authtoken = userdata.get('NGROK_AUTHTOKEN')
    ngrok.set_auth_token(authtoken)

    print("✅ Secrets carregados no ambiente.")

except Exception as e:
    print(f"❌ Erro ao carregar secrets: {e}")
    print("Verifique se todas as 2 chaves (OPENAI e NGROK_AUTHTOKEN) estão salvas corretamente.")


# Mata qualquer túnel ngrok que já esteja rodando
ngrok.kill()

# Roda o app Streamlit em background
!nohup streamlit run interface.py &> streamlit.log &

# Espera para o Streamlit iniciar
import time
print("Aguardando o Streamlit iniciar... ⏳")
time.sleep(5)

# Cria o túnel e mostra o link público (agora autenticado!)
public_url = ngrok.connect(8501)
print("=====================================================================================")
print(f"✅ SEU APLICATIVO ESTÁ NO AR! ACESSE AQUI: {public_url}")
print("=====================================================================================")

```
# PerguntAI – Benchmark


1.Execute o runner para iniciar o benchmark
```text
!python runner_benchmark_configs.py
```

# PerguntAI – Análise do Benchmark
Lê o CSV consolidado gerado pelo runner e cria gráficos para:

Comparação entre tecnologias
Comparação entre parâmetros (chunk_size, chunk_overlap, k)
Pré-requisito: ter um arquivo benchmark_consolidado.csv, ou seja, ter rodado `!python runner_benchmark_configs.py`

```text
%run analise.py
```
