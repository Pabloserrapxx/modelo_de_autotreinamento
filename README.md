### 4. 🌊 Aprendizado Incremental (Online Learning)
> **Stack:** Python, River.

Uma demonstração técnica de **Machine Learning em fluxo contínuo (Streaming)**, uma abordagem essencial para cenários onde os dados chegam em alta velocidade e o modelo precisa se adaptar instantaneamente.
* **Adaptação em Tempo Real:** Diferente do aprendizado em lote tradicional, este modelo (Naive Bayes Multinomial via biblioteca **River**) atualiza seus pesos a cada nova instância de dado processada.
* **Human-in-the-loop:** O sistema foi projetado para interagir com o usuário, solicitando feedback sobre as predições e utilizando as correções para realizar o retreino imediato, melhorando a precisão do modelo durante o uso.
