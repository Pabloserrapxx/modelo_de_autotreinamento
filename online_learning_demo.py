import river
from river import compose
from river import feature_extraction
from river import naive_bayes

# Criação do Pipeline
# 1. BagOfWords: Transforma o texto em vetores de contagem de palavras (tokenização e contagem)
# 2. MultinomialNB: Classificador Naive Bayes Multinomial, ideal para classificação de texto
model = compose.Pipeline(
    ('vectorizer', feature_extraction.BagOfWords()),
    ('classifier', naive_bayes.MultinomialNB())
)

print("=== Classificador de Texto Online com River ===")
print("O modelo aprende incrementalmente. Digite 'sair' para encerrar.")

while True:
    try:
        # Coleta a entrada do usuário
        texto = input("\nDigite uma frase: ").strip()
        
        if not texto:
            continue
            
        if texto.lower() == 'sair':
            break

        # Realiza a previsão
        # predict_one: Prediz a classe para uma única instância
        predicao = model.predict_one(texto)
        
        # predict_proba_one: Retorna um dicionário com as probabilidades das classes
        probabilidades = model.predict_proba_one(texto)

        # Se o modelo ainda não foi treinado, a previsão pode ser None
        if predicao is None:
            print("O modelo ainda não conhece nenhuma categoria.")
            label_correto = input("Qual é a categoria correta para esta frase? ").strip()
            if label_correto:
                model.learn_one(texto, label_correto)
                print(f"Aprendido! '{texto}' foi classificado como '{label_correto}'.")
            continue

        # Obtém a confiança da previsão atual
        confianca = probabilidades.get(predicao, 0.0)

        print(f"Previsão: {predicao} (Confiança: {confianca:.2%})")

        # Solicita feedback do usuário
        feedback = input("A previsão está correta? (s/n): ").lower().strip()

        if feedback == 'n':
            # Se a previsão estiver errada, solicitamos a categoria correta (Rótulo Real)
            label_correto = input("Qual seria a categoria correta? ").strip()
            
            if label_correto:
                # learn_one: Treina o modelo com o novo exemplo (texto, label)
                # O modelo ajusta seus pesos imediatamente
                model.learn_one(texto, label_correto)
                print("Modelo treinado com o novo exemplo (Correção).")
                
        elif feedback == 's':
            # Se a previsão estiver correta, também podemos treinar para reforçar
            # Isso ajuda o modelo a ter mais certeza no futuro
            model.learn_one(texto, predicao)
            print("Modelo reforçado com o exemplo (Confirmação).")
            
    except KeyboardInterrupt:
        print("\nEncerrando...")
        break
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
