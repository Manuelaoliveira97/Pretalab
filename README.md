
# 📸 Classificação e Detecção de Objetos em Imagens com Transformers

Este projeto demonstra como utilizar **modelos pré-treinados da Hugging Face** para realizar **classificação de imagens** e **detecção de objetos**, aplicando técnicas modernas de Visão Computacional com o poder do `transformers`.

---

## 🚀 Tecnologias Utilizadas

- [Transformers](https://huggingface.co/transformers/) - Biblioteca da Hugging Face
- `pipeline` para classificação de imagem e detecção de objetos
- Modelos:
  - `google/vit-base-patch32-384` (ViT - Vision Transformer)
  - `facebook/detr-resnet-50` (DEtection TRansformer)
  - `google/vit-base-patch16-224`
- Python
- [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/) para manipulação de imagens

---

## 📂 Estrutura do Projeto

- **Download da imagem** de demonstração
- **Classificação geral da imagem** usando Vision Transformer (ViT)
- **Detecção de objetos e pessoas** usando o modelo DETR (Facebook)
- **Classificação da pessoa detectada** na imagem

---

## 📥 Requisitos

- Python 3.8+
- `transformers`
- `torch`
- `Pillow`

Instale os pacotes necessários com:

```bash
pip install transformers torch pillow
```

---

## 📷 Exemplo de Execução

### 1. Baixar a imagem:

```python
!wget https://img.freepik.com/fotos-premium/mulher-negra-de-cuidados-com-a-pele-e-creme-para-rosto-detox-organico-facial-e-em-fundo-de-estudio-marrom-garota-afro-americana-e-locao-para-bem-estar-pele-suave-e-clara-para-beleza-e-cosmeticos_590464-122679.jpg
```

### 2. Classificação da Imagem

```python
from transformers import pipeline
from PIL import Image

pipe = pipeline("image-classification", model="google/vit-base-patch32-384", device=0)
imagem = Image.open("mulher-negra-...jpg")
res = pipe(imagem)
```

### 3. Detecção de Objetos e Classificação Focada

```python
detector = pipeline("object-detection", model="facebook/detr-resnet-50")
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

detections = detector(imagem)
person_bbox = [d for d in detections if d['label'] == 'person'][0]

result = classifier(imagem)
```

### 4. Exibição dos Resultados

```python
for item in result:
    print(f"🏷️ {item['label'].upper():<30} | 📊 {item['score']*100:.2f}%")
```

---

## 🎯 Resultado Esperado

O código deve retornar as **principais categorias associadas à imagem** com **probabilidades**, além de detectar objetos/pessoas na foto e aplicar a classificação especificamente à pessoa.

---

## 📌 Observações

- O uso da GPU (`device=0`) melhora significativamente a performance. Certifique-se de que o ambiente (como Google Colab) tenha CUDA disponível.
- Os modelos utilizados são pré-treinados e funcionam bem para tarefas genéricas, mas podem ser substituídos por modelos customizados para domínios específicos.

---

## 🧠 Créditos

- [Hugging Face Models](https://huggingface.co/models)
- [Freepik](https://www.freepik.com) pela imagem de exemplo
