from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase
import unicodedata

class BPE:
    def __init__(self, vocab_size=30000, min_frequency=2, special_tokens=["<pad>", "<unk>"]):
        # Inicializar el tokenizer con normalización
        self.tokenizer = Tokenizer(models.BPE())
        
        # Configurar normalización (minúsculas y sin acentos)
        self.tokenizer.normalizer = normalizers.Sequence([
            NFD(),  # Descomposición Unicode
            StripAccents(),  # Eliminar acentos
            Lowercase()  # Convertir a minúsculas
        ])
        
        # Usar Whitespace pre-tokenizer en lugar de ByteLevel
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        # self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.is_trained = False

    def train(self, training_files):
        """
        Entrena el tokenizer BPE con una lista de archivos
        """
        self.trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            # continuing_subword_prefix="@@"
        )

        self.tokenizer.train(training_files, trainer=self.trainer)
        self.is_trained = True
        print(f"✅ Tokenizer entrenado con {len(training_files)} archivos")

    def encode(self, text):
        """
        Tokeniza un texto
        """
        if not self.is_trained:
            raise ValueError("Tokenizer no ha sido entrenado")
        
        # Aplicar normalización manual adicional para asegurar
        text = self.normalize_text(text)
        return self.tokenizer.encode(text)

    def encode_batch(self, texts):
        """
        Tokeniza un lote de textos
        """
        if not self.is_trained:
            raise ValueError("Tokenizer no ha sido entrenado")
        
        # Normalizar cada texto
        normalized_texts = [self.normalize_text(text) for text in texts]
        return self.tokenizer.encode_batch(normalized_texts)

    def decode(self, tokens):
        """
        Decodifica tokens a texto
        """
        if not self.is_trained:
            raise ValueError("Tokenizer no ha sido entrenado")
        return self.tokenizer.decode(tokens)

    def normalize_text(self, text):
        """
        Normaliza el texto: minúsculas y sin acentos
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar acentos y caracteres especiales
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        return text

    def get_vocab_size(self):
        """
        Obtiene el tamaño del vocabulario
        """
        return self.tokenizer.get_vocab_size()

    def save(self, filename="bpe_tokenizer.json"):
        """
        Guarda el tokenizer entrenado
        """
        if not self.is_trained:
            raise ValueError("Tokenizer no ha sido entrenado")
        self.tokenizer.save(filename)
        print(f"💾 Tokenizer guardado como: {filename}")

    def load(self, filename="bpe_tokenizer.json"):
        """
        Carga un tokenizer previamente entrenado
        """
        self.tokenizer = Tokenizer.from_file(filename)
        self.is_trained = True
        print(f"📂 Tokenizer cargado desde: {filename}")