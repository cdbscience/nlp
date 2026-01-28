from django.db import models
from django.contrib.postgres.fields import ArrayField


class File(models.Model):
    """Armazena informações sobre arquivos processados (PDFs)"""
    name = models.CharField(max_length=255)
    file_path = models.FileField(upload_to='rag_files/')
    file_size = models.BigIntegerField(default=0)  # em bytes
    total_documents = models.IntegerField(default=0)
    is_processed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name


class Document(models.Model):
    """Documentos extraídos de arquivos PDF com embeddings vetorizados"""
    file = models.ForeignKey(File, on_delete=models.CASCADE, related_name='documents', null=True, blank=True)
    title = models.CharField(max_length=255, blank=True)
    content = models.TextField()
    
    # Embedding vetorizado (armazenado como JSON text field)
    embedding_json = models.TextField(default='[]', blank=True)  # Será salvo como JSON
    
    chunk_index = models.IntegerField(default=0)  # Índice do chunk dentro do arquivo
    word_count = models.IntegerField(default=0)
    char_count = models.IntegerField(default=0)
    
    # Metadata em JSON
    metadata = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['file', 'chunk_index']),
            models.Index(fields=['-created_at']),
        ]

    def __str__(self):
        if self.file:
            return f"{self.file.name} - Chunk {self.chunk_index}"
        return f"Document - Chunk {self.chunk_index}"
    
    def get_embedding(self):
        """Retorna o embedding como lista de floats"""
        import json
        try:
            return json.loads(self.embedding_json) if self.embedding_json else []
        except:
            return []
    
    def set_embedding(self, embedding):
        """Salva o embedding como JSON"""
        import json
        self.embedding_json = json.dumps(embedding)


class Query(models.Model):
    """Queries com embeddings vetorizados para busca"""
    question = models.TextField()
    answer = models.TextField(blank=True)
    
    # Embedding da query para busca vetorizada
    embedding_json = models.TextField(default='[]', blank=True)
    
    # Resultados da busca
    relevant_documents = models.JSONField(default=list, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.question[:50]
    
    def get_embedding(self):
        """Retorna o embedding como lista de floats"""
        import json
        try:
            return json.loads(self.embedding_json) if self.embedding_json else []
        except:
            return []
    
    def set_embedding(self, embedding):
        """Salva o embedding como JSON"""
        import json
        self.embedding_json = json.dumps(embedding)
